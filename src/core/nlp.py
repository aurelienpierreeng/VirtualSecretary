"""
High-level natural language processing module for message-like (emails, comments, posts) input.

Supports automatic language detection, word tokenization and stemming for `'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'`.

© 2023 - Aurélien Pierre
"""

from enum import IntEnum
import random
import regex as re
import os
from multiprocessing import Pool, current_process
import multiprocessing

from collections import Counter

import gensim
from gensim.models.callbacks import CallbackAny2Vec

import joblib
import pickle

import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from rank_bm25 import BM25Okapi

from .patterns import *
from .utils import get_models_folder, typography_undo, guess_date, clean_whitespaces, timeit
from .language import *


def guess_language(string: str) -> str:
    """Basic language guesser based on stopwords detection.

    Stopwords are the most common words of a language: for each language, we count how many stopwords we found and return the language having the most matches. It is accurate for paragraphs and long documents, not so much for short sentences.

    Returns:
        2-letters ISO-something language code.
    """

    tokenizer = RegexpTokenizer(r'\w+|[\d\.\,]+|\S+')
    words = {token for token in tokenizer.tokenize(string)}
    scores = []
    for lang in STOPWORDS_DICT:
        scores.append(len(words.intersection(STOPWORDS_DICT[lang])))

    index_max = max(range(len(scores)), key=scores.__getitem__)
    return list(STOPWORDS_DICT.keys())[index_max]

class Tokenizer():
    characters_cleanup: dict[re.Pattern: str] = {
        MULTIPLE_DOTS: "...",
        MULTIPLE_DASHES: "-",
        MULTIPLE_QUESTIONS: "?",
        # Remove non-punctuational repeated characters like xxxxxxxxxxx, or =============
        # (redacted text or ASCII line-like separators)
        REPEATED_CHARACTERS: ' ',
        BB_CODE: " ",
        MARKUP: r" \1 ",
        BASE_64: " "}
    """Dictionnary of regular expressions (keys) to find and replace by the provided strings (values). Cleanup repeated characters, including ellipses and question marks, leftover BBcode and XML markup, base64-encoded strings and French pronominal contractions (e.g "me + a" contracted into "m'a")."""

    internal_meta_tokens: dict[re.Pattern: str] = {
        HASH_PATTERN_FAST: "_HASH_",
        NUMBER_PATTERN_FAST: "_NUMBER_"}
    """Dictionnary of regular expressions (keys) to find in full-tokens and replace by meta-tokens. Use simplified regex patterns for performance."""

    def prefilter(self, string:str, meta_tokens:bool = True) -> str:
        """Tokenizers split words based on unsupervised machine-learned models. Sometimes, they work weird.
        For example, in emails and user handles like `@user`, they would split `@` and `user` as 2 different tokens,
        making it impossible to detect usernames in single tokens later.

        To avoid that, we replace data of interest by meta-tokens before the tokenization, with regular expressions.
        """

        for key, value in self.characters_cleanup.items():
            # Note: since Python 3.8 or so, dictionnaries are ordered.
            # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
            string = key.sub(value, string)

        for key, value in self.abbreviations.items():
            string = string.replace(key, value)


        if meta_tokens:
            for key, value in self.meta_tokens_pipe.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                try:
                    string = key.sub(value, string, timeout=15)
                except TimeoutError:
                    print("Meta-token detection timed out on %s with:\n%s" % (key, string))

        return string


    def lemmatize(self, word: str) -> str:
        """Find the root (lemma) of words to help topical generalization."""
        lemmas = {
            # Simplify double consonants. They are irregular, people misspell them and they vary between languages.
            DOUBLE_CONSONANTS: r"\1",
            # Remove final "s" or "es" as a plural mark.
            # Ex : lenses -> len, lens -> len
            PLURAL_S: "",
            # Replace British spelling of -our words by American spelling
            # Ex : colour -> color, behaviour -> behavior,
            # but tour -> tour, pour -> pour
            BRITISH_OUR: "or",
            # Remove final -ity and -ite from substantives:
            # Ex : activity -> activ, activite -> activ
            # but cite -> cite, city -> city
            # Caveat : due to upstream removal of accents, medical conditions in French
            # based on inflammations (meningite, hepatite, bronchite, vulvite) will get removed there too.
            SUBSTANTIVE_ITY: "",
            # Remove final "e" as feminine mark (in French)
            # Ex : lense -> lens, profile -> profil, manage -> manag, capitale -> capital
            FEMININE_E: "",
            # Remove -tor, -teur, -tric,
            # Ex : acteur -> act, actor -> act, actric -> act
            FEMININE_TRICE: "t",
            # Remove -ing from participle present, maybe used as substantives
            # Ex : being -> be, acting -> act, managing -> manag
            # DISABLED: too much meaning lost.
            # word = PARTICIPLE_ING.sub("", word)
            # Remove -ed from adjectives
            # Ex : acted -> act, managed -> manag, aplied -> apli
            ADJECTIVE_ED: "",
            # Remove -ment and -ement from substantives and adverbs
            # Ex : management -> manag, imediatement -> imediat
            ADVERB_MENT: "",
            # Remove -tion and -sion
            # Ex : action -> act, application -> applicat, comision -> comis
            SUBSTANTIVE_TION: r"\1",
            # Remove -ism and -ist from substantives
            # Ex : feminism -> femin, feminist -> femin, artist -> art
            # but exist -> exist
            # Caveat : consist -> consi
            SUBSTANTIVE_IST: "",
            # Remove -at
            # Note : may finish the job from previous step for -ation
            # Ex : reliquat -> reliqu, optimisat -> optimis, neutralizat -> neutraliz
            SUBSTANTIVE_AT: "",
            # Remove -tif and -tiv from adjectives
            # Note : final -e was already removed above.
            # Ex : actif -> act, activ -> act, optimisation -> optimisat, neutralization -> neutralizat
            ADJECTIVE_TIF: "t",
            # Replace final -y by -i.
            # Note : This is because applied -> aplied -> apli,
            # while apply -> aply, so finish aply -> apli for consistency.
            SUBSTANTIVE_Y: "i",
            # Replace final -er if there is more than 4 letters before
            # Ex : optimizer -> optimiz, instaler -> instal, player -> play, higher -> high
            # but power -> power, her -> her, there -> ther -> ther
            # Caveat : master -> mast and lower -> lower
            STUFF_ER: "",
            # Replace -iz/-iz by -is/-ys for American English, to unify with British and French
            # Ex : optimiz -> optimis, neutraliz -> neutralis, analyz -> analys
            # Caveat : size -> siz -> sis
            VERB_IZ: r"\1s",
            # Replace -eur by -or
            # Ex: serveur -> servor, curseur -> cursor, meileur -> meilor
            SUBSTANTIVE_EUR: "or",
            # We might be tempted to remove -al here, as in
            # profesional, tribal, analytical. Problem is collision with
            # apeal, instal, overal, reveal, portal, gimbal.
            # Leave it as-is and let the embedding figure it out.
            # Replace -iqu by -ic
            # This has the double goal of making French closer to English, but
            # also to stem verbs the same as nouns
            # Ex : aplique -> aplic (same as aplication -> aplicat -> aplic)
            # politiqu -> politic, expliqu -> explic
            SUBSTANTIVE_IQU: "i",
        }

        for key, value in lemmas.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                # Allow only 10 s for each pattern because we run on individual tokens here.
                try:
                    word = key.sub(value, word, timeout=10)
                except TimeoutError:
                    print("Lemmatization timed out on %s with:\n%s" % (key, word))

        return word


    def normalize_token(self, word: str, language: str, meta_tokens: bool = True):
        """Return normalized, lemmatized and stemmed word tokens, where dates, times, digits, monetary units and URLs have their actual value replaced by meta-tokens designating their type. Stopwords ("the", "a", etc.), punctuation etc. is replaced by `None`, which should be filtered out at the next step.

        Arguments:
            word (str): tokenized word in lower case only.
            language (str): the language used to detect dates. Supports `"french"`, `"english"` or `"any"`.
            vocabulary (dict): a `token: list` mapping where `token` is the stemmed token and `list` stores all words from corpus which share this stem. Because stemmed tokens are not user-friendly anymore, this vocabulary can be used to build a reverse mapping `normalized token` -> `natural language keyword` for GUI.

        Examples:
            `10:00` or `10 h` or `10am` or `10 am` will all be replaced by a `_TIME_` meta-token.
            `feb`, `February`, `feb.`, `monday` will all be replaced by a `_DATE_` meta-token.
        """
        string = word.strip("?!#=+-,:;'\"^*./`()[]{}& \n\r\t")

        if len(string) == 0 or " " in string or "\n" in string:
            # empty string or
            # tokenizer failed to split tokens on spaces
            return None

        if string in REPLACEMENTS:
            string = REPLACEMENTS[string]

        if string.upper() in self.meta_tokens:
            return string.upper()

        if "_" in string or "<" in string or ">" in string or "\\" in string or "=" in string or "~" in string:
            # Technical stuff, like markup/code leftovers and such
            return None

        # Last chance of identifying meta-tokens in an atomic way
        if meta_tokens:
            for key, value in self.internal_meta_tokens.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                if key.match(string, timeout=10):
                    return value

        # Lemmatize / Stem
        string = self.lemmatize(string)

        # Check if string is a stopword
        return None if ((self.stopwords is not None) and (string in self.stopwords)) else string


    def tokenize_sentence(self, sentence: str, language: str, meta_tokens: bool = True) -> list[str]:
        """Split a sentence into normalized word tokens and meta-tokens.

        Arguments:
            sentence: the input single sentence.
            language: the language string to be used by the tokenizer. It needs to be one of those supported by the module [core.nlp][].
            meta_tokens: find meta-tokens through regular expressions and replace them in the text. This helps tokenization to keep similar objects together, especially dates that would otherwise be splitted.

        Returns:
            tokens (list[str]): the list of normalized tokens.
        """
        tokens = [self.normalize_token(token, language, meta_tokens=meta_tokens)
                  for token in nltk.word_tokenize(str(sentence), language=language)]
        tokens = [item for item in tokens if isinstance(item, str)]

        if len(tokens) == 0:
            # Tokenization seems to fail on single-word queries, try again without it
            tokens = [self.normalize_token(sentence, "english")]

        return [item for item in tokens if isinstance(item, str)]


    def split_sentences(self, document: str, language: str) -> list[str]:
        """Split a document into sentences using an unsupervised machine learning model.

        Arguments:
            text (str): the paragraph to break into sentences.
            language (str): the language of the text, used to select what pre-trained model will be used.
        """
        return nltk.sent_tokenize(str(document), language=language)


    def tokenize_document(self, document:str, language:str = None, meta_tokens: bool = True) -> list[str]:
        """Cleanup and tokenize a document or a sentence as an atomic element, meaning we don't split it into sentences. Use this either for search-engine purposes (into a document's body) or if the document is already split into sentences. The document text needs to have been prepared and cleaned, which means :

        - lowercased (optional but recommended) with `str.lower()`,
        - translated from Unicode to ASCII (optional but recommended) with [utils.typography_undo()][],
        - cleaned up for sequences of whitespaces with [utils.cleanup_whitespaces()][]

        Note:
            the language is detected internally if not provided as an optional argument. When processing a single sentence extracted from a document, instead of the whole document, it is more accurate to run the language detection on the whole document, ahead of calling this method, and pass on the result here.

        Arguments:
            document (str): the text of the document to tokenize
            language (str): the language of the document. Will be internally inferred if not given.

        Returns:
            tokens (list[str]): a 1D list of normalized tokens and meta-tokens.
        """
        if language is None:
            language = guess_language(document)

        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return self.tokenize_sentence(clean_text, language, meta_tokens=meta_tokens)


    def tokenize_per_sentence(self, document: str, meta_tokens: bool = True) -> list[list[str]]:
        """Cleanup and tokenize a whole document as a list of sentences, meaning we split it into sentences before tokenizing. Use this to train a Word2Vec (embedding) model so each token is properly embedded into its syntactic context. The document text needs to have been prepared and cleaned, which means :

        - lowercased (optional but recommended) with `str.lower()`,
        - translated from Unicode to ASCII (optional but recommended) with [utils.typography_undo()][],
        - cleaned up for sequences of whitespaces with [utils.cleanup_whitespaces()][]

        Note:
            the language is detected internally.

        Returns:
            tokens: a 2D list of sentences (1st axis), each containing a list of normalizel tokens and meta-tokens (2nd axis).
        """
        # TODO: prefilter n-grams ?
        self.processed_items += 1
        p = current_process()
        if self.processed_items % 50 == 0 and p._identity:
            print("P %i processed %i over %i" % (p._identity[0], self.processed_items, self.num_items))

        language = guess_language(document)
        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return [self.tokenize_sentence(sentence, language, meta_tokens=meta_tokens)
                for sentence in self.split_sentences(clean_text, language)]

    def __init__(self,
                 meta_tokens: dict[re.Pattern: str] = None,
                 abbreviations: dict[str: str] = None,
                 stopwords: list[str] = None):
        """Pre-processing pipeline and tokenizer, splitting a string into normalized word tokens.

        Arguments:
            meta_token: the pipeline of regular expressions to replace with meta-tokens. Keys must be `re.Pattern` declared with `re.compile()`, values must be meta-tokens assumed to be nested in underscores. The pipeline dictionnary will be processed in the order of declaration, which relies on using Python >= 3.7 (making `dict` ordered by default). If not provided, it is inited by default with a pipeline suitable for bilingual English/French language processing on technical writings (see notes).
            abbreviations (dict[str: str]): pipeline of abbreviations to replace, as `to_replace: replacement` dictionnary. Will be processed in order of declaration.
        """
        if meta_tokens is None:
            self.meta_tokens_pipe = {
                # Anonymize users/emails and prevent tokenizers from splitting @ from the username
                USER: " _USER_ ",
                # URLs and IPs - need to go before pathes
                # URLs can contain IPs, so process them first
                URL_PATTERN: ' _URL_ ',
                IP_PATTERN: ' _IP_ ',
                # File types - need to go before pathes
                CODE_PATTERN: ' _CODEFILE_ ',
                DATABASE_PATTERN: ' _DATABASEFILE_ ',
                IMAGE_PATTERN: ' _IMAGEFILE_ ',
                DOCUMENT_PATTERN: ' _DOCUMENTFILE_ ',
                TEXT_PATTERN: " _TEXTFILE_ ",
                ARCHIVE_PATTERN: " _ARCHIVEFILE_ ",
                EXECUTABLE_PATTERN: " _BINARYFILE_ ",
                # Dates
                TEXT_DATES: " _DATE_ ",
                DATE_PATTERN:" _DATE_ ",
                TIME_PATTERN: " _TIME_ ",
                # Key/mouse shortcuts
                SHORTCUT_PATTERN: " _SHORTCUT_ ",
                # Note : f4 can be interpreted as diaph aperture or key
                # Shortcuts need to be processed first.
                # Local pathes - get everything with / or \ left over by the previous
                # Need to go after dates for the slash date format
                PATH_PATTERN: ' _PATH_ ',
                # Cleanup long sequences of numbers to help the next filters to run within decent runtimes
                NUMBER_SEQUENCE_PATTERN: "123456789",
                # Unit numbers/quantities
                EXPOSURE: " _EXPOSURE_ ",
                SENSIBILITY: " _SENSIBILITY_ ",
                LUMINANCE: " _LUMINANCE_ ",
                FILE_SIZE: " _FILESIZE_ ",
                DISTANCE: " _DISTANCE_ ",
                WEIGHT: " _WEIGHT_ ",
                TEMPERATURE: " _TEMPERATURE_ ",
                ANGLE: " _ANGLE_ ",
                FREQUENCY: " _FREQUENCY_ ",
                PERCENT: " _PERCENT_ ",
                GAIN: " _GAIN_ ",
                DIAPHRAGM: " _APERTURE_ ",
                PIXELS: " _PIXELS_ ",
                # Numéro/ordinal numbers
                ORDINAL: " _ORDINAL_ ",
                ORDINAL_FR: " _ORDINAL_ ",
                # Numerical : prices and resolutions
                PRICE_PATTERN: " _PRICE_ ",
                RESOLUTION_PATTERN: " _RESOLUTION_ ",
                # Remove numbers
                NUMBER_PATTERN: ' _NUMBER_ ',
                # Remove HEX hashes, like IDs and commit names
                HASH_PATTERN: ' _HASH_ ',
                # In-words dashes : replace by space
                DASHES: " ",
                # French words contractions : expand for generality
                FRANCAIS: r"\1e ",
                # Weird domains that are not really URLs : replace dots by space
                MEMBERS_PATTERN: " ",
                # Missing/partial/invalid file pathes
                PARTIAL_PATH_REGEX: " _PATH_ ",
                # Word alternatives like and/or : replace slash by space
                ALTERNATIVES: " ",
            }
        else:
            self.meta_tokens_pipe = meta_tokens

        self.meta_tokens = {value.strip()
                            for value in self.meta_tokens_pipe.values()
                            if value.startswith(" _") and value.endswith("_ ")}

        if abbreviations is None:
            self.abbreviations = ABBREVIATIONS
        else:
            self.abbreviations = abbreviations

        self.stopwords = stopwords

        self.num_items = 1
        self.processed_items = 0


class Data():
    def __init__(self, text: str, label: str):
        """Represent an item of tagged training data.

        Arguments:
            text (str): the content to label, which will be vectorized
            label (str): the category of the content, which will be predicted by the model
        """
        self.text = text
        self.label = label


class LossLogger(CallbackAny2Vec):
    '''Output loss at each epoch'''
    def __init__(self):
        self.epoch = 1
        self.losses = []

    def on_epoch_begin(self, model):
        print(f'Epoch: {self.epoch}', end='\t')

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'  Loss: {loss}')
        self.epoch += 1


class Word2Vec(gensim.models.Word2Vec):
    def __init__(self, sentences: list[str], name: str = "word2vec", vector_size: int = 300, epochs: int = 200, window: int = 5, min_count=5, sample=0.0005, tokenizer: Tokenizer = None):
        """Train, re-train or retrieve an existing word2vec word embedding model

        Arguments:
            name (str): filename of the model to save and retrieve. If the model exists already, we automatically load it. Note that this will override the `vector_size` with the parameter defined in the saved model.
            vector_size (int): number of dimensions of the word vectors
            epochs (int): number of iterations of training for the machine learning. Small corpora need 2000 and more epochs. Increases the learning time.
            window (int): size of the token collocation window to detect
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer()

        self.pathname = get_models_folder(name)
        self.vector_size = vector_size
        print(f"got {len(sentences)} pieces of text")

        # training = [tokenize_sentences(sentence, language=language) for sentence in sentences]
        random.shuffle(sentences)
        sentences = set(sentences)
        processes = os.cpu_count()

        self.tokenizer.num_items = len(sentences) / processes
        self.tokenizer.processed_items = 0
        training: list[list[list[str]]] = []

        with Pool(processes=processes) as pool:
            runner = pool.imap(self.tokenizer.tokenize_per_sentence, sentences, chunksize=1)
            run = True
            while run:
                try:
                    # Allow 1 minute to complete tokenization.
                    # On Intel Xeon, most documents complete in under a second.
                    # Some documents lead to catastrophic backtracking with the
                    # URL regex pattern and need to be stopped the hard way.
                    doc = runner.next(timeout=60)
                    training.append(doc)
                except multiprocessing.TimeoutError:
                    print("timeout")
                except StopIteration:
                    run = False

        print("tokenization done")

        # Flatten the first dimension of the list of list of list of strings :
        training = [sentence for text in training for sentence in text]
        print(f"got {len(training)} sentences")

        # Dump words to a file to detect stopwords
        words = [word for sentence in training for word in sentence]
        print(f"got {len(words)} individual words")

        counts = Counter(words)
        print(f"got {len(counts)} unique words")

        # Sort words by frequency
        counts = dict(sorted(counts.items(), key=lambda counts: counts[1]))
        with open(get_models_folder("stopwords"), 'w', encoding='utf8') as f:
            for key, value in counts.items():
                f.write(f"{key}: {value}\n")
        print("stopwords saved")

        loss_logger = LossLogger()
        super().__init__(training, vector_size=vector_size, window=window, min_count=min_count, workers=processes, epochs=epochs, ns_exponent=0.75, sample=sample, callbacks=[loss_logger], compute_loss=True, sg=0, max_final_vocab=100000)
        print("training done")

        self.save(self.pathname)
        print("saving done")


    @classmethod
    def load_model(cls, name: str):
        """Load a trained model saved in `models` folders"""
        return cls.load(get_models_folder(name))


    def get_word(self, word: str) -> str | None:
        """Find out if word is in dictionary, optionnaly attempting spell-checking if not found.

        Arguments:
            word: word to find

        Returns:
            (str | None):
                - the original word if found in dictionnary,
                - `None` if both previous conditions were not matched.
        """
        if word:
            if word in self.wv:
                # Word exists in dictionnary
                return word
            else:
                return None

        return None


    def get_wordvec(self, word: str, embed:str = "IN") -> np.ndarray[np.float32] | None:
        """Return the vector associated to a word, through a dictionnary of words.

        Arguments:
            word: the word to convert to a vector.
            embed:
                - `IN` uses the input embedding matrix [gensim.models.Word2Vec.wv][], useful to vectorize queries and documents for classification training.
                - `OUT` uses the output embedding matrix [gensim.models.Word2Vec.syn1neg], useful for the dual-space embedding scheme, to train search engines. [^1]

        [^1]: A Dual Embedding Space Model for Document Ranking (2016), Bhaskar Mitra, Eric Nalisnick, Nick Craswell, Rich Caruana https://arxiv.org/pdf/1602.01137.pdf

        Returns:
            the nD vector if the word was found in the dictionnary, or `None`.
        """
        x = self.get_word(word)

        # The word or its correction are found in DB
        if x is not None:
            if embed == "OUT":
                vec = self.syn1neg[self.wv.key_to_index[x]].astype(np.float32)
            elif embed == "IN":
                vec = self.wv[x].astype(np.float32)
            else:
                raise ValueError("Invalid option")

            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0. else vec
        else:
            return None


    def get_features(self, tokens: list[str], embed: str = "IN") -> np.ndarray[np.float32]:
        """Calls [core.nlp.Word2Vec.get_wordvec][] over a list of tokens and returns a single vector representing the whole list.

        Arguments:
            tokens: list of text tokens.
            embed: see [core.nlp.Word2Vec.get_wordvec][]

        Returns:
            the centroid of word embedding vectors associated with the input tokens (aka the average vector), or the null vector if no word from the list was found in dictionnary.
        """
        features = np.zeros(self.vector_size, dtype=np.float32)
        i = 0

        for token in tokens:
            vector = self.get_wordvec(token, embed=embed)
            if vector is not None:
                features += vector
                i += 1

        # Finish the average calculation (so far, only summed)
        if i > 0:
            features /= i

        return features

class Classifier(nltk.classify.SklearnClassifier):
    def __init__(self,
                 training_set: list[Data],
                 name: str,
                 word2vec: Word2Vec,
                 validate: bool = True,
                 variant: str = "svm"):
        """Handle the word2vec and SVM machine-learning

        Arguments:
            training_set (list[Data]): list of Data elements. If the list is empty, it will try to find a pre-trained model matching the `path` name.
            path : path to save the trained model for reuse, as a Python joblib.
            name (str): name under which the model will be saved for la ter reuse.
            word2vec (Word2Vec): the instance of word embedding model.
            validate (bool): if `True`, split the `feature_list` between a training set (95%) and a testing set (5%) and print in terminal the predictive performance of the model on the testing set. This is useful to choose a classifier.
            variant (str):
                - `svm`: use a Support Vector Machine with a radial-basis kernel. This is a well-rounded classifier, robust and stable, that performs well for all kinds of training samples sizes.
                - `linear svm`: uses a linear Support Vector Machine. It runs faster than the previous and may generalize better for high numbers of features (high dimensionality).
                - `forest`: Random Forest Classifier, which is a set of decision trees. It runs about 15-20% faster than linear SVM but tends to perform marginally better in some contexts, however it produces very large models (several GB to save on disk, where SVM needs a few dozens of MB).
            features (int): the number of model features (dimensions) to retain. This sets the number of dimensions for word vectors found by word2vec, which will also be the dimensions in the last training layer.
        """
        print("init")

        if word2vec:
            self.word2vec = word2vec
        else:
            raise ValueError("wv needs to be a dictionnary-like map")

        # Single-threaded variant :
        # new_featureset = [(get_features(post.text, self.vector_size, self.wv, language=self.language), post.label)
        #                   for post in training_set]

        # Multi-threaded variant :
        with Pool() as pool:
            new_featureset: list = pool.map(self.get_features_parallel, training_set)

        print("feature set :", len(new_featureset), "/", len(training_set))

        # If validation is on, split the set into a training and a test subsets
        if validate:
            size = int(len(new_featureset) * 0.05)
        else:
            size = 0

        random.shuffle(new_featureset)  # shuffle in-place
        train_set, test_set = new_featureset[size:], new_featureset[:size]

        if variant == "linear svm":
            # C is regularization, decrease below 1 if noisy training input.
            # Here, noise got filtered already in word2vec, so no need and 15 is empiric optimum.
            classifier = SVC(kernel="linear", probability=True, C=1)
        elif variant == "svm":
            # C is regularization, decrease below 1 if noisy training input.
            # Here, noise got filtered already in word2vec, so no need and 15 is empiric optimum.
            classifier = SVC(kernel="rbf", probability=True,
                             C=15, gamma='scale')
        elif variant == "forest":
            # n_jobs = -1 means use all available cores
            classifier = RandomForestClassifier(n_jobs=os.cpu_count())
        else:
            raise ValueError("Invalid classifier")

        super().__init__(classifier)
        self.train(train_set)
        print("model trained")

        if validate:
            print("accuracy against test set:", nltk.classify.accuracy(self, test_set))

        print("accuracy against train set:", nltk.classify.accuracy(self, train_set))

        # We don't need the heavy syn1neg dictionnary of Word2Vec
        del self.word2vec.syn1neg

        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"))


    def get_features_parallel(self, post: Data) -> tuple[str, str]:
        """Thread-safe call to `.get_features()` to be called in multiprocessing.Pool map"""
        tokens = self.word2vec.tokenizer.tokenize_document(post.text)
        features = dict(enumerate(self.word2vec.get_features(tokens)))
        return (features, post.label)


    @classmethod
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        model = joblib.load(get_models_folder(name) + ".joblib")
        if isinstance(model, nltk.SklearnClassifier):
            return model
        else:
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))


    def classify(self, post: str) -> str:
        """Apply a label on a post based on the trained model."""
        tokens = self.word2vec.tokenizer.tokenize_document(post)
        features = dict(enumerate(self.word2vec.get_features(tokens)))
        return super().classify(features)


    def prob_classify(self, post: str) -> tuple[str, float]:
        """Apply a label on a post based on the trained model and output the probability too."""
        tokens = self.word2vec.tokenizer.tokenize_document(post)
        features = dict(enumerate(self.word2vec.get_features(tokens)))

        # This returns a weird distribution of probabilities for each label that is not quite a dict
        proba_distro = super().prob_classify(features)

        # Build the list of dictionnaries like `label: probability`
        output = {i: proba_distro.prob(i) for i in proba_distro.samples()}

        # Finally, return label and probability only for the max proba of each element
        return (max(output, key=output.get), max(output.values()))

class search_methods(IntEnum):
    """Search methods available"""
    AI = 1
    FUZZY = 2
    GREP = 3

class Indexer():
    @timeit()
    def __init__(self,
                 data_set: list,
                 name: str,
                 word2vec: Word2Vec):
        """Search engine based on word similarity.

        Arguments:
            training_set (list): list of Data elements. If the list is empty, it will try to find a pre-trained model matching the `path` name.
            path : path to save the trained model for reuse, as a Python joblib.
            name (str): name under which the model will be saved for la ter reuse.
            word2vec (Word2Vec): the instance of word embedding model.
            validate (bool): if `True`, split the `feature_list` between a training set (95%) and a testing set (5%) and print in terminal the predictive performance of the model on the testing set. This is useful to choose a classifier.
            variant (str):
                - `svm`: use a Support Vector Machine with a radial-basis kernel. This is a well-rounded classifier, robust and stable, that performs well for all kinds of training samples sizes.
                - `linear svm`: uses a linear Support Vector Machine. It runs faster than the previous and may generalize better for high numbers of features (high dimensionality).
                - `forest`: Random Forest Classifier, which is a set of decision trees. It runs about 15-20% faster than linear SVM but tends to perform marginally better in some contexts, however it produces very large models (several GB to save on disk, where SVM needs a few dozens of MB).
            features (int): the number of model features (dimensions) to retain. This sets the number of dimensions for word vectors found by word2vec, which will also be the dimensions in the last training layer.
        """
        if word2vec:
            self.word2vec = word2vec
        else:
            raise ValueError("wv needs to be a dictionnary-like map")

        # Shuffle the dataset prior to multithreading to even the probability that
        # one process hangs significantly longer than the others
        random.shuffle(data_set)
        processes = os.cpu_count()
        print(f"Init. Got {len(data_set)} items.")

        # The database of web pages with limited features.
        for post in data_set:
            self.create_index(post)

        self.index = {post["url"]: post for post in data_set}
        del data_set # gargage collection to spare RAM

        print("index built")

        # List of tokens/meta-tokens and vector translation for each document
        docs: list = []
        with Pool(processes=processes) as pool:
            runner = pool.imap(self.tokenize_parallel, list(self.index.values()), chunksize=20)
            run = True
            while run:
                try:
                    # Allow 1 minute to complete tokenization.
                    # On Intel Xeon, most documents complete in under a second.
                    # Some documents lead to catastrophic backtracking with the
                    # URL regex pattern and need to be stopped the hard way.
                    doc = next(runner)
                    docs.append(doc)
                except multiprocessing.TimeoutError:
                    print("timeout")
                except StopIteration:
                    run = False

        self.vectors_all = np.array([doc[1] for doc in docs], dtype=np.float32)
        self.all_norms = np.linalg.norm(self.vectors_all, axis=1)

        print("vectorization done")

        # Extract the 5 most relevant topics of each doc
        for i, key in enumerate(self.index):
            vector, norm, tokens = self.vectorize_query(docs[i][0])
            if len(tokens) > 0:
                topics = self.get_related((vector, "", []), n=5)
                self.index[key]["topics"] = topics
            else:
                self.index[key]["topics"] = []

        # Values from https://arxiv.org/pdf/1602.01137.pdf, p.6, section 3.3
        self.ranker = BM25Okapi([doc[0] for doc in docs], k1=1.7, b=0.95)

        # garbage collection for RAM use
        del docs

        # From there we only need the input embedding matrix, ditch the output one to spare RAM
        del self.word2vec.syn1neg

        # Remove the parsed text variant for the same reason.
        for post in self.index.values():
            del post["parsed"]

        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib.bz2"), compress=9, protocol=pickle.HIGHEST_PROTOCOL)


    def create_index(self, post: dict):
        if not post["excerpt"] or len(post["excerpt"]) < 600:
            post["excerpt"] = str(post["content"])[0:min(len(post["content"]), 600)]
        else:
            post["excerpt"] = str(post["excerpt"])

        if "datetime" not in post:
            # Reuse data set by crawler.Deduplicator if available
            post["datetime"] = guess_date(str(post["date"])) if post["date"] else None

        if "parsed" not in post:
            # Reuse data set by crawler.Deduplicator if available
            post["content"] = clean_whitespaces(str(post["content"]))
            post["parsed"] = typography_undo(post["content"].lower())

        post["category"] = str(post["category"]) if "category" in post else None
        post["title"] = str(post["title"])
        return post


    def tokenize_parallel(self, post: dict) -> list[str]:
        content = typography_undo(str(post["title"]).lower())

        if post["h1"]:
            content += "\n\n" + typography_undo("\n\n".join(list(post["h1"])).lower())

        if post["h2"]:
            content += "\n\n" + typography_undo("\n\n".join(list(post["h2"])).lower())

        content += "\n\n" + post["parsed"]

        tokens = self.word2vec.tokenizer.tokenize_document(content, meta_tokens=True)
        features = self.word2vec.get_features(tokens, embed="OUT")

        return (tokens, features)


    @classmethod
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        try:
            model = joblib.load(get_models_folder(name) + ".joblib")
            if isinstance(model, Indexer):
                return model
        except:
            model = joblib.load(get_models_folder(name) + ".joblib.bz2")
            if isinstance(model, Indexer):
                return model
            else:
                raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))


    def tokenize_query(self, query:str, language: str = None, meta_tokens: bool = True) -> list[str]:
        return self.word2vec.tokenizer.tokenize_document(query, language=language, meta_tokens=meta_tokens)


    def vectorize_query(self, tokenized_query: list[str]) -> tuple[np.ndarray, float, list[str]]:
        """Prepare a text search query: cleanup, tokenize and get the centroid vector.

        Returns:
            tuple[vector, norm, tokens]
        """

        if not tokenized_query:
            return np.array([]), 0., []

        # Get the the centroid of the word embedding vector
        vector = self.word2vec.get_features(tokenized_query, embed="IN")
        norm = np.linalg.norm(vector)
        norm = 1.0 if norm == 0.0 else norm

        return vector, norm, tokenized_query


    def rank_grep(self, query: re.Pattern|str) -> np.ndarray:
        if not (isinstance(query, str) or isinstance(query, re.Pattern)):
            raise ValueError("Wrong query type (%s) for GREP ranking method. Should be string or regular expression pattern" % type(query))

        results = np.array([len(re.findall(query, document["content"], timeout=60))
                            for document in self.index.values()], dtype=np.float64)
        max_rank = np.amax(results)
        if max_rank > 0.: results /= max_rank
        return results


    def rank_fuzzy(self, tokens: list[str]) -> np.ndarray:
        if not isinstance(tokens, list):
            raise ValueError("Wrong query type (%s) for FUZZY ranking method. Should be a list of strings." % type(tokens))

        return self.ranker.get_scores(tokens)


    def rank_ai(self, query: tuple) -> np.ndarray:
        if not isinstance(query, tuple):
            raise ValueError("Wrong query type (%s) for AI ranking method. Should be a `(vector, norm, tokens)` tuple" % type(query))

        vector = query[0]
        norm = query[1]
        tokens = query[2]

        # Compute the cosine similarity of centroids between query and documents,
        # then aggregate the ranking from BM25+ to it for each URL.
        # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
        norm *= len(tokens)
        return 0.97 * np.dot(self.vectors_all, vector) / (norm * self.all_norms) + 0.03 * self.ranker.get_scores(tokens)


    def rank(self, query: str|tuple|re.Pattern, method: search_methods,
             filter_callback: callable = None, pattern: str | re.Pattern = None, **kargs) -> list[tuple[str, float]]:
        """Apply a label on a post based on the trained model.

        Arguments:
            query (str | tuple | re.Pattern): the query to search. `re.Pattern` is available only with the `grep` method.
            method (str): `ai`, `fuzzy` or `grep`. `ai` use word embedding and meta-tokens with dual-embedding space, `fuzzy` uses meta-tokens with BM25Okapi stats model, `grep` uses direct string and regex search.
            filter_callback (callable): a function returning a boolean to filter in/out the results of the ranker.
            pattern: optional pattern/text search to add on top of AI search
            **kargs: arguments passed as-is to the `filter_callback`

        Returns:
            list: the list of best-matching results as (url, similarity) tuples.
        """
        # Note : match needs at least Python 3.10
        match method:
            case search_methods.AI:
                aggregates = self.rank_ai(query)
            case search_methods.FUZZY:
                aggregates = self.rank_fuzzy(query)
            case search_methods.GREP:
                aggregates = self.rank_grep(query)
            case _:
                raise ValueError("Unknown ranking method (%s)" % method)

        results = zip(self.index.keys(), # URLs
                      np.nan_to_num(aggregates)) # similarity metric

        # Filter out documents NOT matching the pattern
        if pattern:
            matches = [True if re.findall(pattern, document["content"], flags=re.IGNORECASE, timeout=60) else False
                       for document in self.index.values()]
            matches = zip(results, matches)
            results = [result_tuple for result_tuple, match in matches if match]

        # Filter out documents based on URL filter if provided
        if filter_callback:
            results = [(url, similarity) for url, similarity in results if filter_callback(url, **kargs)]

        return sorted(set(results), key=lambda x:x[1], reverse=True)


    def get_page(self, url:str) -> dict:
        """Retrieve the requested page data object from the index by url.

        Warning:
            For performance's sake, it doesn't check if the url exists in the index.
            This is no issue if you feed it the output of `self.rank()` but mind that otherwise.
        """
        return self.index[url]

    def get_related(self, post: tuple, n: int = 15) -> list:
        """Get the n closest keywords from the query."""

        if not isinstance(post, tuple):
            raise TypeError("The argument should be either a (vector, norm) tuple or a string")

        vector = post[0]
        tokens = post[2]

        # wv.similar_by_vector returns a list of (word, distance) tuples
        return [elem[0] for elem in self.word2vec.wv.similar_by_vector(vector, topn=n) if elem[0] not in tokens]
