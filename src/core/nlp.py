"""
High-level natural language processing module for message-like (emails, comments, posts) input.

Supports automatic language detection, word tokenization and stemming for `'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'`.

© 2023 - Aurélien Pierre
"""

import random
import regex as re
import os
import sys
from multiprocessing import Pool

from collections import Counter

import gensim
from gensim.models.callbacks import CallbackAny2Vec

import joblib

import numpy as np

import nltk
from nltk.classify import SklearnClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from rank_bm25 import BM25Okapi

from core.patterns import *
from core.utils import get_models_folder, typography_undo, guess_date
from core.language import *


def guess_language(string: str) -> str:
    """Basic language guesser based on stopwords detection. Stopwords are the most common words of a language: for each language, we count how many stopwords we found and return the language having the most matches. It is not perfect but the rutime vs. accuracy ratio is good.
    """

    tokenizer = RegexpTokenizer(r'\w+|[\d\.\,]+|\S+')
    words = {token.lower() for token in tokenizer.tokenize(string)}
    scores = []
    for lang in STOPWORDS_DICT:
        scores.append(len(words.intersection(STOPWORDS_DICT[lang])))

    index_max = max(range(len(scores)), key=scores.__getitem__)
    return list(STOPWORDS_DICT.keys())[index_max]


# Find English dates like `01 Jan 20` or `01 Jan. 2020` but avoid capturing adjacent time like `12:08`.
# Find French dates like `01 Jan 20` or `01 Jan. 2020` but avoid capturing adjacent time like `12:08`.
TEXT_DATES = re.compile(r"([0-9]{1,2})? (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|jan|fév|mar|avr|mai|jui|jui|aou|sep|oct|nov|déc|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\.?( [0-9]{1,2})?( [0-9]{2,4})(?!:)",
                        flags=re.IGNORECASE | re.MULTILINE)
BASE_64 = re.compile(r"((?:[A-Za-z0-9+\/]{4}){64,}(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=)?)")
BB_CODE = re.compile(r"\[(img|quote)[a-zA-Z0-9 =\"]*?\].*?\[\/\1\]")
MARKUP = re.compile(r"(?:\[|\{|\<)([^\n\r]+?)(?:\]|\}|\>)")
USER = re.compile(r"(\S+)?@(\S+)|(user\-?\d+)")
REPEATED_CHARACTERS = re.compile(r"(.)\1{9,}")
UNFINISHED_SENTENCES = re.compile(r"(?<![?!.;:])\n\n")
MULTIPLE_DOTS = re.compile(r"\.{2,}")
MULTIPLE_DASHES = re.compile(r"-{1,}")
MULTIPLE_QUESTIONS = re.compile(r"\?{1,}")
ORDINAL_FR = re.compile(r"n° ?([0-9]+)")

# Trailing/leading lost punctuation resulting from broken tokenization through composed words
TRAILING = re.compile(r"^((-|\.|\,)(?=[a-zéèàêâîôûïüäëö]))|((?<=[a-zéèàêâîôûïüäëö])(-|\.|\,))", flags=re.IGNORECASE)

# pronoms/déterminants + apostrophes + mot
FRANCAIS = re.compile(r"(?<=^|[\s\(\[\:])(j|t|s|l|d|qu|m)\'(?=[a-zéèàêâîôûïüäëö])", flags=re.IGNORECASE)

class Tokenizer():
    def clean_whitespaces(self, string:str) -> str:
        # Collapse multiple newlines and spaces
        string = MULTIPLE_LINES.sub("\n\n", string)
        string = MULTIPLE_SPACES.sub(" ", string)

        # Paragraphs (ended with \n\n) that don't have ending punctuation should have one.
        string = UNFINISHED_SENTENCES.sub(".\n\n", string)

        return string.strip()


    def prefilter(self, string:str, meta_tokens:bool = True) -> str:
        """Tokenizers split words based on unsupervised machine-learned models. Sometimes, they work weird.
        For example, in emails and user handles like `@user`, they would split `@` and `user` as 2 different tokens,
        making it impossible to detect usernames in single tokens later.

        To avoid that, we replace data of interest by meta-tokens before the tokenization, with regular expressions.
        """
        if meta_tokens:
            for key, value in self.pipeline.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                string = key.sub(value, string)

        for key, value in self.abbreviations.items():
            string = string.replace(key, value)

        return self.clean_whitespaces(string)


    def lemmatize(self, string:str) -> str:
        return self.lemmatizer.lemmatize(string)


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
        string = word.strip("-,:'^ ")

        # Remove leading/trailing characters that may result from faulty tokenization
        string = TRAILING.sub("", string)

        if len(string) == 0:
            # empty string
            return None

        if string in self.meta_tokens:
            # Input is lowercase, need to fix that for meta tokens.
            return string.upper()

        if "_" in string or "<" in string or ">" in string or "\\" in string or "=" in string or "~" in string or "#" in string:
            # Technical stuff, like markup/code leftovers and such
            return None

        # Lemmatizer : canonical form of the word, if found
        # Note : still produces natural language words
        string = self.lemmatize(string)

        if re.match(r"^[a-zéèàêâîôûïüäëö]{1}$", string):
            return None

        if string in REPLACEMENTS:
            string = REPLACEMENTS[string]

        if string in STOPWORDS:
            return None

        if meta_tokens:
            for key, value in self.pipeline.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                if key.search(string):
                    return value.strip()

        return string.strip()


    def tokenize_sentence(self, sentence: str, language: str) -> list[str]:
        """Split a sentence into normalized word tokens and meta-tokens."""
        tokens = [self.normalize_token(token.lower(), language)
                  for token in nltk.word_tokenize(sentence, language=language)]
        tokens = [item for item in tokens
                  if item is not None]

        if not tokens or len(tokens) == 1:
            # Tokenization seems to fail on single-word queries, try again without it
            tokens = [self.normalize_token(sentence.lower(), "english")]

        return tokens


    def split_sentences(self, document: str, language: str) -> list[str]:
        """Split a document into sentences using an unsupervised machine learning model.

        Arguments:
            text (str): the paragraph to break into sentences.
            language (str): the language of the text, used to select what pre-trained model will be used.
        """
        return nltk.sent_tokenize(document, language=language)


    def tokenize_document(self, document:str, language:str = None) -> list[str]:
        """Cleanup and tokenize a document or a sentence as an atomic element, meaning we don't split it into sentences. Use this either for search-engine purposes (into a document's body) or if the document is already split into sentences.

        Note:
            the language is detected internally if not provided as an optional argument. When processing a single sentence extracted from a document, instead of the whole document, it is more accurate to run the language detection on the whole document, ahead of calling this method, and pass on the result here.

        Arguments:
            document (str): the text of the document to tokenize
            language (str): the language of the document. Will be internally inferred if not given.

        Returns:
            tokens (list[str]): a 1D list of normalized tokens and meta-tokens.
        """
        document = typography_undo(document)

        if language is None:
            language = guess_language(document)

        document = self.prefilter(document)
        return self.tokenize_sentence(document, language)


    def tokenize_per_sentence(self, document: str) -> list[list[str]]:
        """Cleanup and tokenize a whole document as a list of sentences, meaning we split it into sentences before tokenizing. Use this to train a Word2Vec (embedding) model so each token is properly embedded into its syntactic context.

        Note:
            the language is detected internally.

        Returns:
            tokens (list[list[str]]): a 2D list of sentences (1st axis), each containing a list of normalizel tokens and meta-tokens (2nd axis).
        """
        clean_text = typography_undo(document)
        language = guess_language(clean_text)
        clean_text = self.prefilter(clean_text)
        return [self.tokenize_sentence(sentence, language)
                for sentence in self.split_sentences(clean_text, language)]


    def __init__(self, pipeline:dict[re.Pattern: str] = None, abbreviations:dict[str: str] = None, lemmatizer = None):
        if pipeline is None:
            self.pipeline = {
                MULTIPLE_DOTS: "...",
                MULTIPLE_DASHES: "-",
                MULTIPLE_QUESTIONS: "?",
                # Remove non-punctuational repeated characters like xxxxxxxxxxx, or =============
                # (redacted text or ASCII line-like separators)
                REPEATED_CHARACTERS: ' ',
                BB_CODE: " ",
                MARKUP: r" \1 ",
                BASE_64: ' _BASE64_ ',
                # Remove french contractions: m', j', qu' etc.
                FRANCAIS: "",
                # Anonymize users/emails and prevent tokenizers from splitting @ from the username
                USER: " _USER_ ",
                # URLs and IPs - need to go before pathes
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
                # Local pathes - get everything with / or \ left over by the previous
                # Need to go after dates for the slash date format
                PATH_PATTERN: ' _PATH_ ',
                # Unit numbers/quantities
                EXPOSURE: " _EXPOSURE_ ",
                SENSIBILITY: " _SENSIBILITY_ ",
                LUMINANCE: " _LUMINANCE_ ",
                FILE_SIZE: " _FILESIZE_ ",
                DISTANCE: " _DISTANCE_ ",
                WEIGHT: " _WEIGHT_ ",
                ANGLE: " _ANGLE_ ",
                FREQUENCY: " _FREQUENCY_ ",
                PERCENT: " _PERCENT_ ",
                GAIN: " _GAIN_ ",
                TEMPERATURE: " _TEMPERATURE_ ",
                # Numéro/ordinal numbers
                ORDINAL: " _ORDINAL_ ",
                ORDINAL_FR: " _ORDINAL_ ",
                # Numerical : prices and resolutions
                PRICE_US_PATTERN: " _PRICE_ ",
                PRICE_EU_PATTERN: " _PRICE_ ",
                RESOLUTION_PATTERN: " _RESOLUTION_ ",
                # Remove HEX hashes, like IDs and commit names
                HASH_PATTERN: ' _HASH_ ',
                # Remove numbers
                NUMBER_PATTERN: ' _NUMBER_ ',
            }
        else:
            self.pipeline = pipeline

        self.meta_tokens = [value.lower().strip()
                            for value in self.pipeline.values()
                            if value.startswith(" _") and value.endswith("_ ")]

        if abbreviations is None:
            self.abbreviations = ABBREVIATIONS
        else:
            self.abbreviations = abbreviations

        if lemmatizer is None:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = lemmatizer


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
    def __init__(self, sentences: list[str], name: str = "word2vec", vector_size: int = 300, epochs: int = 200, window: int = 5, tokenizer: Tokenizer = None):
        """Train, re-train or retrieve an existing word2vec word embedding model

        Arguments:
            name (str): filename of the model to save and retrieve. If the model exists already, we automatically load it. Note that this will override the `vector_size` with the parameter defined in the saved model.
            vector_size (int): number of dimensions of the word vectors
            epochs (int): number of iterations of training for the machine learning. Small corpora need 2000 and more epochs. Increases the learning time.
        """
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer()

        self.pathname = get_models_folder(name)
        self.vector_size = vector_size
        print(f"got {len(sentences)} pieces of text")

        # training = [tokenize_sentences(sentence, language=language) for sentence in sentences]
        sentences = set(sentences)
        processes = os.cpu_count()
        with Pool(processes=processes) as pool:
            training: list[list[list[str]]] = pool.map(self.tokenizer.tokenize_per_sentence, sentences, chunksize=1)

        print("tokenization done")

        # Flatten the first dimension of the list of list of list of strings :
        training = [sentence for text in training for sentence in text]
        print(f"got {len(training)} sentences")

        # Dump words to a file to detect stopwords
        words = [word for sentence in training for word in sentence]
        counts = Counter(words)

        # Sort words by frequency
        counts = dict(sorted(counts.items(), key=lambda counts: counts[1]))
        with open(get_models_folder("stopwords"), 'w', encoding='utf8') as f:
            for key, value in counts.items():
                f.write(f"{key}: {value}\n")
        print("stopwords saved")

        loss_logger = LossLogger()
        super().__init__(training, vector_size=vector_size, window=window, min_count=5, workers=processes, epochs=epochs, ns_exponent=-0.5, sample=0.001, callbacks=[loss_logger], compute_loss=True, sg=1)
        print("training done")

        self.save(self.pathname)
        print("saving done")


    @classmethod
    def load_model(cls, name: str):
        """Load a trained model saved in `models` folders"""
        return cls.load(get_models_folder(name))


    def get_wordvec(self, word: str, embed:str = "IN") -> np.array:
        """Return the vector associated to a word, through a dictionnary of words.

        Arguments:
            word (str): the word to convert to a vector.
            embed (str): `IN` or `OUT`. The default, `IN` usis the input embedding matrix (gensim.Word2Vec.wv), useful to vectorize queries and documents for classification training. `OUT` uses `gensim.Word2Vec.syn1neg`, useful for the dual-space embedding scheme, to train search engines.

        References:
            A Dual Embedding Space Model for Document Ranking (2016), Bhaskar Mitra, Eric Nalisnick, Nick Craswell, Rich Caruana
            https://arxiv.org/pdf/1602.01137.pdf


        Returns:
            the nD vector.
        """
        if word and word in self.wv:
            if embed == "OUT":
                vec = self.syn1neg[self.wv.key_to_index[word]]
            elif embed == "IN":
                vec = self.wv[word]
            else:
                raise ValueError("Invalid option")

            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0. else vec
        else:
            return np.zeros(self.wv.vector_size)


    def get_features(self, tokens: list, embed:str = "IN") -> dict:
        """Extract word features from the text of `post`.

        We use meta-features like date, prices, time, number that discard the actual value but retain the property.
        That is, we don't care about the actual date, price or time, we only care that there is a date, price or time.
        Meta-features are tagged with random hashes as to not be mistaken with text.

        For everything else, we use the actual words.

        Arguments:
            tokens (list[str]): if given, we discard internal tokenization and normalization and directly use this list of tokens. The need to be normalized already.
            num_features (int): the number of dimensions of the featureset. This is vector size used in the `Word2Vec` model.
            wv (gensim.models.KeyedVector): the dictionnary mapping words with vectors,
            syn1neg (np.array): the W_out matrix for word embedding, in the Dual Embedding Space Model. [^1] If not provided, embedding uses the default W_in matrix. W_out is better to vectorize documents for search-engine purposes.
            language (str): the language used to detect dates and detect words separators used in tokenization. Supports `"french"` and `"english"`.

        [^1]: https://arxiv.org/pdf/1602.01137.pdf

        Return:
            (dict): dictionnary of features, where keys are initialized with the positional number of vector elements and their value, plus the optional external data.
        """
        features = np.zeros(self.vector_size)
        i = len(tokens)

        for token in tokens:
            vector = self.get_wordvec(token, embed)
            features += vector

        # Finish the average calculation (so far, only summed)
        if i > 0:
            features /= i

        # NLTK models take dictionnaries of features as input, so bake that.
        # TODO: in Indexer, we need to revert that to numpy. Fix the API to avoid this useless step
        #return dict(enumerate(features))
        return features

class Classifier(SklearnClassifier):
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


class Indexer(SklearnClassifier):
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
        print(f"Init. Got {len(data_set)} items.")

        if word2vec:
            self.word2vec = word2vec
        else:
            raise ValueError("wv needs to be a dictionnary-like map")

        # Remove duplicated content if any.
        # For example, translated content when non-existent translations are inited with the original language.
        cleaned_set = {}
        for post in data_set:
            cleaned_set.setdefault(post["content"], []).append(post)

        data_set = []
        for value in cleaned_set.values():
            # Lazy trick : measure memory size of each duplicate and keep the heaviest
            # assuming it's the most "complete"
            sizes = [sys.getsizeof(elem) for elem in value]
            idx_max = sizes.index(max(sizes))
            data_set.append(value[idx_max])

        # Posts too short contain probably nothing useful
        data_set = [post for post in data_set if len(self.word2vec.tokenizer.clean_whitespaces(post["content"])) > 250]

        print(f"Cleanup. Got {len(data_set)} remaining items.")

        # The database of web pages with limited features.
        # Keep only pages having at least 250 letters in their content
        self.index = {post["url"]:
                      {"title": post["title"],
                       "excerpt": post["excerpt"] if post["excerpt"] else post["content"][0:250],
                       "date": guess_date(post["date"]) if post["date"] else None,
                       "url": post["url"],
                       "language": guess_language(post["content"]),
                       "sentences": list({s
                                          for s in set(
                                              self.word2vec.tokenizer.split_sentences(
                                                  self.word2vec.tokenizer.clean_whitespaces(
                                                      post["content"]),
                                                  guess_language(post["content"])))
                                          if len(s) > 60})
                       }
                      for post in data_set}

        # Build the training set from webpages
        training_set = [Data(post["title"] + "\n\n" + post["content"], post["url"])
                        for post in data_set]

        # Prepare the ranker for BM25 : list of tokens for each document
        with Pool() as pool:
            ranker_docs: list = pool.map(self.tokenize_parallel, training_set)

        # Turn tokens into features
        with Pool() as pool:
            docs: list = pool.map(self.get_features_parallel, ranker_docs)

        self.vectors_all = np.array(docs)
        self.all_norms = np.linalg.norm(self.vectors_all, axis=1)
        self.urls = [post.label for post in training_set]

        # Values from https://arxiv.org/pdf/1602.01137.pdf, p.6, section 3.3
        self.ranker = BM25Okapi(ranker_docs, k1=1.7, b=0.95)

        # Garbage collection to avoid storing in the saved model stuff we won't need anymore
        # Can't do that anymore if we need to embed document at run time
        #del self.syn1neg

        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"))


    def get_features_parallel(self, tokens: list[str]) -> tuple[str, str]:
        """Thread-safe call to `.get_features()` to be called in multiprocessing.Pool map"""
        # Language doesn't matter, tokenization and normalization are done already
        return self.word2vec.get_features(tokens, embed="OUT")


    def tokenize_parallel(self, post: Data) -> list[str]:
        tokens = self.word2vec.tokenizer.tokenize_document(post.text)
        return tokens


    @classmethod
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        model = joblib.load(get_models_folder(name) + ".joblib")
        if isinstance(model, Indexer):
            return model
        else:
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))


    def vectorize_query(self, post: str) -> tuple[np.array, float, list[str]]:
        """Prepare a text search query: cleanup, tokenize and get the centroid vector.

        Returns:
            tuple[vector, norm, tokens]
        """
        tokenized_query = self.word2vec.tokenizer.tokenize_document(post)

        if not tokenized_query:
            return np.array([]), 0., []

        # Get the the centroid of the word embedding vector
        vector = self.word2vec.get_features(tokenized_query)
        norm = np.linalg.norm(vector)
        norm = 1.0 if norm == 0.0 else norm

        return vector, norm, tokenized_query


    def rank(self, post: str|tuple, method: str = "ai", filter_callback: callable = None, **kargs) -> list:
        """Apply a label on a post based on the trained model."""

        if isinstance(post, tuple):
            # Input is vectorized already, unpack the tuple
            vector = post[0]
            norm = post[1]
            tokens = post[2]
        elif isinstance(post, str):
            vector, norm, tokens = self.vectorize_query(post)
        else:
            raise TypeError("The argument should be either a (vector, norm) tuple or a string")

        # Compute the cosine similarity of centroids between query and documents,
        # then aggregate the ranking from BM25+ to it for each URL.
        # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
        if method.lower() == "ai":
            norm *= len(tokens)
            aggregates = 0.97 * np.dot(self.vectors_all, vector) / (norm * self.all_norms) + 0.03 * self.ranker.get_scores(tokens)
        elif method.lower() == "fuzzy":
            aggregates = self.ranker.get_scores(tokens)
        else:
            pass

        results = zip(self.urls, np.nan_to_num(aggregates))

        if filter_callback is None:
            results = {(url, similarity) for url, similarity in results if similarity > 0.}
        else:
            results = {(url, similarity) for url, similarity in results if similarity > 0. and filter_callback(url, **kargs)}

        # Return the 100 most similar documents by (url, similarity)
        return sorted(results, key=lambda x:x[1], reverse=True)[0:100]


    def get_page(self, url:str) -> dict:
        """Retrieve the requested page data object from the index by url.

        Warning:
            For performance, it doesn't check if the url exists in the index. This is no issue if you feed it the output of `self.rank()`.
        """
        return self.index[url]


    def get_snippet(self, page:dict, query: tuple):
        vector = query[0]
        norm = query[1]
        tokens_query = query[2]

        sentences = page['sentences']

        vectors_all = []
        penalties = []

        for sentence in sentences:
            tokens = self.word2vec.tokenizer.tokenize_document(sentence)

            # Count how many times the tokens of the query appear in the tokens of the sentence.
            # Since the similarity relies on averaging word vectors, a short section title containing only the query keyword
            # would cheat the metric and get a very high score, despite being irrelevant.
            # This penalty ensures word typically associated through embedding count enough in the mix too.
            penalty = 0
            for token in tokens_query:
                penalty += tokens.count(token)

            penalties.append(np.sqrt(penalty) if penalty > 0. else 1.)
            vectors_all.append(self.word2vec.get_features(tokens, embed="OUT"))

        if vectors_all:
            vectors_all = np.array(vectors_all)
            all_norms = np.linalg.norm(vectors_all, axis=1)
            similarities = np.dot(vectors_all, vector) / (norm * all_norms) / np.array(penalties)
            similarities = np.nan_to_num(similarities)

            # Return the n most similar sentences in the document in descending order of similarity
            # That is, if we have at least n sentences
            num_elem = min(similarities.size, 5)
            index_best = list(np.argpartition(similarities, -num_elem)[-num_elem:])

            if len(index_best) > 0:
                return [(sentences[i], similarities[i]) for i in sorted(index_best) if similarities[i]]
            else:
                return []
        else:
            return []


    def get_related(self, post: str|tuple, n:int = 15) -> list:
        """Get the n closest keywords from the query."""

        if isinstance(post, tuple):
            # Input is vectorized already, unpack the tuple
            vector = post[0]
            norm = post[1]
            tokens = post[2]
        elif isinstance(post, str):
            vector, norm, tokens = self.vectorize_query(post)
        else:
            raise TypeError("The argument should be either a (vector, norm) tuple or a string")

        # wv.similar_by_vector returns a list of (word, distance) tuples
        return [elem[0] for elem in self.word2vec.wv.similar_by_vector(vector, topn=n) if elem[0] not in tokens]
