"""
High-level natural language processing module for message-like (emails, comments, posts) input.

Supports automatic language detection, word tokenization and stemming for `'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'`.

© 2023 - Aurélien Pierre
"""

import random
import re
import os
import sys
from multiprocessing import Pool

import gensim
import joblib

import numpy as np

import nltk
from nltk.classify import SklearnClassifier
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from rank_bm25 import BM25Okapi

from core.patterns import DATE_PATTERN, TIME_PATTERN, URL_PATTERN, IMAGE_PATTERN, CODE_PATTERN, DOCUMENT_PATTERN, TEXT_PATTERN, ARCHIVE_PATTERN, EXECUTABLE_PATTERN, PATH_PATTERN
from core.utils import get_models_folder, typography_undo

#nltk.download('punkt')
#nltk.download('stopwords')

# The set of languages supported at the same time by NLTK tokenizer, stemmer and stopwords data is not consistent.
# We build the least common denominator here, that is languages supported in the 3 modules.
# See SnowballStemmer.languages and stopwords.fileids()
_supported_lang = {'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'}

# Day/month tokens and their abbreviations
DATES = dict.fromkeys(_supported_lang, {})

DATES["english"] = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "aug", "sept", "oct", "nov", "dec",
    "jan.", "feb.", "mar.", "apr.", "aug.", "sept.", "oct.", "nov.", "dec.",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun.",
}

DATES["french"] =  {
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
    "jan", "fév", "mar", "avr", "jui", "juil", "aou", "sept", "oct", "nov", "déc",
    "jan.", "fév.", "mar.", "avr.", "jui.", "juil.", "aou.", "sept.", "oct.", "nov.", "déc.",
    "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
    "lun", "mar", "mer", "jeu", "ven", "sam", "dim",
    "lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.",
}


# Static dict of stopwords for language detection, inited from NLTK corpus
STOPWORDS_DICT = { language: list(stopwords.words(language)) for language in stopwords.fileids() if language in _supported_lang }

# Some stopwords are missing from the corpus for some languages, add them
STOPWORDS_DICT["french"] += ["ça", "ceci", "cela", "tout", "tous", "toutes", "toute",
                             "plusieurs", "certain", "certaine", "certains", "certaines",
                             "meilleur", "meilleure", "meilleurs", "meilleures", "plus",
                             "aujourd'hui", "demain", "hier", "tôt", "tard",
                             "salut", "bonjour", "va", "aller", "venir", "viens", "vient", "viennent", "vienne",
                             "oui", "non",
                             "gauche", "droite", "droit", "haut", "bas", "devant", "derrière", "avant", "après",
                             "clair", "claire",
                             "sûr", "sûre", "sûrement",
                             "cordialement", "salutations",
                             "qui", "que", "quoi", "dont", "où", "pourquoi", "comment", "duquel", "auquel", "lequel", "auxquels", "auxquelles", "lesquelles",
                             "dehors", "hors", "chez", "avec", "vers", "tant", "si", "de",
                             "à", "travers", "pour", "contre", "sans", "afin"]
STOPWORDS_DICT["english"] += ["best", "better", "more", "all", "every",
                              "some", "any", "many", "few", "little",
                              "today", "tomorrow", "yesterday", "early", "late", "earlier", "later",
                              "hi", "hello", "good", "morning", "go", "come", "coming", "going",
                              "yes", "no", "yeah",
                              "left", "right", "ahead", "top", "bottom", "before", "behind", "front", "after",
                              "clear",
                              "sure", "surely",
                              "greetings",
                              "who", "which", "where", "whose", "why", "what", "how", "that",
                              "out", "by", "at", "with", "toward", "long", "as", "if", "of",
                              "through", "for", "against", "without"]
STOPWORDS_DICT["german"] += ["best", "beste", "besten", "besser", "mehr", "alle", "ganz", "ganze",
                             "mehrere", "etwa", "etwas", "manche", "klein", "groß",
                             "heute", "morgen", "gestern", "früh", "spät", "früher", "später",
                             "hallo", "guten", "tag", "geht", "gehen", "kom", "kommen", "komt",
                             "ja", "nein",
                             "links", "rechts", "gerade", "oben", "unten", "vor", "hinter", "nach",
                             "werde", "werden", "wurde", "würde", "stimmt", "klar",
                             "sicher", "sicherlich",
                             "herzlich", "herzliche", "grüße",
                             "wer", "wo", "wann", "wenn", "als", "ob", "was", "warum",
                             "aus", "bei", "mit", "nach", "seit", "von", "zu",
                             "durch", "für", "gegen", "ohne", "um"]

# Build a dict of sets
STOPWORDS_DICT = { language: set(STOPWORDS_DICT[language]) for language in STOPWORDS_DICT}


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


def prefilter_tokenizer(string: str) -> str:
    """Tokenizers split words based on unsupervised machine-learned models. Sometimes, they work weird.
    For example, in emails and user handles like `@user`, they would split `@` and `user` as 2 different tokens,
    making it impossible to detect usernames in single tokens later.

    To avoid that, we replace data of interest by meta-tokens before the tokenization, with regular expressions.
    """

    # Anonymize users and prevent tokenizers from splitting @ from the username
    string = re.sub(r"(\S+)?@(\S+)", "_USER_", string)

    # Find English dates like `01 Jan 20` or `01 Jan. 2020` but avoid capturing adjacent time like `12:08`.
    string = re.sub(r"([0-9]{1,2})? (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\.?( [0-9]{1,2})?( [0-9]{2,4})(?!:)",
                    "_DATE_", string, flags=re.IGNORECASE | re.MULTILINE)

    # Find French dates like `01 Jan 20` or `01 Jan. 2020` but avoid capturing adjacent time like `12:08`.
    string = re.sub(r"([0-9]{1,2})? (jan|fév|mar|avr|mai|jui|jui|aou|sep|oct|nov|déc)\.?( [0-9]{1,2})?( [0-9]{2,4})(?!:)",
                    "_DATE_", string, flags=re.IGNORECASE | re.MULTILINE)

    # Note : dates in ISO format like `2022-08-10` will not be split apart in tokens and will be captured
    # below in `normalize_token()`

    return string


def normalize_token(word: str, language: str):
    """Return normalized word tokens, where dates, times, digits, monetary units and URLs have their actual value replaced by meta-tokens designating their type. Stopwords ("the", "a", etc.) are removed and words are stemmed ("marketing" becomes "market", "apples" becomes "apple").

    Arguments:
        word (str): tokenized word in lower case only.
        language (str): the language used to detect dates. Supports `"french"`, `"english"` or `"any"`.

    Examples:
        `10:00` or `10 h` or `10am` or `10 am` will all be replaced by a `_TIME_` meta-token.
        `feb`, `February`, `feb.`, `monday` will all be replaced by a `_DATE_` meta-token.
    """
    value = word

    if re.match(r"user\d+", word) or re.match(r"@\S+", word) or "_user_" in word:
        # Discard usernames
        value = '_USER_'

    elif re.match(DATE_PATTERN, word) or "_date_" in word:
        # Record dates format - we don't need to know which dateé
        value = '_DATE_'

    elif re.match(TIME_PATTERN, word):
        # Record time/hour format - we don't need to know what time
        value = '_TIME_'

    elif re.match(URL_PATTERN, word):
        # Contains url - we don't need to know which site
        value = '_URL_'

    elif re.match(CODE_PATTERN, word):
        # Contains a filename referencing an code extension
        value = '_CODEFILE_'

    elif re.match(IMAGE_PATTERN, word):
        # Contains a filename referencing an image extension
        value = '_IMAGEFILE_'

    elif re.match(DOCUMENT_PATTERN, word):
        # Contains a filename referencing a document extension
        value = '_DOCUMENTFILE_'

    elif re.match(TEXT_PATTERN, word):
        # Contains a filename referencing a text extension
        value = '_TEXTFILE_'

    elif re.match(ARCHIVE_PATTERN, word):
        # Contains a filename referencing an archive package extension
        value = '_ARCHIVEFILE_'

    elif re.match(EXECUTABLE_PATTERN, word):
        # Contains a filename referencing an executable package extension
        value = '_BINARYFILE_'

    elif re.match(PATH_PATTERN, word):
        # Contains path - we don't need to know which file/folder
        value = '_PATH_'

    elif word in DATES[language]:
        # Record textual dates - we don't need to know which date
        value = '_DATE_'

    elif re.match(r"(CAD|USD|EUR|€|\$|£)", word):
        # Record price - we don't need to know which one
        # Note that the word tokenizer will split numbers and monetary units, so no need to regex digits + unit
        value = '_PRICE_'

    elif re.match(r"^\d+", word):
        # Discard numbers, possibly followed by unit, just record that we have a number
        value = '_NUMBER_'

    elif re.match(r"\.{2,}", word):
        # Teenagers need to calm the fuck down, ellipses need no more than three dots and two dots are not a thing
        value = '...'

    elif re.match(r"-{1,}", word):
        # Same with dashes
        value = '-'

    elif re.match(r"\?{1,}", word):
        # Same with question marks
        return '?'

    else:
        # Stem the word
        stemmer = SnowballStemmer(language)
        value = stemmer.stem(value)

    return value


def tokenize_sentences(sentence, language: str) -> list[str]:
    """Split a sentence into word tokens"""
    return [normalize_token(token.lower(), language)
            for token in nltk.word_tokenize(sentence, language=language)]


def split_sentences(text, language: str) -> list[str]:
    """Split a text into sentences"""
    return nltk.sent_tokenize(text, language=language)


class Data():
    def __init__(self, text: str, label: str):
        """Represent an item of training data

        Arguments:
            text (str): the content to label, which will be vectorized
            label (str): the category of the content, which will be predicted by the model
        """
        self.text = text
        self.label = label


def get_wordvec(wv: gensim.models.KeyedVectors, word: str, language: str, syn1neg: np.array = None, normalize = True) -> np.array:
    """Return the vector associated to a word, through a dictionnary of words.

    If `syn1neg` is provided, we used "OUT" matrix of the embedding, for the dual-space embedding scheme.
    This is useful when using word2vec to build a search-engine, the documents to index are better embedded using
    the "OUT" matrix (stored in gensim.Word2Vec.syn1neg matrix), while the queries are better embedded using the "IN" matrix
    (gensim.Word2Vec.wv).

    References:
        A Dual Embedding Space Model for Document Ranking (2016), Bhaskar Mitra, Eric Nalisnick, Nick Craswell, Rich Caruana
        https://arxiv.org/pdf/1602.01137.pdf


    Returns:
        the nD vector.
    """
    if normalize:
        word = normalize_token(word.lower(), language)

    if word and word in wv:
        if syn1neg is not None:
            vec = syn1neg[wv.key_to_index[word]]
        else:
            vec = wv[word]

        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0. else vec
    else:
        return np.zeros(wv.vector_size)


class Word2Vec(gensim.models.Word2Vec):
    def __init__(self, sentences: list[str], name: str = "word2vec", vector_size: int = 300):
        """Train or retrieve an existing word2vec word embedding model

        Arguments:
            name (str): filename of the model to save and retrieve. If the model exists already, we automatically load it. Note that this will override the `vector_size` with the parameter defined in the saved model.
            vector_size (int): number of dimensions of the word vectors
            force_recompute (bool): if `True`, don't load an existing model matching `name`.
        """
        self.pathname = get_models_folder(name)
        self.vector_size = vector_size
        print(f"got {len(sentences)} pieces of text")

        # training = [tokenize_sentences(sentence, language=language) for sentence in set(sentences)]
        with Pool() as pool:
            training: list[list[list[str]]] = pool.map(self.tokenize_sentences_parallel, set(sentences))

        print("tokenization done")

        # Flatten the first dimension of the list of list of list of strings :
        training = [sentence for text in training for sentence in text]
        print(f"got {len(training)} sentences")
        super().__init__(training, vector_size=vector_size, window=7, min_count=5, workers=os.cpu_count(), epochs=150, ns_exponent=-0.5, sample=5e-5)
        print("training done")
        self.save(self.pathname)
        self.wv.save(self.pathname + "_vectors")
        print("saving done")


    def tokenize_sentences_parallel(self, text):
        """Thread-safe call to `.tokenize_sentences()` to be called in multiprocessing.Pool map"""
        clean_text = typography_undo(text)
        language = guess_language(clean_text)
        clean_text = prefilter_tokenizer(clean_text)
        return [tokenize_sentences(sentence, language) for sentence in split_sentences(clean_text, language)]


    @classmethod
    def load_model(cls, name: str):
        """Load a trained model saved in `models` folders"""
        return cls.load(get_models_folder(name))


def get_features(post: str, num_features: int, wv: gensim.models.KeyedVectors, language: str, syn1neg: np.array = None, tokens: list = None) -> dict:
    """Extract word features from the text of `post`.

    We use meta-features like date, prices, time, number that discard the actual value but retain the property.
    That is, we don't care about the actual date, price or time, we only care that there is a date, price or time.
    Meta-features are tagged with random hashes as to not be mistaken with text.

    For everything else, we use the actual words.

    Arguments:
        post (Data): the training text (message or sentence) that will be tokenized and turned into features.
        num_features (int): the number of dimensions of the featureset. This is vector size used in the `Word2Vec` model.
        wv (gensim.models.KeyedVector): the dictionnary mapping words with vectors,
        syn1neg (np.array): the W_out matrix for word embedding, in the Dual Embedding Space Model. [^1] If not provided, embedding uses the default W_in matrix. W_out is better to vectorize documents for search-engine purposes.
        language (str): the language used to detect dates and detect words separators used in tokenization. Supports `"french"` and `"english"`.
        tokens (list[str]): if given, we discard internal tokenization and normalization and directly use this list of tokens. The need to be normalized already.

    [^1]: https://arxiv.org/pdf/1602.01137.pdf

    Return:
        (dict): dictionnary of features, where keys are initialized with the positional number of vector elements and their value, plus the optional external data.
    """
    features = np.zeros(num_features)

    if tokens is not None:
        i = len(tokens)
        for token in tokens:
            vector = get_wordvec(wv, token, language, normalize=False)
            features += vector
    else:
        i = 0
        for word in nltk.word_tokenize(post, language=language):
            vector = get_wordvec(wv, word, language, syn1neg)
            features += vector
            i += 1

    # Finish the average calculation (so far, only summed)
    if i > 0:
        features /= i

    # NLTK models take dictionnaries of features as input, so bake that.
    return dict(enumerate(features, 1))

class Classifier(SklearnClassifier):
    def __init__(self,
                 training_set: list[Data],
                 name: str,
                 wv: gensim.models.KeyedVectors,
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

        if wv and isinstance(wv, gensim.models.KeyedVectors):
            self.wv = wv
            self.vector_size = wv.vector_size
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

        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"))


    def get_features_parallel(self, post: Data) -> tuple[str, str]:
        """Thread-safe call to `.get_features()` to be called in multiprocessing.Pool map"""
        clean_sentence = typography_undo(post.text)
        language = guess_language(clean_sentence)
        clean_sentence = prefilter_tokenizer(clean_sentence)
        return (get_features(clean_sentence, self.vector_size, self.wv, language), post.label)


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
        clean_sentence = typography_undo(post)
        language = guess_language(clean_sentence)
        clean_sentence = prefilter_tokenizer(clean_sentence)
        item = get_features(clean_sentence, self.vector_size, self.wv, language)
        return super().classify(item)

    def prob_classify(self, post: str) -> tuple[str, float]:
        """Apply a label on a post based on the trained model and output the probability too."""
        clean_sentence = typography_undo(post)
        language = guess_language(clean_sentence)
        clean_sentence = prefilter_tokenizer(clean_sentence)
        item = get_features(clean_sentence, self.vector_size, self.wv, language)

        # This returns a weird distribution of probabilities for each label that is not quite a dict
        proba_distro = super().prob_classify(item)

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
        print(f"Init. Got {len(data_set)} items.")

        if word2vec and isinstance(word2vec, Word2Vec):
            self.wv = word2vec.wv
            self.vector_size = word2vec.vector_size
            self.syn1neg = word2vec.syn1neg
        else:
            raise ValueError("wv needs to be a dictionnary-like map")

        # Remove duplicated content if any. May happen with translated content when non-existent translations are inited with the original language.
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

        print(f"Cleanup. Got {len(data_set)} remaining items.")

        # Build the training set from webpages
        training_set = [Data(post["title"] + "\n\n" + post["content"] + "\n\n".join(post["h1"]) + "\n\n".join(post["h2"]), post["url"])
                        for post in data_set
                        if len(post["content"]) > 50]

        # Prepare the ranker for BM25 : list of tokens for each document
        with Pool() as pool:
            ranker_docs: list = pool.map(self.tokenize_parallel, training_set)

        # Turn tokens into features
        with Pool() as pool:
            documents: list = pool.map(self.get_features_parallel, ranker_docs)

        docs = [list(document.values()) for document in documents]

        self.vectors_all = np.array(docs)
        self.all_norms = np.linalg.norm(self.vectors_all, axis=1)
        self.urls = [post.label for post in training_set]

        # The database of web pages with limited features
        self.index = {post["url"]: {"title": post["title"], "excerpt": post["excerpt"], "date": post["date"]}
                      for post in data_set if len(post["content"]) > 50}

        # Values from https://arxiv.org/pdf/1602.01137.pdf, p.6, section 3.3
        self.ranker = BM25Okapi(ranker_docs, k1=1.7, b=0.95)

        # Garbage collection to avoid storing in the saved model stuff we won't need anymore
        del self.syn1neg

        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"))


    def get_features_parallel(self, tokens: list[str]) -> tuple[str, str]:
        """Thread-safe call to `.get_features()` to be called in multiprocessing.Pool map"""
        # Language doesn't matter, tokenization and normalization are done already
        return get_features(None, self.vector_size, self.wv, language="english", syn1neg=self.syn1neg, tokens=tokens)


    def tokenize_parallel(self, post: Data) -> list[str]:
        clean_sentence = typography_undo(post.text)
        language = guess_language(clean_sentence)
        clean_sentence = prefilter_tokenizer(clean_sentence)
        return [normalize_token(word.lower(), language) for word in nltk.word_tokenize(clean_sentence, language=language)]

    @classmethod
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        model = joblib.load(get_models_folder(name) + ".joblib")
        if isinstance(model, Indexer):
            return model
        else:
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))

    def rank(self, post: str) -> str:
        """Apply a label on a post based on the trained model."""
        clean_sentence = typography_undo(post)
        language = guess_language(clean_sentence)
        clean_sentence = prefilter_tokenizer(clean_sentence)
        tokenized_query = [normalize_token(word.lower(), language) for word in nltk.word_tokenize(clean_sentence, language=language)]

        # Get the the centroid of the word embedding vector
        query = get_features(clean_sentence, self.vector_size, self.wv, language=language, tokens=tokenized_query)
        vector = np.array(list(query.values()))
        norm = np.linalg.norm(vector)
        norm = 1.0 if norm == 0.0 else norm

        # Compute the cosine similarity of centroids between query and documents,
        # then aggregate the ranking from BM25+ to it for each URL.
        # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
        aggregates = 0.97 * np.dot(self.vectors_all, vector) / (norm * self.all_norms) + 0.03 * self.ranker.get_scores(tokenized_query)

        results = {(url, similarity) for url, similarity in zip(self.urls, aggregates) if similarity > 0.35}

        # Return the 20 most similar documents
        return sorted(results, key=lambda x:x[1], reverse=True)[0:50]
