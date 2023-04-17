"""
High-level natural language processing module for message-like (emails, comments, posts) input.

© 2023 - Aurélien Pierre
"""

import random
import re
import os
from multiprocessing import Pool

import gensim
import joblib

import numpy as np

import nltk
from nltk.classify import SklearnClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from core.patterns import DATE_PATTERN, TIME_PATTERN, URL_PATTERN
from core.utils import get_models_folder

#nltk.download('punkt')

# Day/month tokens and their abbreviations
DATES = {
    "english": [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "aug", "sept", "oct", "nov", "dec",
        "jan.", "feb.", "mar.", "apr.", "aug.", "sept.", "oct.", "nov.", "dec.",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun.",
    ],
    "french": [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        "jan", "fév", "mars", "avr", "jui", "juil", "aou", "sept", "oct", "nov", "déc",
        "jan.", "fév.", "mars.", "avr.", "jui.", "juil.", "aou.", "sept.", "oct.", "nov.", "déc.",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
        "lun", "mar", "mer", "jeu", "ven", "sam", "dim",
        "lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.",
    ]
}

DATES["any"] = DATES["english"] + DATES["french"]


def normalize_token(word: str, language: str = "any"):
    """Return normalized word tokens, where dates, times, digits, monetary units and URLs have their actual value replaced by meta-tokens designating their type.

    Arguments:
        word (str): tokenized word
        language (str): the language used to detect dates. Supports `"french"`, `"english"` or `"any"`.

    Examples:
        `10:00` or `10 h` or `10am` or `10 am` will all be replaced by a `_TIME_` meta-token.

        `feb`, `February`, `feb.`, `monday` will all be replaced by a `_DATE_` meta-token.
    """
    value = word

    if re.match(r".*user\d+", word):
        # Discard the names of chat users
        value = ''

    elif re.match(DATE_PATTERN, word):
        # Record dates format - we don't need to know which dateé
        value = '_DATE_'

    elif re.match(TIME_PATTERN, word):
        # Record time/hour format - we don't need to know what time
        value = '_TIME_'

    elif re.match(r"(CAD|USD|EUR|€|\$|£)", word):
        # Record price - we don't need to know which one
        # Note that the word tokenizer will split numbers and monetary units, so no need to regex digits + unit
        value = '_PRICE_'

    elif re.match(r"^\d+$", word):
        # Discard numbers, just record that we have a number
        value = '_NUMBER_'

    elif re.match(r"\d", word):
        # We have a mix of digits and something else. Weird stuff, count it as digit.
        value = '_NUMBER_'

    elif re.match(URL_PATTERN, word):
        # Contains url - we don't need to know which site
        value = '_URL_'

    elif re.match(r"\.{2,}", word):
        # Teenagers need to calm the fuck down, ellipses need no more than three dots
        value = '...'

    elif re.match(r"-{1,}", word):
        # Same with dashes
        value = '-'

    elif re.match(r"\?{1,}", word):
        # Same with question marks
        return '?'

    elif word in DATES[language]:
        # Record textual dates - we don't need to know which date
        value = '_DATE_'

    return value


def tokenize_sentences(sentence, language: str = "any") -> list[str]:
    """Split a sentence into word tokens"""
    return [normalize_token(token.lower(), language)
            for token in nltk.word_tokenize(sentence, language=(language if language != "any" else "english"))]


def split_sentences(text, language: str = "any") -> list[str]:
    """Split a text into sentences"""
    if language == "any":
        language = "english"
    return nltk.sent_tokenize(text, language=language)


def __update_dict(dt: dict, vector: list):
    # Increment the number of occurences for each feature
    for key, value in zip(dt.keys(), vector):
        dt[key] += value

class Data():
    def __init__(self, text: str, label: str):
        """Represent an item of training data

        Arguments:
            text (str): the content to label, which will be vectorized
            label (str): the category of the content, which will be predicted by the model
        """
        self.text = text
        self.label = label


def get_wordvec(wv: gensim.models.KeyedVectors, word: str, language: str = "any") -> np.array:
    """Return the vector associated to a word, through a dictionnary of words.

    Returns:
        the 1D vector.
    """
    if word in wv:
        return wv[normalize_token(word.lower(), language)]
    else:
        return np.zeros(wv.vector_size)


class Word2Vec(gensim.models.Word2Vec):
    def __init__(self, sentences: list[str], name: str = "word2vec", vector_size: int = 300, language: str = "any"):
        """Train or retrieve an existing word2vec word embedding model

        Arguments:
            name (str): filename of the model to save and retrieve. If the model exists already, we automatically load it. Note that this will override the `vector_size` with the parameter defined in the saved model.
            vector_size (int): number of dimensions of the word vectors
            force_recompute (bool): if `True`, don't load an existing model matching `name`.
            language (str): The language used to detect months/days in dates. Supports `"french"`, "`english`" or `"any" (all languages will be tried).
        """
        self.pathname = get_models_folder(name)
        self.vector_size = vector_size

        training = [tokenize_sentences(sentence, language=language) for sentence in sentences]
        super().__init__(training, vector_size=vector_size, window=20, min_count=5, workers=8, epochs=100, ns_exponent=-0.5)
        self.save(self.pathname)
        self.wv.save(self.pathname + "_vectors")


    @classmethod
    def load_model(cls, name: str):
        """Load a trained model saved in `models` folders"""
        return cls.load(get_models_folder(name))


def get_features(post: str, num_features: int, wv: gensim.models.KeyedVectors, external_data: dict = {}, language: str = 'any') -> dict:
    """Extract word features from the text of `post`.

    We use meta-features like date, prices, time, number that discard the actual value but retain the property.
    That is, we don't care about the actual date, price or time, we only care that there is a date, price or time.
    Meta-features are tagged with random hashes as to not be mistaken with text.

    For everything else, we use the actual words.

    Arguments:
        post (Data): the training text (message or sentence) that will be tokenized and turned into features.
        num_features (int): the number of dimensions of the featureset. This is vector size used in the `Word2Vec` model.
        external_data (list): optional, arbitrary set of (meta)data that can be added to the feature set. Will be used as extra keys in the output dictionnary, where values are set to boolean `1`.
        language (str): the language used to detect dates and detect words separators used in tokenization. Supports `"french"` and `"english"`.

    Return:
        (dict): dictionnary of features, where keys are initialized with the positional number of vector elements and their value, plus the optional external data.
    """

    # The dict is not the best suited data representation for us here
    # we create bullshit keys from integer indices of the word2vec vector size
    features = dict.fromkeys(range(num_features), 0.)

    i = 0
    if language == "any":
        language = "english"

    for word in nltk.word_tokenize(post, language=language):
        __update_dict(features, get_wordvec(wv, word, language))
        i += 1

    # Normalize (average)
    if i > 0:
        for key in features:
            features[key] /= i

    # Add the external features if any
    if external_data:
        features.update(external_data)

    return features


class Classifier(SklearnClassifier):
    def __init__(self,
                 training_set: list[Data],
                 name: str,
                 wv: gensim.models.KeyedVectors,
                 validate: bool = True,
                 language: str = "any",
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

        self.language = language

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
        return (get_features(post.text, self.vector_size, self.wv, language=self.language), post.label)


    @classmethod
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        model = joblib.load(get_models_folder(name) + ".joblib")
        if isinstance(model, nltk.SklearnClassifier):
            return model
        else:
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))

    def classify(self, post: str, external_data: dict = None) -> str:
        """Apply a label on a post based on the trained model."""
        item = get_features(post, self.vector_size, self.wv, language=self.language, external_data=external_data)
        return super().classify(item)

    def prob_classify(self, post: str, external_data: dict = None) -> tuple[str, float]:
        """Apply a label on a post based on the trained model and output the probability too."""
        item = get_features(post, self.vector_size, self.wv, language=self.language, external_data=external_data)

        # This returns a weird distribution of probabilities for each label that is not quite a dict
        proba_distro = super().prob_classify(item)

        # Build the list of dictionnaries like `label: probability`
        output = {i: proba_distro.prob(i) for i in proba_distro.samples()}

        # Finally, return label and probability only for the max proba of each element
        return (max(output, key=output.get), max(output.values()))
