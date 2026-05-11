"""
High-level natural language processing module for message-like (emails, comments, posts) input.

Supports automatic language detection, word tokenization and stemming for `'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'`.

© 2023 - Aurélien Pierre
"""

import random
import regex as re
import os
import concurrent
import unicodedata as ud
import multiprocessing

from collections import Counter

import gensim
from gensim.models.callbacks import CallbackAny2Vec

import joblib
import sqlite3

import numpy as np

import nltk
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from fast_langdetect import detect
import blingfire
import langcodes
import pycountry

from .patterns import *
from .utils import get_models_folder, typography_undo, clean_whitespaces, timeit, guess_date
from .language import *
from .crawler import web_page

DOCUMENTS = None
TOKENIZER = None

latin_letters = {}

def _is_latin(uchr):
    try:
        return latin_letters[uchr]
    except KeyError:
        return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def _roman_chars(unistr):
    return [_is_latin(uchr) for uchr in unistr if uchr.isalpha()]


def parse_lang_to_iso639_1(value: str | None) -> str | None:
    """Parse any string language code to ISO 639-1 language code"""
    try:
        lang = langcodes.find(value)
        code = lang.language

        # Ensure it's actually ISO 639-1
        if len(code) == 2:
            return code

        # Convert ISO639-3 -> ISO639-1 if possible
        macro = lang.to_alpha3()

        # langcodes itself cannot always reverse-map
        entry = pycountry.languages.get(alpha_3=macro)
        if entry and hasattr(entry, "alpha_2"):
            return entry.alpha_2

    except Exception:
        pass

    return None


def guess_language(string: str, stopwords_threshold: float = 0.05, letters_threshold: float = 0.8) -> str | None:
    """Basic language guesser based on stopwords detection.

    Stopwords are the most common words of a language: for each language, we count how many stopwords we found and return the language having the most matches. It is accurate for paragraphs and long documents, not so much for short sentences.

    Params:
        string: the string to analyze. Needs to be lowercased but to retain accents and diacritics.
        stopwords_threshold: the minimum ratio of stopwords divided by total words in strings to be found to conclude on a language. For example, Japanese companies often have technical reports written in Japanese but still containing some English. If less than 5% of the words are known English stopwords, we could conclude it's not English.
        letters_threshold: the minimum ratio of roman (latin) characters among all characters (including numbers, symbols and non-latin alphabets) to be found to conclude on a language.

    Returns:
        ISO 639-1 language code. Defaults to "en" if nothing found.
    """

    # Number of roman characters
    # Note : scientific papers written in English may contain some Greek in equations.
    roman = sum(_roman_chars(string.lower()))
    letters = [uchr for uchr in string if uchr.isalpha()]
    if len(letters) == 0:
        return None

    if roman / len(letters) < letters_threshold:
        return None

    # else: we have mostly latin characters. Guess language
    tokenizer = RegexpTokenizer(r'\w+|[\d\.\,]+|\S+')
    words = [token for token in tokenizer.tokenize(string.lower())]
    scores = []
    for lang in STOPWORDS_DICT:
        scores.append(len(set(words).intersection(STOPWORDS_DICT[lang])))

    index_max = max(range(len(scores)), key=scores.__getitem__)
    language = list(STOPWORDS_DICT.keys())[index_max]
    value = scores[index_max]

    if value > max(stopwords_threshold * len(words), 1):
        return LANG_MAP_REVERSE[language]
    else:
        # The best language found still has a too-low ratio of use in string
        return None


def detect_language(text: str) -> str | None:
    """
    Detect language from arbitrary text safely.

    Returns:
        ISO 639-1 language code.
    """

    if not text or len(text.strip()) < 5:
        return None

    try:
        result = detect(text, model="full")
        lang = str(result[0]["lang"])
        score = float(result[0]["score"])

        # Confidence threshold
        if score < 0.70:
            return guess_language(text)

        return lang

    except Exception as e:
        print(e)
        return None


def tokenize_document_to_words(text: str, language: str | None = None, backend: str = "blingfire") -> list[str]:
    """Split a text into single words

    Arguments:
        language: ISO 639-1 language code.

    Returns:
        Bag of words for the whole document. Sentence delimiters are removed.
    """
    if backend == "blingfire" or not language:
        return blingfire.text_to_words(text).split()
    elif backend == "nltk":
        return nltk.word_tokenize(text, language=LANG_MAP[language])
    

def split_document_to_sentences(text: str, language: str | None = None, backend: str = "blingfire") -> list[str]:
    """Split a text into a list of sentences.
    
    Arguments:
        language: ISO 639-1 language code.

    Returns:
        List of sentences as full text.
    """
    if backend == "blingfire" or not language:
        return blingfire.text_to_sentences(text).splitlines()
    elif backend == "nltk":
        return nltk.sent_tokenize(text, language=LANG_MAP[language])


def tokenize_document_to_sentences(text: str, language: str | None = None, backend: str = "blingfire") -> list[list[str]]:
    """Split a text into single words as a list of lists

    Arguments:
        language: ISO 639-1 language code.

    Returns: 
        List of sentences, each sentence is itself a list of words.
    """
    result = []
    sentences = split_document_to_sentences(text, language=language, backend=backend)
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        result.append(tokenize_document_to_words(sent, language=language, backend=backend))

    return result


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
            string = key.sub(value, string, concurrent=True)

        for key, value in self.abbreviations.items():
            string = string.replace(key, value)

        if meta_tokens:
            for key, value in self.meta_tokens_pipe.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                try:
                    string = key.sub(value, string, timeout=15, concurrent=True)
                except TimeoutError:
                    print("Meta-token detection timed out on %s with:\n%s" % (key, string))

        # WARNING: URLs and file pathes need to have been parsed before
        string = string.replace(":", " ") # C++ members
        string = string.replace("\\", "") # LaTeX commands

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
                    word = key.sub(value, word, timeout=10, concurrent=True)
                except TimeoutError:
                    print("Lemmatization timed out on %s with:\n%s" % (key, word))

        return word


    def normalize_text(self, document:str) -> str:
        """Prepare text for tokenization by converting it to lowercase ASCII characters.

        This will loose accents, diacritics and capitals, which means some nuance will be lost
        at the benefit of generality. In case this does not suit your usecase, you may
        inherit the `Tokenizer` class, build a child class and re-implement this method
        """
        return typography_undo(str(document).lower())


    def normalize_token(self, 
                        word: str, 
                        language: str, 
                        normalize: bool = True,
                        meta_tokens: bool = True, 
                        stem: bool = True,
                        remove_stopwords: bool = True) -> str | None:
        """Return normalized, lemmatized and stemmed word tokens, where dates, times, digits, monetary units and URLs have their actual value replaced by meta-tokens designating their type. Stopwords ("the", "a", etc.), punctuation etc. is replaced by `None`, which should be filtered out at the next step.

        Arguments:
            word (str): tokenized word in lower case only.
            language (str): the ISO 369-1 language code used to remove typical stopwords.
            normalize (str): remove punctuation and leading/trailing symbols, replace abbreviations, etc.
            meta_tokens (bool): replace string patterns by meta_tokens
            stem (bool): remove word suffixes, double consonnants, etc.
            remove_stopwords (bool): remove stopwords

        Examples:
            `10:00` or `10 h` or `10am` or `10 am` will all be replaced by a `_TIME_` meta-token.
            `feb`, `February`, `feb.`, `monday` will all be replaced by a `_DATE_` meta-token.
        """

        string = word

        if normalize:
            string = word.strip("?!#=+-,:;'\"^*./`()[]{}& \n\r\t<>")

            if len(string) == 0 or " " in string or "\n" in string:
                # empty string or
                # tokenizer failed to split tokens on spaces
                return None

        # Replace abbreviations by full text or meta-tokens, as defined in the replacement dict
        if self.replacements and string in self.replacements:
            test_string = self.replacements[string]
            if test_string in self.meta_tokens:
                # In case replacement yields a meta-token:
                # 1. if meta_tokens mode enabled, return immediately : we are done.
                if meta_tokens:
                    return test_string
                # 2. if meta_tokens mode disabled, do nothing : cancel meta_tokens.
            else:
                string = test_string

        # If token is a meta-token, nothing more to do.
        string_upper = string.upper()
        if string_upper in self.meta_tokens:
            return string_upper

        # Remove Markdown markers for _italics_ and __bold__
        # Note: internal under_scores and da-shes are already handled in metatokens regex loop.
        string = string.strip("_")

        # Last chance of identifying meta-tokens in an atomic way
        if meta_tokens:
            for key, value in self.internal_meta_tokens.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                if key.match(string, timeout=10, concurrent=True):
                    return value
                
        # Check if string is a stopword from our custom list
        if remove_stopwords:
            # Language-specific stopwords don't use stemmed words
            if self.lang_stopwords and language in self.lang_stopwords:
                if string in self.lang_stopwords[language]:
                    return None

        # Lemmatize / Stem
        if stem:
            string = self.lemmatize(string)

        # Check if string is a stopword from our custom list
        if remove_stopwords:
            # Language-agnostic stopwords may use stemmed words
            if self.stopwords and string in self.stopwords:
                return None
            
        return string


    def tokenize_text(self, 
                      sentence: str, 
                      language: str | None = None, 
                      normalize: bool = True,
                      meta_tokens: bool = True, 
                      stem: bool = True,
                      remove_stopwords: bool = True) -> list[str]:
        """Split an arbitrary text into normalized word tokens and meta-tokens.
        No sentence or paragraph detection will be attempted.

        Arguments:
            sentence: the input single sentence.
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided

        Returns:
            tokens (list[str]): the list of normalized tokens as a bag of words.
        """
        tokens = [self.normalize_token(token, language, meta_tokens=meta_tokens, stem=stem, normalize=normalize, remove_stopwords=remove_stopwords)
                  for token in tokenize_document_to_words(sentence, language=language, backend=self.backend)]
 
        return [item for item in tokens if isinstance(item, str)]


    def tokenize_document_flat(self, 
                                document:str, 
                                language: str | None = None, 
                                normalize: bool = True,
                                meta_tokens: bool = True, 
                                stem: bool = True,
                                remove_stopwords: bool = True) -> list[str]:
        """Cleanup and tokenize a document or a sentence as an atomic element, meaning we don't split it into sentences. 
        Use this either for search-engine purposes (into a document's body) or if the document is already split into sentences. 
        The document text needs to have been prepared and cleaned, which means :

        - lowercased (optional but recommended) with `str.lower()`,
        - translated from Unicode to ASCII (optional but recommended) with [utils.typography_undo()][],
        - cleaned up for sequences of whitespaces with [utils.cleanup_whitespaces()][]

        Note:
            the language is detected internally if not provided as an optional argument. When processing a single sentence extracted from a document, instead of the whole document, it is more accurate to run the language detection on the whole document, ahead of calling this method, and pass on the result here.

        Arguments:
            document (str): the text of the document to tokenize
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided. The text is prefiltered with [self.prefilter][].

        Returns:
            tokens (list[str]): a 1D list of normalized tokens and meta-tokens.
        """
        if language is None:
            language = detect_language(document)

        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return self.tokenize_text(clean_text, language=language, meta_tokens=meta_tokens, stem=stem, normalize=normalize, remove_stopwords=remove_stopwords)


    def tokenize_document_per_sentence(self, 
                                       document: str, 
                                       language: str | None = None, 
                                        normalize: bool = True,
                                        meta_tokens: bool = True, 
                                        stem: bool = True,
                                        remove_stopwords: bool = True) -> list[list[str]]:
        """Cleanup and tokenize a whole document as a list of sentences, meaning we split it into sentences before tokenizing. 
        Use this to train a Word2Vec (embedding) model so each token is properly embedded into its syntactic context. 
        The document text needs to have been prepared and cleaned, which means :

        - lowercased (optional but recommended) with `str.lower()`,
        - translated from Unicode to ASCII (optional but recommended) with [utils.typography_undo()][],
        - cleaned up for sequences of whitespaces with [utils.cleanup_whitespaces()][]

        Arguments:
            document (str): the text of the document to tokenize
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided. The text is prefiltered with [self.prefilter][]

        Returns:
            tokens: a 2D list of sentences (1st axis), each containing a list of normalized tokens and meta-tokens (2nd axis).
        """
        # TODO: prefilter n-grams ?
        if language is None:
            language = detect_language(document)

        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return [self.tokenize_text(sentence, language=language, meta_tokens=meta_tokens, stem=stem, normalize=normalize, remove_stopwords=remove_stopwords)
                for sentence in split_document_to_sentences(clean_text, language=language, backend=self.backend)]


    def tokenize_document_per_paragraph(self, 
                                       document: str, 
                                       language: str | None = None, 
                                        normalize: bool = True,
                                        meta_tokens: bool = True, 
                                        stem: bool = True,
                                        remove_stopwords: bool = True) -> list[list[str]]:
        """Cleanup and tokenize a whole document as a list of paragraphs, meaning we split it on `\n\n` or `\r\n` before tokenizing. 
        Use this to train a Word2Vec (embedding) model so each token is properly embedded into its syntactic context. 
        The document text needs to have been prepared and cleaned, which means :

        - lowercased (optional but recommended) with `str.lower()`,
        - translated from Unicode to ASCII (optional but recommended) with [utils.typography_undo()][],
        - cleaned up for sequences of whitespaces with [utils.cleanup_whitespaces()][]

        Arguments:
            document (str): the text of the document to tokenize
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided. The text is prefiltered with [self.prefilter][]

        Returns:
            tokens: a 2D list of paragraphs (1st axis), each containing a list of normalized tokens and meta-tokens (2nd axis).
        """
        # TODO: prefilter n-grams ?
        if language is None:
            language = detect_language(document)

        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return [self.tokenize_text(paragraph, language=language, meta_tokens=meta_tokens, stem=stem, normalize=normalize, remove_stopwords=remove_stopwords)
                for paragraph in re.split(r'(?:\r\n|\r|\n){2,}', clean_text)]


    def __init__(self,
                 meta_tokens: dict[re.Pattern: str] | None = None,
                 abbreviations: dict[str: str] | None = None,
                 replacements: dict[str: str] | None = None,
                 stopwords: set[str] | None = None,
                 lang_stopwords: dict[str: set[str]] | None = None,
                 backend: str = "blingfire"):
        """Pre-processing pipeline and tokenizer, splitting a string into normalized word tokens.

        Arguments:
            meta_token: the pipeline of regular expressions to replace with meta-tokens. Keys must be `re.Pattern` declared with `re.compile()`, values must be meta-tokens assumed to be nested in underscores. The pipeline dictionnary will be processed in the order of declaration, which relies on using Python >= 3.7 (making `dict` ordered by default). If not provided, it is inited by default with a pipeline suitable for bilingual English/French language processing on technical writings (see notes).
            abbreviations (dict[str: str]): pipeline of abbreviations to replace, as `to_replace: replacement` dictionnary. Will be processed in order of declaration.
            replacements: dictionnary used to replace 1:1 `key` with `value` in tokens.
            stopwords: flat list of language-agnostic stopwords to remove from tokens.
            lang_stopwords: language-specific stopwords as a dictionnary. Keys have to be ISO 639-1 language code, and values the set of stopwords.
            backend: `blingfire` or `nltk`, choose which Python library will perform the actual tokenization.
            `blingfire` uses Microsoft Blingfire default tokenizer (pattern-based), while `nltk` uses Punkt. 
        """
        self.backend = backend

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
                PHOTOSPEED: " _SHUTTERSPEED_ ",
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
                # In-words dashes/compound words : replace by underscore for uniform handling with n-grams
                DASHES: "_",
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

        if replacements is None:
            self.replacements = REPLACEMENTS
        else:
            self.replacements = replacements

        if stopwords:
            self.stopwords = set(stopwords)
        else:
            self.stopwords = None

        if lang_stopwords is None:
            self.lang_stopwords = { LANG_MAP_REVERSE[lang]: value 
                                    for lang, value in STOPWORDS_DICT.items() }
        else:
            self.lang_stopwords = None


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

def unique_terms(doc: list[list[str]]) -> set[str]:
    return { word for sentence in doc for word in sentence }

class Word2Vec(gensim.models.Word2Vec):
    @timeit()
    def __init__(self, documents: list[list[str]], 
                 name: str = "word2vec", 
                 vector_size: int = 300, 
                 epochs: int = 200, 
                 window: int = 5, 
                 min_count: int = 5, 
                 sample: float = 0.0005, 
                 tokenizer: Tokenizer = None,
                 compute_idf: bool = False,
                 **kwargs):
        """Train, re-train or retrieve an existing word2vec word embedding model

        Arguments:
            documents (list[list[str]]): the pre-tokenized training data. The outermost list is the documents, aka a list of sentences. Each sentence is a list of tokens.
            name (str): filename of the model to save and retrieve. If the model exists already, we automatically load it. Note that this will override the `vector_size` with the parameter defined in the saved model.
            vector_size (int): number of dimensions of the word vectors
            epochs (int): number of iterations of training for the machine learning. Small corpora need 2000 and more epochs. Increases the learning time.
            window (int): size of the token collocation window to detect
            min_count (int): remove all words used fewer times than this from the vocabulary
            sample (float):
            tokenizer: instance of tokenization.
            compute_idf: compute and store corpus IDF data for SIF weighting. Disable it
                to keep saved model artifacts smaller when `use_sif` is not needed.
            kwargs: passed directly through to `gensim.Word2Vec.__init__()`
        """
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        """Tokenizer used to train the model. We store it to be sure to use the same when using it."""

        self.pathname = get_models_folder(name)
        self.vector_size = vector_size

        self.N_docs = len(documents) 
        """Number of documents in the training corpus"""
        print(f"got {self.N_docs} documents")

        # Flatten the first dimension of the list of list of list of strings :
        sentences = [sentence for text in documents for sentence in text]
        self.N_sentences = len(sentences)
        """Number of sentences in the training corpus"""
        print(f"got {self.N_sentences} sentences")

        words = [word for sentence in sentences for word in sentence]
        self.N_words = len(words)
        """Number of words (tokens) in the training corpus"""
        print(f"got {self.N_words} words (tokens)")

        # Frequency of terms aka unique words
        counts = Counter(words)
        self.N_terms = len(counts)
        """Number of terms (unique words) in the training corpus"""
        print(f"got {self.N_terms} unique terms")
        del words

        # Normalize frequencies
        counts = {w: c / self.N_words for w, c in counts.items()}

        # Sort terms by frequency
        counts = dict(sorted(counts.items(), key=lambda counts: counts[1]))

        # Save to file for manual review of stopwords
        with open(get_models_folder("stopwords"), 'w', encoding='utf8') as f:
            for key, value in counts.items():
                f.write(f"{key}: {value}\n")
        print("stopwords saved")

        self.idf: dict[str, float] | None = None
        """Inverse Document Frequency, used only for SIF weighting when enabled."""

        self.avg_doc_len: float | None = None
        """Average number of words in documents of the training corpus, available with IDF stats."""

        if compute_idf:
            self.compute_idf(documents)

        del counts

        loss_logger = LossLogger()
        super().__init__(sentences, vector_size=vector_size, window=window, min_count=min_count, 
                         epochs=epochs, sample=sample, callbacks=[loss_logger], 
                         compute_loss=True, sg=1, max_final_vocab=100000, hs=0, negative=20, alpha=0.020,
                         workers=os.cpu_count() or 1, batch_words=100000, **kwargs)
        print("training done")

        if self.idf is not None:
            self.prune_idf()
        self.save(self.pathname)
        print("saving done")
    

    def compute_idf(self, documents: list[list[str]]) -> None:
        """Compute and store IDF statistics from a tokenized document corpus."""

        with concurrent.futures.ProcessPoolExecutor() as executor:
            doc_sets = executor.map(unique_terms, documents)

        df_counts = Counter()
        for s in doc_sets:
            df_counts.update(s)

        self.idf = { term: self.N_docs / df for term, df in df_counts.items() }

        doc_lens = [sum(len(sentence) for sentence in doc) for doc in documents]
        self.avg_doc_len = sum(doc_lens) / len(doc_lens) if len(doc_lens) > 0 else 1.0


    def update_idf(self, documents: list[list[str]]) -> None:
        """Update IDF statistics and corpus-dependent metadata with new documents.

        Arguments:
            documents: New pre-tokenized documents.
        """

        if getattr(self, "idf", None) is None or getattr(self, "avg_doc_len", None) is None:
            raise RuntimeError("IDF stats were not computed for this model")

        # Prepare new documents stats
        new_N_docs = len(documents)
        doc_lens = [ sum(len(sentence) for sentence in doc) for doc in documents ]
        total_new_tokens = sum(doc_lens)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            doc_sets = executor.map(unique_terms, documents)

        new_df_counts = Counter()
        for s in doc_sets:
            new_df_counts.update(s)

        old_N_docs = self.N_docs

        # Update stats with new and old documents
        self.N_docs += new_N_docs
        self.N_words += total_new_tokens
        self.N_sentences += sum(len(doc) for doc in documents)
 
        old_total_length = self.avg_doc_len * old_N_docs
        new_total_length = sum(doc_lens)
        self.avg_doc_len = (old_total_length + new_total_length) / self.N_docs

        # Recover original DF counts from IDF
        old_df_counts = { term: max(1, int(round(old_N_docs / idf)))
                          for term, idf in self.idf.items() }

        # Merge new and old DF counts
        merged_df = Counter(old_df_counts)
        merged_df.update(new_df_counts)

        # Compute new IDF
        self.idf = { term: self.N_docs / df
                     for term, df in merged_df.items() }

        self.prune_idf()


    def prune_idf(self):
        """Prune IDF entries to the actual model vocabulary (remove tokens
        that were filtered out by gensim during `super().__init__`).
        """

        if getattr(self, "idf", None) is None:
            return

        vocab = set(self.wv.key_to_index.keys())
        original = len(self.idf)
        self.idf = {t: v for t, v in self.idf.items() if t in vocab}
        print(f"pruned idf: {original} -> {len(self.idf)} terms")


    def retrain(self, corpus_iterable: list[list[str]], **kwargs) -> tuple[int, int]:

        # Flatten docs into a list of sentences
        new_corpus = [sentence for document in corpus_iterable for sentence in document]

        # Add new vocabulary from new corpus
        self.build_vocab(new_corpus, update=True)

        # Continue training
        loss_logger = LossLogger()
        result = self.train(corpus_iterable=new_corpus, total_examples=len(new_corpus), epochs=self.epochs,
                               callbacks=[loss_logger], compute_loss=True, **kwargs)
        
        if getattr(self, "idf", None) is not None:
            self.update_idf(corpus_iterable)
        else:
            self.N_docs += len(corpus_iterable)
            self.N_words += sum(len(sentence) for doc in corpus_iterable for sentence in doc)
            self.N_sentences += sum(len(doc) for doc in corpus_iterable)

        self.save(self.pathname)

        return result


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


    def get_wordvec(self, word: str, embed:str = "IN", normalize: bool = True) -> np.ndarray[np.float32] | None:
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
                if hasattr(self, 'syn1'):
                    # Model was trained with hierarchical softmax
                    embedding = self.syn1
                elif hasattr(self, 'syn1neg'):
                    # Model was trained with negative sampling
                    embedding = self.syn1neg
                else:
                    raise RuntimeError("No output embedding matrix found in the model")

                vec = embedding[self.wv.key_to_index[x]].astype(np.float32)
            elif embed == "IN":
                vec = self.wv[x].astype(np.float32)
            else:
                raise ValueError("Invalid option")

            if normalize:
                norm = np.linalg.norm(vec)

            return vec / norm if normalize and norm > 0. else vec
        else:
            return None


    def get_features(self, tokens: list[str], embed: str = "IN", use_sif: bool = False, sif_smoothing: float = 1e-3) -> np.ndarray[np.float32]:
        """Calls [core.nlp.Word2Vec.get_wordvec][] over a list of tokens and returns a single vector representing the whole list.

        Arguments:
            tokens: list of text tokens.
            embed: see [core.nlp.Word2Vec.get_wordvec][]
            use_sif: Use SIF weighting on each term when embedding a full sentence
            or document. See [core.nlp.Word2Vec.SIF][].
            sif_smoothing: The SIF smoothing coefficient.
        Returns:
            the normalized centroid of word embedding vectors associated with the input tokens 
            (aka the average vector), or the null vector if no word from the list was found in dictionnary.
        """
        features = np.zeros(self.vector_size, dtype=np.float32)
        weights = 0.

        for token in tokens:
            vector = self.get_wordvec(token, normalize=True, embed=embed)
            if vector is None:
                continue

            weight = self.SIF(token, a=sif_smoothing) if use_sif else 1.0
            features += vector * weight
            weights += weight

        if weights > 0:
            features /= weights

        # Normalize by the L2 norm
        features /= (np.linalg.norm(features) + 1e-8)
        
        return features


    def SIF(self, token: str, a: float = 1e-3) -> float:
        """Smooth inverse frequency weighting

        Taken from _A simple but tough-to-beat baseline for sentence embeddings_,
        Sanjeev Arora, Yingyu Liang, Tengyu Ma. https://openreview.net/pdf?id=SyK00v5xx
        
        This helps refining semantics by under-weighting stopwords,
        however it's unsuited for File Information Retrieval (search engines)
        because it over-smoothen the embedding space geometry and hinders
        relevance discrimination with regard to a query.

        Arguments:
            token: the token to weight. It should be in the model vocabulary.
        
        Return:
            The SIF weight associated with the token or 0. if the token was not found in the vocabulary.
        """
        if getattr(self, "idf", None) is None:
            raise RuntimeError("IDF stats were not computed for this model")

        # Note: SIF technically use token frequency in the training dataset, 
        # aka the number of times the token is found in the whole training bag of words.
        # Here we use document frequency, aka number of documents in which the token is found (TF-IDF style).
        freq = 1. / self.idf.get(token, 0.0)
        return a / (a + freq)


    def tokens_to_indices(self, tokens: list[str]) -> np.ndarray[np.int32]:
        """Convert a list of tokens to a list of their index number in the Word2Vec vocabulary.
        This yields a more compact, albeit purely symbolic, representation of a tokenized document
        as a series of integers.

        The conversion is reversible and the original token can be found with `self.wv.index_to_key[i]`,
        where `i` is the index number output (for each token) from here.

        Return:
            the list of indices as 32 bits integers, meaning the Word2Vec vocabulary needs to contain fewer
            than 4.29 billions words.
        """
        return np.array([self.wv.get_index(token)
                         for token in tokens
                         if token in self.wv], dtype=np.int32)


class Doc2Vec(gensim.models.doc2vec.Doc2Vec):
    @timeit()
    def __init__(self,
                 training_set: list[web_page],
                 tags_list: list[any],
                 tokenizer: Tokenizer = None,
                 name: str = "doc2vec",
                 ):

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer()

        sentences = [post["parsed"]
                     if "parsed" in post else self.tokenizer.normalize_text(clean_whitespaces(post["content"]))
                     for post in training_set]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            sentences = executor.map(self.tokenizer.tokenize_document_flat, sentences, chunksize=32)

        train_corpus = [gensim.models.doc2vec.TaggedDocument(tokens, tags)
                        for tokens, tags in zip(sentences, tags_list)]

        super().__init__(vector_size=512, epochs=20, window=7, min_count=10, sample=0.00001, dm_concat=1)

        self.build_vocab(train_corpus)
        print("vocab built")

        self.train(train_corpus, total_examples=self.corpus_count, epochs=self.epochs)
        print("model trained")

        self.save(get_models_folder(name))
        print("model saved")

        ranks = []
        for doc_id in range(len(train_corpus)):
            inferred_vector = self.infer_vector(train_corpus[doc_id].words, epochs=20)
            sims = self.dv.most_similar([inferred_vector], topn=len(self.dv))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)

        counter = Counter(ranks)
        print(counter)


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
        # del self.word2vec.syn1neg

        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"))


    def get_features_parallel(self, post: Data) -> tuple[str, str]:
        """Thread-safe call to `.get_features()` to be called in multiprocessing.Pool map"""
        tokens = self.word2vec.tokenizer.tokenize_document_flat(post.text)
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
        tokens = self.word2vec.tokenizer.tokenize_document_flat(post)
        features = dict(enumerate(self.word2vec.get_features(tokens)))
        return super().classify(features)


    def prob_classify(self, post: str) -> tuple[str, float]:
        """Apply a label on a post based on the trained model and output the probability too."""
        tokens = self.word2vec.tokenizer.tokenize_document_flat(post)
        features = dict(enumerate(self.word2vec.get_features(tokens)))

        # This returns a weird distribution of probabilities for each label that is not quite a dict
        proba_distro = super().prob_classify(features)

        # Build the list of dictionnaries like `label: probability`
        output = {i: proba_distro.prob(i) for i in proba_distro.samples()}

        # Finally, return label and probability only for the max proba of each element
        return (max(output, key=output.get), max(output.values()))


DOCUMENTS = None
TOKENIZER = None


def clean_worker(i):
    text = DOCUMENTS[i]["content"]
    text = str(text).encode("utf-8", "replace").decode("utf-8")
    return i, clean_whitespaces(text)

def normalize_worker(i):
    doc = DOCUMENTS[i]
    text = (
        str(doc["title"]).encode("utf-8", "replace").decode("utf-8")
        + "\n\n"
        + doc["content"]
    )
    return i, TOKENIZER.normalize_text(text)

@timeit()
def batch_normalize(documents: list[web_page], tokenizer: Tokenizer, chunksize: int = 512, cores: int | bool = False) -> list[web_page]:
    global DOCUMENTS, TOKENIZER

    DOCUMENTS = documents
    TOKENIZER = tokenizer

    num_cpu = cores or os.cpu_count()

    ctx = multiprocessing.get_context("fork")

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        mp_context=ctx,
    ) as executor:

        # Cleanup whitespaces
        for i, cleaned in executor.map(
            clean_worker,
            range(len(documents)),
            chunksize=chunksize
        ):
            documents[i]["content"] = cleaned

        # Normalize text
        for i, parsed in executor.map(
            normalize_worker,
            range(len(documents)),
            chunksize=chunksize
        ):
            documents[i]["parsed"] = parsed

    return documents


def _init_worker(tokenizer):
    global TOKENIZER
    TOKENIZER = tokenizer


def _batch_tokenize_worker(inputs: tuple[int, str, str, str | None]) -> tuple[str, str, int]:
    # Unroll SQL params
    rowid, content, parsed, lang = inputs

    # Guess lang
    lang_mem = lang
    lang = parse_lang_to_iso639_1(lang)
    if lang is None:
        lang = detect_language(content)
    if lang is None:
        lang = detect_language(parsed)

    # Tokenize without stemming/lemmatization and keep stopwords
    tokenized = TOKENIZER.tokenize_document_per_sentence(parsed, lang, 
                                                         normalize=False, meta_tokens=True, 
                                                         stem=False, remove_stopwords=False)

    return lang, tokenized, rowid # keep order in sync with updating SQL query



@timeit()
def batch_tokenize(db: sqlite3.Connection, tokenizer: Tokenizer, chunksize: int = 256, urls: list[str] | None = None):
    """Tokenize a list of `web_pages` in parallel, in a RAM-friendly way.

    Populate the `tokens` key to `web_page` elements (taken from a list), containing the tokenized
    parsed content as a list of lists (each sentence is a list of tokens, the document is a list of sentences).
    Tokens are taken from the concatenated `web_page` `title` and `parsed` values, where `parsed` is the
    parsed `content`, pre-normalized through the tokenizer method.

    The list is processed in parallel, broken down in chunks, saved temporarily and individually to disk cache.
    You need to ensure you have enough space on your disk. The function doesn't check for it.

    Arguments:
        urls: list of URLs to tokenize. If None, the whole database is processed.
    """
    # Prepare SQLite DB for max throughput in batch environment
    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA synchronous=NORMAL;")
    db.execute("PRAGMA temp_store=MEMORY;")

    num_cpu = os.cpu_count()
    batch_size = (num_cpu or 1) * chunksize

    if urls is not None:
        cursor = db.execute(
            f"""
            SELECT rowid, content, parsed, lang
            FROM pages
            WHERE url IN ({ ",".join(["?" for _ in urls]) })
            """,
            urls,
        )
    else:
        cursor = db.execute('SELECT rowid, content, parsed, lang FROM pages')

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_worker,
        initargs=(tokenizer,),
    ) as executor:       
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            results = executor.map(_batch_tokenize_worker, batch, chunksize=chunksize)
            db.executemany('UPDATE pages SET lang=?, tokenized=? WHERE rowid=?', results)
            db.commit()


@timeit()
def batch_vectorize(db: sqlite3.Connection, word2vec: Word2Vec, chunksize: int = 256):
    """Vectorize a column of the `db` database using the provided `word2vec` model
    using all available cores.

    Works on the `tokenized` column of the database and writes the `vectorized` column.
    Vectors are normalized as per `nlp.Word2Vec.get_features()` output.    
    """
    num_cpu = os.cpu_count()
    cursor = db.execute('SELECT rowid, tokenized FROM pages')
    batch_size = num_cpu * chunksize

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            ids = [item[0] for item in batch]
            parsable = [[token for sentence in item[1] for token in sentence] for item in batch]
            vectors = executor.map(word2vec.get_features, parsable, ["OUT" for i in ids], chunksize=chunksize)
            del parsable

            db.executemany('UPDATE pages SET vectorized=? WHERE rowid=?', list(zip(vectors, ids)))
            db.commit()

            del ids
            del vectors

            #TODO:
            #indices = word2vec.tokens_to_indices(tokens)

@timeit()
def batch_guess_dates(db: sqlite3.Connection, chunksize: int = 256):
    """Refresh the datetime database objects for each row by reading the textual date extracted from pages.    
    """
    num_cpu = os.cpu_count()
    cursor = db.execute('SELECT rowid, date FROM pages')
    batch_size = num_cpu * chunksize

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            ids = [item[0] for item in batch]
            parsable = [item[1] for item in batch]
            datetimes = executor.map(guess_date, parsable, chunksize=chunksize)
            del parsable

            db.executemany('UPDATE pages SET datetime=? WHERE rowid=?', list(zip(datetimes, ids)))
            db.commit()

            del ids
            del datetimes

            #TODO:
            #indices = word2vec.tokens_to_indices(tokens)
