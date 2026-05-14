"""
High-level natural language processing module for message-like (emails, comments, posts) input.

Supports automatic language detection, word tokenization and stemming for `'danish', 'dutch', 'english', 'finnish', 'french', 'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'`.

© 2023 - Aurélien Pierre
"""

from __future__ import annotations

import random
import regex as re
import os
import concurrent
import unicodedata as ud

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import gensim
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.phrases import Phrases, Phraser

import joblib
import pickle
import json

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
from .utils import get_models_folder, typography_undo, clean_whitespaces, timeit, guess_date, sanitize_unicode
from .language import *
from .crawler import web_page


latin_letters = {}

def _is_latin(uchr):
    try:
        return latin_letters[uchr]
    except KeyError:
        return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))


def _roman_chars(unistr):
    return [_is_latin(uchr) for uchr in unistr if uchr.isalpha()]


def parse_lang_to_iso639_1(value: str | None) -> str | None:
    """Normalize language identifier to ISO 639-1."""

    if not value:
        return None

    try:
        lang = langcodes.get(value)

        if lang.language and len(lang.language) == 2:
            return lang.language

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


@dataclass(slots=True)
class Lexicon:
    """
    Mutable token frequency index with canonicalization helpers for:
    - malformed n-grams,
    - merged/split variants,
    - plural compound normalization.

    Examples:
        liber_tarian  -> libertarian
        etres_humains -> etre_humain
    """

    counts: Counter[str] = field(default_factory=Counter)

    def update(self, corpus: Iterable[Iterable[str]]) -> None:
        """
        Update token frequencies from a corpus of tokenized sentences.

        Args:
            corpus:
                Iterable of tokenized sentences:
                [
                    ["this", "is", "a", "sentence"],
                    ["another", "sentence"]
                ]
        """
        for sentence in corpus:
            self.counts.update(sentence)


    def frequency(self, token: str) -> int:
        """Return token frequency."""
        return self.counts[token]


    def exists(self, token: str) -> bool:
        """Check whether a token exists in the lexicon."""
        return token in self.counts
    

    def prune(self, min_count: int = 10) -> None:
        """
        Remove all entries whose frequency is lower than `min_count`.

        Args:
            min_count:
                Minimum frequency to keep.
        """

        self.counts = Counter({
            token: count
            for token, count in self.counts.items()
            if count >= min_count
        })
    

    @staticmethod
    def _singularize(token: str) -> str:
        """
        Very lightweight EN/FR singularization heuristic.

        Conservative on purpose:
        avoids damaging valid words.
        """

        # French plurals
        if token.endswith("aux") and len(token) > 4:
            return token[:-3] + "al"

        if token.endswith("s") and len(token) > 3:
            return token[:-1]

        # English plurals
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"

        return token


    def resolve_token(self, token: str, separator: str = "_", min_ratio: float = 1.0) -> str:
        """
        Attempt to canonicalize malformed n-grams.

        Operations:
        1. malformed n-grams:
            liber_tarian -> libertarian

        2. plural compound reduction:
            etres_humains -> etre_humain

        Strategy:
        - if token exists already -> keep it
        - otherwise:
            - remove separators,
            - check if merged variant exists,
            - compare frequencies,
            - prefer merged form if sufficiently frequent.

        Args:
            token:
                Token to canonicalize.

            separator:
                N-gram separator.

            min_ratio:
                Require merged token frequency to be at least
                `min_ratio` times the split variant frequency.

                Helps avoid false positives.

        Returns:
            Canonicalized token.
        """

        # Fast path. Ignore meta-tokens which have trailing and leading _
        if separator not in token[1:-1]:
            return token

        original_freq = self.counts.get(token, 0)

        candidates: list[str] = []

        # Candidate 1:
        # merge the compound entirely
        # liber_tarian -> libertarian
        merged = token.replace(separator, "")

        if merged in self.counts:
            candidates.append(merged)

        # Candidate 2:
        # singularize each compound segment
        # etres_humains -> etre_humain
        parts = token.split(separator)

        singular_parts = [
            self._singularize(part)
            for part in parts
        ]

        singular = separator.join(singular_parts)

        if singular != token and singular in self.counts:
            candidates.append(singular)

        # Candidate 3:
        # singularize + merge
        # etres_humains -> etrehumain
        merged_singular = "".join(singular_parts)

        if merged_singular in self.counts:
            candidates.append(merged_singular)

        if not candidates:
            return token

        # Prefer highest-frequency candidate
        best = max(
            candidates,
            key=lambda candidate: self.counts[candidate]
        )

        best_freq = self.counts[best]

        # If original token does not exist,
        # aggressively canonicalize
        if original_freq == 0:
            return best

        # Otherwise require statistical dominance
        if best_freq >= original_freq * min_ratio:
            return best

        return token


    def canonicalize_sentence(self, sentence: list[str], separator: str = "_", min_ratio: float = 1.0) -> list[str]:
        """
        Canonicalize all tokens in a sentence.
        """

        return [
            self.resolve_token(
                token,
                separator=separator,
                min_ratio=min_ratio,
            )
            for token in sentence
        ]


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

        if self.abbreviations:
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
            # French words contractions : expand for generality
            FRANCAIS: r"\1 ",
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
        return typography_undo(document.lower())


    def normalize_token(self, 
                        word: str, 
                        language: str | None, 
                        normalize: bool = True,
                        meta_tokens: bool = True, 
                        stem: bool = True,
                        remove_stopwords: bool = True) -> str | None:
        """Return normalized, lemmatized and stemmed word tokens, where dates, times, digits, monetary units 
        and URLs have their actual value replaced by meta-tokens designating their type. 
        Stopwords ("the", "a", etc.), punctuation etc. is replaced by `None`, which should be filtered out at the next step.

        Arguments:
            word (str): tokenized word in lower case only.
            language (str): the ISO 369-1 language code used to remove typical stopwords.
            normalize (str): remove punctuation and leading/trailing symbols.
            meta_tokens (bool): replace string patterns by meta_tokens
            stem (bool): remove word suffixes, double consonnants, etc.
            remove_stopwords (bool): remove stopwords

        NOTE:
            Tokenization is non-destructive (full sentences can be reconstructed entirely from token lists)
            if `normalize=False`, `meta_tokens=False`, `stem=False` and `remove_stopwords=False`. In this setting,
            only 1:1 token replacements defined in `self.replacements` will be applied, which can allow
            to replace abbreviations or accronyms.
            Other modes start generalizing semantics by removing meaning.

        Examples:
            Meta-tokens:
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
            
        # Find out if this is an n-gram that also exists as a singleton word.
        # Wrong splitting of compound words and hyphenation, especially in OCRed PDF
        # will often duplicate words as n-grams, which challenges Word2Vec learning.
        # This helps generalizing by catching this error.
        string = self.vocabulary.resolve_token(string)

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

        # For n-grams, limit post-processing to stopwords removal
        # We do lazy detection of undescores within the string,
        # which will also track function names and technical stuff.
        # We probably don't wont to filter those either anyway.
        is_ngram = "_" in string[1:-1]
        
        # Remove Markdown markers for _italics_ and __bold__
        # Note: internal under_scores and da-shes are already handled in metatokens regex loop.
        # This is mandatory since we use `_` as n-gram delimiter later.
        if not is_ngram:
            string = string.strip("_ ")

        # Last chance of identifying meta-tokens in an atomic way
        if meta_tokens and not is_ngram:
            for key, value in self.internal_meta_tokens.items():
                # Note: since Python 3.8 or so, dictionnaries are ordered.
                # Treating the pre-processing pipeline as dict wouldn't work for ealier versions.
                if key.match(string, timeout=10, concurrent=True):
                    return value
                
        # Check if string is a stopword from our custom list
        if remove_stopwords:
            # Language-specific stopwords
            if self.lang_stopwords and language in self.lang_stopwords:
                if string in self.lang_stopwords[language]:
                    return None
                
            # Language-agnostic stopwords
            if self.stopwords and string in self.stopwords:
                return None

        # Lemmatize / Stem
        if stem and not is_ngram:
            string = self.lemmatize(string)

        # Last chance to catch badly-hyphenated n-grams
        string = self.vocabulary.resolve_token(string)
            
        return string


    def tokenize_text(self, 
                      sentence: str, 
                      language: str | None = None, 
                      n_grams: bool = True,
                      normalize: bool = True,
                      meta_tokens: bool = True, 
                      stem: bool = True,
                      remove_stopwords: bool = True) -> list[str]:
        """Split an arbitrary text into normalized word tokens and meta-tokens.
        No sentence or paragraph detection will be attempted.

        Arguments:
            sentence: the input single sentence.
            n_grams: apply n-grams detection and collapsing on tokens. Need to have trained the 
            n-grams model with [self.train_ngrams][]
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided

        Returns:
            tokens (list[str]): the list of normalized tokens as a bag of words.
        """

        if n_grams and self.supports_ngrams:
            # First pass needs basic tokenization so the N-gram detection can pick up
            tokens = self.post_filter_tokens(tokenize_document_to_words(sentence, 
                                                                        language=language, 
                                                                        backend=self.backend),
                                             language,
                                             meta_tokens=meta_tokens,
                                             stem=False,
                                             normalize=False,
                                             remove_stopwords=False)

            # Spot n-grams into tokens and collapse them into a single token,
            # then finish normalization
            return self.post_filter_tokens(self.replace_ngrams(tokens),
                                            language,
                                            meta_tokens=meta_tokens,
                                            stem=stem,
                                            normalize=normalize,
                                            remove_stopwords=remove_stopwords)
        else:

            return self.post_filter_tokens(tokenize_document_to_words(sentence, 
                                                                      language=language, 
                                                                      backend=self.backend),
                                           language,
                                           meta_tokens=meta_tokens,
                                           stem=stem,
                                           normalize=normalize,
                                           remove_stopwords=remove_stopwords)


    def post_filter_tokens(self,
                           tokens: list[str], 
                           language: str | None = None,
                           meta_tokens: bool = True,
                           stem: bool = False,
                           normalize: bool = False,
                           remove_stopwords: bool = False) -> list[str]:
        """Apply a post-processing step (normalization, etc.) on an existing list of tokens.
        
        Arguments:
            See [core.nlp.Tokenizer.normalize_token][]
        """
        normalize_token = self.normalize_token

        return [
            t
            for token in tokens
            if (
                t := normalize_token(
                    token,
                    language,
                    meta_tokens=meta_tokens,
                    stem=stem,
                    normalize=normalize,
                    remove_stopwords=remove_stopwords
                )
            )
        ]
 

    def tokenize_document_flat(self, 
                                document:str, 
                                language: str | None = None, 
                                n_grams: bool = True,
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
            n_grams (bool): see [core.nlp.Tokenizer.tokenize_text][]
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided. The text is prefiltered with [self.prefilter][].

        Returns:
            tokens (list[str]): a 1D list of normalized tokens and meta-tokens.
        """
        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return self.tokenize_text(clean_text, language=language, meta_tokens=meta_tokens, 
                                  stem=stem, normalize=normalize, remove_stopwords=remove_stopwords,
                                  n_grams=n_grams)


    def tokenize_document_per_sentence(self, 
                                       document: str, 
                                       language: str | None = None, 
                                       n_grams: bool = True,
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
            n_grams (bool): see [core.nlp.Tokenizer.tokenize_text][]
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided. The text is prefiltered with [self.prefilter][]

        Returns:
            tokens: a 2D list of sentences (1st axis), each containing a list of normalized tokens and meta-tokens (2nd axis).
        """
        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return [self.tokenize_text(sentence, language=language, meta_tokens=meta_tokens, 
                                   stem=stem, normalize=normalize, remove_stopwords=remove_stopwords,
                                   n_grams=n_grams)
                for sentence in split_document_to_sentences(clean_text, language=language, backend=self.backend)]


    def tokenize_document_per_paragraph(self, 
                                        document: str, 
                                        language: str | None = None, 
                                        n_grams: bool = True,
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
            n_grams (bool): see [core.nlp.Tokenizer.tokenize_text][]
            others: see [core.nlp.Tokenizer.normalize_token][] arguments

        Note:
            the language is detected internally if not provided. The text is prefiltered with [self.prefilter][]

        Returns:
            tokens: a 2D list of paragraphs (1st axis), each containing a list of normalized tokens and meta-tokens (2nd axis).
        """
        clean_text = self.prefilter(document, meta_tokens=meta_tokens)
        return [self.tokenize_text(paragraph, language=language, meta_tokens=meta_tokens, 
                                   stem=stem, normalize=normalize, remove_stopwords=remove_stopwords,
                                   n_grams=n_grams)
                for paragraph in re.split(r'(?:\r\n|\r|\n){2,}', clean_text, concurrent=True)]


    def __init__(self,
                 meta_tokens: dict[re.Pattern, str] | None = None,
                 abbreviations: dict[str, str] | None = None,
                 replacements: dict[str, str] | None = None,
                 stopwords: set[str] | None = None,
                 lang_stopwords: dict[str, set[str]] | None = None,
                 backend: str = "blingfire"):
        """Pre-processing pipeline and tokenizer, splitting a string into normalized word tokens.

        Arguments:
            meta_token: the pipeline of regular expressions to replace with meta-tokens in documents.
            Keys must be `re.Pattern` declared with `re.compile()`, values must be meta-tokens assumed to be nested in underscores. 
            The pipeline dictionnary will be processed in the order of declaration, which relies on using Python >= 3.8 (making `dict` ordered by default). 
            If not provided, it is inited by default with a pipeline suitable for bilingual English/French language processing on technical writings (see notes).
            abbreviations (dict[str: str]): pipeline of abbreviations to replace, as `to_replace: replacement` dictionnary. 
            Will be processed in order of declaration.
            replacements: dictionnary used to replace 1:1 `key` with `value` as strings in tokens.
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
                # Missing/partial/invalid file pathes
                PARTIAL_PATH_REGEX: " _PATH_ ",
                # In-words dashes/compound words : replace by underscore for uniform handling with n-grams
                DASHES: "_",
            }
        else:
            self.meta_tokens_pipe = meta_tokens

        self.meta_tokens = {value.strip()
                            for value in self.meta_tokens_pipe.values()
                            if value.startswith(" _") and value.endswith("_ ")}


        self.abbreviations = abbreviations
        """Abbreviations and contractions to replace in full documents"""

        self.replacements = replacements
        """Arbitrary string replacements in single tokens"""

        self.stopwords = set(stopwords) if stopwords else None
        """Language-agnostic stopwords"""

        self.lang_stopwords = lang_stopwords
        """Language-specific stopwords"""

        self.supports_ngrams: bool = False
        """Whether or not the tokenizer has an embedded n-grams model"""

        self.ngrams_trie = {}
        """Prefix tree of known n-grams for efficient lookups"""

        self.vocabulary: Lexicon = Lexicon()
        """Known tokens, if trained for n-grams."""


    def save(self, name: str):
        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"), compress=0, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    @timeit()
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        try:
            model = joblib.load(get_models_folder(name) + ".joblib")
        except FileNotFoundError:
            model = joblib.load(get_models_folder(name) + ".joblib.bz2")
            
        if not isinstance(model, Tokenizer):
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))

        return model
    

    def members_from_ngram(self, token: str | None) -> list[str] | None:
        """Recover n-grams members from a single tokenized phrase, separated with `_`.
        This expects lower-case tokens, except for meta-tokens which are expected capitalized.

        Returns:
            the list of n-gram members, or None if the token was not an n-gram but a singleton.
        """
        if not token:
            return None
        
        # Split phrases on _ and keep only the non-empty tokens
        # Note: empty tokens are the _ separators that got removed
        members = [s for s in token.split("_") if s]

        # The above will destroy leading and trailing _ of our meta-tokens,
        # but we can safely rely on the assumption that only meta-tokens are
        # capitalized at this stage of the pipeline, so restore them.
        members = [f"_{t}_" 
                    if t.isupper() and len(t) > 1 
                    else t 
                    for t in members]
        
        if len(members) > 1:
            return members
        else:
            return None
            

    @timeit()
    def train_ngrams(self, 
                     sentences: list[str], 
                     connector_words: str = "", 
                     min_count=10,
                     threshold=0.7,
                     scoring="npmi"):
        """Train an n-gram model for bigrams and trigrams, detecting phrases like `New York City`
        as one single token.

        Arguments:
            sentences: training corpus,
            connector_words: a flat string of space-separated connector words that are allowed
            to join bigrams and trigrams in the target language, like `by` in `piece by piece`,
            others: see [gensim.models.phrases.Phrases][] documentation.

        Warning:
            N-gram training needs to run on single sentences, tokenized in a non-destructive way,
            meaning without stemming and punctuation removal. See [core.nlp.Tokenizer.normalize_token][] arguments.

        Note:
            - output an `ngrams` log file in the models folder containing the list of all n-grams found.
            - this can safely be called several time, for example once for each language. n-grams are appended to the existing list.

        """
        print(f"Training n-grams with {len(sentences)} sentences.")

        self.vocabulary.update(sentences)
        self.vocabulary.prune(min_count=min_count)

        connectors = frozenset(connector_words.split()) if connector_words else frozenset()

        # Train models
        bigrams = Phrases(sentences, min_count=min_count,
                          threshold=threshold, scoring=scoring,
                          connector_words=connectors,
                          delimiter="_")

        trigrams = Phrases(bigrams[sentences], min_count=min_count,
                          threshold=threshold, scoring=scoring,
                          connector_words=connectors,
                          delimiter="_")
        
        # As part of the meta-tokenization, we also replace dashes in compound words
        # by underscores. They will vote too here in Gensim stats for phrases.
        # So this helps generalizing n-grams with properly
        # and improperly dashed (and possibly hyphenated) individual tokens in corpora.
        ngrams: list[str] = []
        for k in (set(bigrams.export_phrases().keys()) | set(trigrams.export_phrases().keys())):
            # Split phrases on _ and keep only the non-empty tokens
            members = self.members_from_ngram(k)
            if members:
                ngrams.append("_".join(members))

        self.compile_ngrams(ngrams)
        self.supports_ngrams = True

    
    def compile_ngrams(self, ngrams: list[str]):
        """Build a nested n-grams dictionnary for efficient querying, like:
        ```
        {
            "new": {
                "york": {
                    "__value__": "new_york",
                    "city": {
                        "__value__": "new_york_city"
                    }
                }
            }
        }
        ```
        """

        # Hack for French n-grams starting with those:
        # we need to add them to valid word connectors in Phrases learning
        # because they are valid, but the algo allows them to appear as leading member.
        # Other connectors (de, le, etc.) behave properly, it's only the 
        # apostrophe that makes Gensim loose it.
        INVALID_START = {"l'", "d'"}

        for token in ngrams:
            members = self.members_from_ngram(token)
            node = self.ngrams_trie
            if members and members[0] not in INVALID_START:
                for token in members:
                    node = node.setdefault(token, {})

                node["__value__"] = "_".join(members)

        # Dump the JSON for debug
        with open(get_models_folder("ngrams.json"), "w", encoding="utf-8") as f:
            json.dump(self.ngrams_trie, f, ensure_ascii=False, indent=2)


    def replace_ngrams(self, tokens: list[str]) -> list[str]:
        """Identify n-grams among tokens and collapse them into single tokens.
        N-grams should have been trained before, with [self.train_ngrams][].

        Returns:
            the collapsed list of strings, or the original list if no n-grams
            was found or the n-grams model has not been trained.
        """
        if not self.supports_ngrams:
            return tokens

        out = []
        length = len(tokens)
        i = 0

        trie = self.ngrams_trie

        while i < length:
            node = trie

            j = i
            best = None
            best_j = i

            while j < length:
                token = tokens[j]

                # Descend from current node
                node = node.get(token)

                if node is None:
                    break

                value = node.get("__value__")

                if value is not None:
                    best = value
                    best_j = j + 1

                j += 1

            if best is not None:
                out.append(best)
                i = best_j
            else:
                out.append(tokens[i])
                i += 1

        return out


    def lookup_ngram(self, members: list[str] | tuple[str, ...]) -> str | None:
        """
        Lookup an n-gram in the trie from its token members.

        Arguments:
            members: the tokens iterable
        
        Returns:
            the collapsed n-gram if found in the trie, or `None` if the input members match
            no known n-gram.

        Example:
            lookup_ngram(("new", "york"))
            -> "new_york"

            lookup_ngram(("new", "york", "city"))
            -> "new_york_city"

            lookup_ngram(("foo", "bar"))
            -> None
        """

        node = self.ngrams_trie

        for member in members:
            node = node.get(member)

            if node is None:
                return None

        return node.get("__value__")


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