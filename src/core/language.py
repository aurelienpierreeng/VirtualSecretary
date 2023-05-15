from nltk.corpus import stopwords

# The set of languages supported at the same time by NLTK tokenizer, stemmer and stopwords data is not consistent.
# We build the least common denominator here, that is languages supported in the 3 modules.
# See SnowballStemmer.languages and stopwords.fileids()
_supported_lang = {'danish', 'dutch', 'english', 'finnish', 'french',
                   'german', 'italian', 'norwegian', 'portuguese', 'spanish', 'swedish'}

# Stopwords to remove
# We remove stopwords only if they are grammatical operators not briging semantic meaning.
# NLTK stopwords lists are too aggressive and will remove auxiliaries and wh- tags
# We keep nounds, verbs, question and negation markers.
# We ditch quantifiers, adverbs, pronouns.
# Those assume the following pipeline :
# typography_undo -> prefilter_token -> tokenize -> strip(".:'^ ") -> (len() > 1)
STOPWORDS_EN = {
    # EN
    "it", "it's", "that", "this", "these", "those", "that'll", "that's", "there", "here",
    "the", "a", "an", "one", "any", "all", "none", "such", "to", "which", "whose", "much", "many", "several", "few", "little",
    "always", "never", "sometimes",
    "my", "mine", "your", "yours", "their", "theirs", "his", "hers", "its", "us",
    "you", "he", "she", "her", "them", "we", "our",
    "also", "like", "get",
    "with", "in", "but", "so", "just", "and", "only", "because", "of", "as", "very", "from", "other",
    "if", "then", "however", "maybe", "now", "really", "actually", "something", "everything", "later", "sooner", "late", "soon",
    "probably", "guess", "perhaps", "still", "though", "even",
    "definitely", "indeed", "for", "some", "everytime", "every", "am",
    "on", "at", "by", "out", "they", "than", "http", "up",
    "well", "ok", "me", "please",
    "either", "both", "lot", "yet", "too", "each", "far", "again", "s",
}

STOPWORDS_FR = {
    # FR
    "ça", "à", "au", "aux", "que", "qu'il", "qu'elle", "qu'", "ce", "cette", "ces", "cettes", "cela", "ceci", "qu'on", "qu'un",
    "le", "la", "les", "l", "l'", "de", "du", "d'", "un", "une", "des", "toi", "moi", "eux", "te",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "leur", "leurs", "votre", "vos", "c'est à dire", "lui", "c'est-à-dire",
    "y", "là", "ici", "là-bas", "c'est", "bien",
    "parfois", "certain", "certains", "certaine", "certaines", "quelque", "quelques", "nombreux", "nombreuses", "peu", "plusieurs",
    "beaucoup", "tout", "toute", "tous", "toutes", "aucun", "aucune", "comme", "si",
    "en", "dans", "or", "ou", "où", "just", "et", "alors", "parce", "seulement", "ni", "car",
    "très", "donc", "pas", "mais", "même", "aussi", "avec", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "pour", "sur", "par", "se", "j'ai", "j", "ai", "suis",
    "d'un", "d'une",
    "ci-dessus", "ci-dessous",
}

STOPWORDS_PUNCT = {
    # Lonely punctuation without semantic meaning
    "(", ")", "{", "}", "[", "]", ",", ":", "/", ";", "_", "-", "\\", '“', '”', '‘', '’', "<", ">", "*", "'", '"',
    "''", "``", "&",
    # Misc meaningless stuff
    "http", "https",
    "oh", "ah", "ha", "heh",
}

STOPWORDS = set(list(STOPWORDS_EN) +
                list(STOPWORDS_FR) + list(STOPWORDS_PUNCT))

# Static dict of stopwords for language detection, inited from NLTK corpus
STOPWORDS_DICT = {language: list(stopwords.words(
    language)) for language in stopwords.fileids() if language in _supported_lang}

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
STOPWORDS_DICT["french"] += STOPWORDS_FR

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
STOPWORDS_DICT["english"] += STOPWORDS_EN

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
STOPWORDS_DICT = {language: set(
    STOPWORDS_DICT[language]) for language in STOPWORDS_DICT}

# Abbreviations and common typos, as `original: replacement`
# Those replacements assume the following pipeline :
# typography_undo -> prefilter_token -> tokenize -> lemmatize -> stem
#                                          |----[ replace ]-------|
# Replacement are the stemmed output of Lemmatizer + SnowballStemmer applied after tokenizer
REPLACEMENTS = {# Abbreviations
                "photo": "photograph",
                "doesnt": "doesn",
                "doesn't": "doesn",
                "whi": "why",
                "aurelien": "aurélien",
                "eg": "e.g",
                "'m": "am",
                "ve": "have",
                "n't": "not",
                "ll": "will",
                "u": "you",
                "hi": "hello",
                "yeah": "yes",

                # Fix the stemmer fuck-ups
                "cinema": "cinem",
                "photographi": "photograph",
                "camera": "camer",

                "zero": "_NUMBER_",
                "one": "_NUMBER_",
                "two": "_NUMBER_",
                "three": "_NUMBER_",
                "four": "_NUMBER_",
                "five": "_NUMBER_",
                "six": "_NUMBER_",
                "seven": "_NUMBER_",
                "eight": "_NUMBER_",
                "nine": "_NUMBER_",
                "ten": "_NUMBER_",
                "eleven": "_NUMBER_",
                "twelve": "_NUMBER_",

                "un": "_NUMBER_",
                "deux": "_NUMBER_",
                "trois": "_NUMBER_",
                "quatre": "_NUMBER_",
                "cinq": "_NUMBER_",
                # six : déjà géré en anglais
                "sept": "_NUMBER_",
                "huit": "_NUMBER_",
                # neuf : collision avec neuf:nouveau/récent
                "dix": "_NUMBER_",
                "onze": "_NUMBER_",
                "douze": "_NUMBER_",

                "1st": "_ORDINAL_",
                "first": "_ORDINAL_",
                "2nd": "_ORDINAL_",
                "second": "_ORDINAL_",
                "3rd": "_ORDINAL_",
                "third": "_ORDINAL_",
                "4th": "_ORDINAL_",
                "fourth": "_ORDINAL_",
                "5th": "_ORDINAL_",
                "fifth": "_ORDINAL_",
                "6th": "_ORDINAL_",

                "1er": "_ORDINAL_",
                "2e": "_ORDINAL_",
                "2ème": "_ORDINAL_",
                "3e": "_ORDINAL_",
                "3ème": "_ORDINAL_",
                "4e": "_ORDINAL_",
                "4ème": "_ORDINAL_",
                "5e": "_ORDINAL_",
                "5ème": "_ORDINAL_",
                "6e": "_ORDINAL_",
                "6ème": "_ORDINAL_",

                "lundi": "_DATE_",
                "mardi": "_DATE_",
                "mercredi": "_DATE_",
                "jeudi": "_DATE_",
                "vendredi": "_DATE_",
                "samedi": "_DATE_",
                "dimanche": "_DATE_",

                "monday": "_DATE_",
                "tuesday": "_DATE_",
                "wednesday": "_DATE_",
                "thursday": "_DATE_",
                "friday": "_DATE_",
                "saturday": "_DATE_",
                "sunday": "_DATE_",

                "janvier": "_DATE_",
                "février": "_DATE_",
                "fevrier": "_DATE_",
                "mars": "_DATE_",
                "avril": "_DATE_",
                "mai": "_DATE_",
                "juin": "_DATE_",
                "juillet": "_DATE_",
                "août": "_DATE_",
                "aout": "_DATE_",
                "septembre": "_DATE_",
                "octobre": "_DATE_",
                "novembre": "_DATE_",
                "décembre": "_DATE_",
                "decembre": "_DATE_",

                "today": "_DATE_",
                "yesterday": "_DATE_",
                "tomorrow": "_DATE_",
                "aujourd'hui": "_DATE_",
                "demain": "_DATE_",
                "hier": "_DATE_",

                "dollar": "_MONEY_"
                }
