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
    "definitely", "indeed", "for", "some", "everytime", "every",
    "on", "at", "by", "out", "they", "than", "up",
    "well", "ok", "me", "please",
    "either", "both", "lot", "yet", "too", "each", "far", "again",
}

STOPWORDS_FR = {
    # FR
    "ça", "à", "au", "aux", "que", "qu'il", "qu'elle", "qu'", "ce", "cette", "ces", "cettes", "cela", "ceci", "qu'on", "qu'un",
    "le", "la", "les", "l", "l'", "de", "du", "d'", "un", "une", "des", "toi", "moi", "eux", "te", "qu",
    "mon", "ma", "mes", "ta", "tes", "sa", "ses", "leur", "leurs", "votre", "vos", "c'est à dire", "lui", "c'est-à-dire",
    "y", "là", "ici", "là-bas", "c'est", "bien",
    "parfois", "certain", "certains", "certaine", "certaines", "quelque", "quelques", "nombreux", "nombreuses", "peu", "plusieurs",
    "beaucoup", "tout", "toute", "tous", "toutes", "aucun", "aucune", "comme", "si",
    "en", "dans", "or", "ou", "où", "just", "et", "alors", "parce", "seulement", "ni", "car",
    "très", "donc", "pas", "mais", "même", "aussi", "avec", "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "pour", "sur", "par", "se", "j'ai", "j", "ai", "suis",
    "d'un", "d'une",
    "ci-dessus", "ci-dessous",
    "lequel", "duquel", "auquel", "laquelle", "auquel", "lesquels", "duquel",
    "lesquelles", "auxquels", "auxquelles", "desquels", "desquelles", "desquel",
    "quelquefois", "parfois", "toujours",
}

STOPWORDS_PUNCT = {
    # Lonely punctuation without semantic meaning
    "(", ")", "{", "}", "[", "]", ",", ":", "/", ";", "_", "-", "<", ">", "*", "'", '"', "&", "@", "``", "‘",
    # Misc meaningless stuff
    "http", "https",
    "oh", "ah", "ha", "heh", "re", "eh", "huh", "uh", "wow", "ow", "dang", "um",
    "yay", "ugh", "hehe", "hehehe",
}

STOPWORDS = set(list(STOPWORDS_PUNCT))

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
REPLACEMENTS = {
    "photo": "photograph",
    "eg": "e.g",
    "yeah": "yes",
    "yea": "yes",
    "yep": "yes",
    "u": "you",
    "hi": "hello",
    "hey": "hello",
    "ya": "you",
    "ur": "your",
    "hiya": "hello",
    "ve": "have",
    "n't": "not",

    "pdfs": "pdf",

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
    # neuf : collision avec neuf=nouveau/récent
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
    "premier": "_ORDINAL_",
    "2e": "_ORDINAL_",
    "2ème": "_ORDINAL_",
    "deuxième": "_ORDINAL_",
    "3e": "_ORDINAL_",
    "3ème": "_ORDINAL_",
    "troisième": "_ORDINAL_",
    "4e": "_ORDINAL_",
    "4ème": "_ORDINAL_",
    "quatrième": "_ORDINAL_",
    "5e": "_ORDINAL_",
    "5ème": "_ORDINAL_",
    "cinquième": "_ORDINAL_",
    "6e": "_ORDINAL_",
    "6ème": "_ORDINAL_",
    "sixième": "_ORDINAL_",

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

    "dollar": "_MONEY_",
    "€": "_MONEY_",
    "£": "_MONEY_",
    "$": "_MONEY_",

    "quelle": "quel",
    "quelles": "quel",
    "quels": "quel",
}

# Normalize contractions and abbreviations
ABBREVIATIONS = {
    " n'": " ne ",
    " c'": " ce ",
    " j'": " je ",
    " t'": " te ",
    "qu'": "que ",
    " s'": " se ",
    " l'": " le ",
    " d'": " de ",
    " m'": " me ",

    "what's": "what is",
    "what're": "what are",
    "who's": "who is",
    "who're": "who are",
    "where's": "where is",
    "where're": "where are",
    "when's": "when is",
    "when're": "when are",
    "how's": "how is",
    "how're": "how are",

    "i'm": "i am",
    "we're": "we are",
    "you're": "you are",
    "they're": "they are",
    "it's": "it is",
    "he's": "he is",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "there're": "there are",

    "i've": "i have",
    "we've": "we have",
    "you've": "you have",
    "they've": "they have",
    "who've": "who have",
    "would've": "would have",
    "not've": "not have",

    "i'll": "i will",
    "we'll": "we will",
    "you'll": "you will",
    "he'll": "he will",
    "she'll": "she will",
    "it'll": "it will",
    "they'll": "they will",

    "isn't": "is not",
    "wasn't": "was not",
    "aren't": "are not",
    "weren't": "were not",
    "can't": "can not",
    "cannot": "can not",
    "couldn't": "could not",
    "don't": "do not",
    "didn't": "did not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "doesn't": "does not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "won't": "will not",
    "shan't": "shall not",

    " didnt ": " did not ",
    " cant ": " can not ",
    " wont ": " will not ",
    " shouldnt ": " should not ",
    " shant ": " shall not ",

    "'cause": "because",
    "'till": "until",
    "till": "until",
    "outta": "out of",
    "gonna": "going to",
    "wanna": "want to",

    "(s)": "",
}

meta_tokens = [
    "_NUMBER_", "_PATH_", "_ORDINAL_", "_TIME_", "_DATE_", "_FILESIZE_", "_USER_", "_URL_"]

punctuation = [
    ".", "?", "!", "...", "+", "%"]

english_adv = [
    "reali",
]
determinants = [
    "l", "le", "la", "de", "du", "un", "une", "a", "aux",
    "th", "of", "a", "an"]

demonstratives = [
    "ce", "cet", "ceci", "cela", "c'est", "ça",
    "thi", "thos", "th" ]

misc = [
    "oui", "non",
    "yes", "no",
    "ok", "etc", "eg", "e.g",
    "suntil", # ??? - probably tokenizer hickup
    "both", "neither", "either",
    "via",
    "abl", "posibl",
    "thank", "merci", "hi", "hello", "bonjor", "bon", "pleas",
    "cas", "onc", "thing", "chos",
    "within", "easi", "rath", # rather
    "plea",
    "tim", "foi", "mot", "word",
    ]

conjunctions_1 = [
    "mai", "ou", "et", "donc", "or", "ni", "car",
    "but", "wher", "and", "thu", "therefor", "so", "henc", "nor", "becaus"]


conjunctions_2 = [
    "qui", "que", "quoi", "quand", "pourquoi", "com", "dont", "où", "pendant", "ou", "lequel", "laquel", "duquel", "desquel", "lesquel", "auxquel", "parc",
    "who", "that", "than", "what", "when", "whi", "how", "which", "wh", "whil", "or", "whos", "becaus"]


verbs = [
    "etr", "sui", "e", "est", "som", "et", "sont",
    "be", "am", "are", "i", "wa", "wer", "been",
    "avoir", "ai", "a", "avon", "avez", "ont", "eu",
    "hav", "ha", "had",
    "peux", "peut", "pouvon", "pouvez", "peuvent",
    "can",
    "veux", "veut", "voulons", "voulez", "veulent",
    "wil", "want", "wanted", "vouloir", "veux", "veut", "veulent",
    "fait", "fai",
    "do", "did", "mak", "mad",
    "fait", "fair", # collides with "foire" but probably not a big deal
    "sai", "sait", "savon", "savez", "savent",
    "know", "knew",
    "pens", "penson", "pensez", "pensent",
    "think", "thought",
    "would", "could", "should", "must",
    "go", "get", "went",
    "mai", "might", "mayb",
    "tri", "sai", "let", "keep",
    "put", "metr", "mi",
    "utilis",
    "mov", "act",
    "feel", "felt",
    "mean",
    "seem", "apear",
    "tak", "taken", "took", "prendr", "prit", "pri", "prend",
    "seen", "see", "saw", "voir", "vu", "voyant",
    "sai", "said", "dir", "dit", "di",
    "alow", "permetr", "permi", "permetant",
    ]

adverbs = [
    "ici", "la", "propos", "ausi", "trop", "seul", "autr", "mem", "maintenant", "tel", "autour", "cependant", "toujour", "jamai", "parfoi", "souvent", "encor", "asez", "tr", "parmi", "maintenant", "presq", "cependant",
    "her", "ther", "about", "also", "too", "onl", "other", "any", "som", "even", "now", "such", "around", "however", "alwai", "everytim", "anytim", "never", "sometim", "often", "again", "enough", "instead", "veri", "among", "yet", "almost", "although", "though", "someon", "everyon", "anyon", "somewh", "everywh", "anywh",]

quantifiers = [
    # Note : "n'importe" (as "import") is not included for obvious reasons
    "certain", "quelqu", "tou", "plusieur", "beaucoup", "chaqu", "peu", "tout", "uniq", "seul",
    "some", "someth", "al", "everyth", "any", "anyth", "several", "much", "many", "lot", "each", "everi", "few", "litl", "al", "onli", "alon",
]

prepositions = [
    "vers", "dan", "au", "à", "en", "pour", "with", "without", "sur", "hors", "ne", "pas", "ni", "tant", "com", "depui", "si", "alor", "autr", "chez", "par", "traver", "lor", "lorsqu", "jusque",
    "to", "in", "into", "for", "avec", "san", "on", "out", "over", "not", "neither", "nor", "as", "sam", "like", "from", "sinc", "if", "wheth", "then", "els", "at", "by", "through",
    "just", "until",]

pronouns = [
    "je", "tu", "il", "el", "nou", "vou", "moi",
    "i", "you", "he", "her", "it", "we", "they",
    "me", "him", "her", "us", "them", "myself", "yourself", "themself", "themselv", "himself", "herself",
    "s", "t",
    # yes, the followings are not technically pronouns
    # Notes :
    #   "ton" is not there : it collides with "tone" and "ton" as "tonalité"
    #   "son" is not there : it collides with "son" (child) and "son" as "sonorité"
    "mon", "ma", "m", "mien", "ta", "t", "tien", "sa", "s", "notr", "no", "votr", "vo", "leur", "lui", "t",
    "mi", "min", "your", "his", "its", "our", "their", ]

adv_2 = ["fairli", "reali", "likeli", "probabli", "mostli", ]
TOPICS_STOPWORDS = set( pronouns + prepositions + adverbs + verbs + conjunctions_2 + conjunctions_1 + misc + determinants + quantifiers + demonstratives + punctuation + meta_tokens + adv_2)
