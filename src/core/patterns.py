"""
Contains global regular expression patterns re-used in the app.
You can use https://regex101.com/ to test these conveniently.

© 2023 - Aurélien Pierre
"""

import regex as re

regex_starter = r"(?<=^|\s|\[|\(|\{|\<|\'|\"|`|;|\>)"
"""Start of line, or start of document, or start of markup"""

regex_stopper = r"(?=$|\s|\]|\)|\}|\>|\'|\"|`|;|\<)"
"""End of line, or end of document, or end of markup"""

end_of_word = r"(?=$|\s|\]|\)|\}|\>|\'|\"|`|;|:|,|\?|\!|\.|\<)"
"""End of word, or end of line, or end of document, or end of markup"""

regex_algebra = r"[\+\-\=\≠\±]"
"""Algebraic signs"""

"""
Note : the *_FAST patterns are meant to be used on tokens (single words).
They expect to find full matches (aka the whole token matches the pattern).

The non *_FAST patterns can be used on tokens or on full documents. They try
to find matches anywhere, enclosed in whitespaces, parenthesis, brackets, etc.
That makes them slower but more generic.
"""

# Internet-specific patterns

regex_ip = r"((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[\da-f]{1,4}:){3,8}(?::?\/?[\da-f]{1,4})*))"
IP_PATTERN_FAST = re.compile(r"^%s$" % regex_ip, re.IGNORECASE)
IP_PATTERN = re.compile(r"%s%s%s" % (regex_starter, regex_ip, regex_stopper), re.IGNORECASE)
"""IPv4 and IPv6 patterns where the whole IP is captured in the first group."""

EMAIL_PATTERN = re.compile(r"<?([0-9a-z\-\_\+\.]+?@[0-9a-z\-\_\+]+(\.[0-9a-z\_\-]{2,})+)>?", re.IGNORECASE)
"""Emails patterns like `<me@mail.com>` or `me@mail.com` where the whole address is captured in the first group."""

regex_url = r"((?:http|ftp)s?)?:?\/\/([A-Za-z0-9-_.~]+)(?:\:[0-9]*)?(\/?[A-Za-z0-9-_.~:\/\[\]@!$&'()*+,;=%]*)(\?[A-Za-z0-9-_.~:\/\[\]@!$&'()*+,;=%]*)?(\#[A-Za-z0-9-_.~:\/\[\]@!$&'()*+,;=%]*)?"
URL_PATTERN_FAST = re.compile(r"^%s$" % regex_url, re.IGNORECASE)
URL_PATTERN = re.compile(r"%s%s%s" % (regex_starter, regex_url, end_of_word), re.IGNORECASE)
"""URL patterns like `http(s)://domain.ext/page/subpage?q=x&r=0:1#anchor` or `//domain.ext/page`.
URL must follow [RFC3986](https://www.rfc-editor.org/rfc/rfc3986#section-4.1), meaning query parameters
should be before anchors, if any. Relying on this assumption allows a faster regex parsing.

- the protocol (ftp, ftps, http, https) is captured as the first group,
- `domain.ext` is captured as the second group,
- `/page/etc` is the third group, including leading and trailing `/`,
- page query parameters `?s=x&r=0`, including `?`, is
the fourth group if the URL declares `...?params#anchor`,
- anchor `#anchor` is the fifth group, including `#`, if the URL declares `...?params#anchor`.

URLs are captured if they are:

 - alone on their own line,
 - enclosed in `{}`, `[]`, `()`
 - enclosed in whitespaces.

 Warning: URLs enclosed in `()`, `[]` and `{}` may retain the closing sign
 as part of the page name since `()` and `[]` are valid in URL pathes
 and parameters. This pattern will work on plain text only: Markdown, XML, HTML and JSON
 will need to be parsed ahead.
 """

MEMBERS_PATTERN = re.compile(r"(?<=[a-z])(\.)(?=[a-z])", re.IGNORECASE)
"""Domain patterns without leading protocol like `cdn.company.com`
or class members in object-oriented programming languages like `params.cookies.client`."""

# Date/time

date_regex = r"(?<=^|\s|\[|\(|\{|\<|\'|\"|`|;)((\d{1,4})(?:-|\/)(\d{2})(?:-|\/)(\d{2,4}))(?=$|\s|\]|\)|\}|\>|\'|\"|`|;|:|,|\?|\!|\.|t)"
DATE_PATTERN = re.compile(date_regex, re.IGNORECASE)
"""Dates like `2022-12-01`, `01-12-2022`, `01-12-22`, `01/12/2022`, `01/12/22` where the whole date is captured in the first group, then each group of digits is captured in the order of appearance, in the next 3 groups"""

time_regex = r"(?<=^|\s|\[|\(|\{|\<|\'|\"|`|;|t)(\d{1,2}) ?(?:(h|H|:|am|pm)) ?(\d{2}|)?(?:\:(\d{2}))? ?(h|am|pm|z|utc)? ?((?:\+|\-)\d{1,4})?(?=$|\s|\]|\)|\}|\>|\'|\"|`|;|:|,|\?|\!|\.)"
TIME_PATTERN = re.compile(time_regex, re.IGNORECASE)
"""Identify more or less standard time patterns, like :

- 12h15
- 12:15
- 12:15:00
- 12am
- 12 am
- 12 h
- 12:15:00Z
- 12:15:00+01
- 12:15:00 UTC+1
- 11:27:45+0000

Returns:
  0 (str): 1- or 2-digits hour,
  1 (str): hour/minutes separator or half-day marker among `["h", ":", "am", "pm"]` (case-insensitive)
  2 (str): 2-digits minutes, if any, or `None`
  3 (str): 2-digits seconds, if any.
  4 (str): hour marker (`h` or `H`), half-day marker (case-insensitive `["am", "pm"]`), or time zone marker (case-sensitive `["Z", "UTC"]`)
  5 (str): 1-or 2-digits signed integer timezone shift (referred to UTC).

Examples:
    see https://regex101.com/r/QNtZAK/2

    see `src/tests/test-patterns.py`
"""

# IMAP-specific patterns

DOMAIN_PATTERN = re.compile(r"from ((?:[a-z0-9\-_]{0,61}\.)+[a-z]{2,})", re.IGNORECASE)
"""Matches patterns like `from (domain.ext)` from RFC-822 `Received` header in emails."""

UID_PATTERN = re.compile(r"UID ([0-9]+)")
"""Matches email integer UID from IMAP headers."""

FLAGS_PATTERN = re.compile(r"FLAGS \((.*?)\)")
"""Matches email flags from IMAP headers."""

# Filenames patterns
# Need to be tested BEFORE path pattern if both are used because a path is a more general case
# See https://stackoverflow.com/a/76113333/7087604

# All characters allowed in file names, aka not the following:
# Note: whitespace is technically allowed in file names, but it's a mess to include in regexs
filename = r"[^\#\%\<\>\&\*\{\}\\\/\?\$\!\|\=\"\'\@\n\r\t\b ]+"
non_filename = r"[\#\%\<\>\&\*\{\}\\\/\?\$\!\|\=\"\'\@\n\r\t\b ]"

path_regex = r"([A-Z]\:\\\\|\.\/|\~\/|\/)((?:%s(?:\\|\/)?)+)" % filename
PATH_PATTERN = re.compile(r"%s%s%s" % (regex_starter, path_regex, end_of_word))
"""File path pattern like `~/file`, `/home/file`, `./file` or `C:\\windows`"""

partial_path_regex = r"(%s(?:\\|\/)){2,}(%s)?" % (filename, filename)
PARTIAL_PATH_REGEX = re.compile(r"%s%s%s" % (regex_starter, partial_path_regex, end_of_word))
"""Partial, invalid path patterns missing the leading root, like `home/user/stuff`.
We start capturing after at least two folder separators (slash or backslash).

WARNING: this will collide with date detection, so run it after in the pipeline.
"""

filename = r"(?<=%s)%s" % (non_filename, filename)

IMAGE_PATTERN = re.compile(r"%s\.(bmp|jpg|jpeg|jpe|jp2|j2c|j2k|jpc|jpf|jpx|png|ico|svg|webp|heif|heic|tif|tiff|hdr|exr|ppm|pfm|nef|rw2|cr2|cr3|crw|dng|raf|arw|srf|sr2|iiq|3fr|dcr|ari|pef|x3f|erf|raw|rwz|orf)(?![\.\S]\S)" % filename, re.IGNORECASE)

CODE_PATTERN = re.compile(r"%s\.(php|m|py|sh|c|cxx|cpp|h|hxx|a|asm|awk|asp|class|java|yml|yaml|js|css|cl)(?![\.\S]\S)" % filename, re.IGNORECASE)

TEXT_PATTERN = re.compile(r"%s\.(txt|md|html|xml|xhtml|xmp|json|tex|rst|rtf)(?![\.\S]\S)" % filename, re.IGNORECASE)

DOCUMENT_PATTERN = re.compile(r"%s\.(xfc|kra|psd|ai|indd|ps|eps|pdf|xlsx|docx|pptx|doc|xls|ppt|odt|ods|odp|odg|odf|wpd)(?![\.\S]\S)" % filename, re.IGNORECASE)

ARCHIVE_PATTERN = re.compile(r"%s\.(zip|gzip|gz|tar|bz|iso|rar|img)(?![\.\S]\S)" % filename, re.IGNORECASE)

DATABASE_PATTERN = re.compile(r"%s\.(db|sql|sqlite)(?![\.\S]\S)" % filename, re.IGNORECASE)

EXECUTABLE_PATTERN = re.compile(r"%s\.(so|exe|dmg|appimage|bin|run|apk|jar|cmd|jar|workflow|action|autorun|osx|app|vb|dll|scr|bin|rpm|deb|distinfo)((?:\.[a-z0-9]+)+)?(?![a-zA-Z])" % filename, re.IGNORECASE)

shortcut_regex = r"(?:(?:fn|tab|ctrl|shift|alt|altgr|maj|command|cmd|option|menu|⌘))(?: ?\+ ?(?:⌘|tab|ctrl|shift|maj|alt|altgr|command|cmd|option|menu|click|clic|up|down|left|right|top|bottom|enter|return|del|suppr|home|end|pageup|pagedown|fn|home|end|insert|numlock|scroll|drag|f1|f2|f3|f4|f5|f6|f7|f8|f9|f10|f11|f12|[a-z]))+"
SHORTCUT_PATTERN = re.compile(r"%s%s%s" % (regex_starter, shortcut_regex, end_of_word), flags=re.IGNORECASE)

# For performance, on some technical log files where long sequences of numbers are found,
# we need to collapse them into smaller numbers sequences for the physical units patterns to work within reasonable timings.
# To exclude dates and times, we detect only starting at 5 digits.
NUMBER_SEQUENCE_PATTERN = re.compile(r"(?<=^|\s|\[|\'|\(|\")(%s?(?:\d+[\.\,\/\+\-\s]*){5,})(?=\s|$|\:|\;|,|\?|\!|\]|\)|\'|\")" % regex_algebra)

# For some reason, merging both patterns in the same triggers infinite loop, so split it…
price_regex = r"(%s)?(?:(?:(k)?(usd|eur|\€|\$|\£) ?((?:\d+[\.\,\+\- ]?)+))|(?:((?:\d+[\.\,\+\- ]?)+) ?(k)?(usd|eur|\€|\$|\£)))" % regex_algebra
PRICE_PATTERN = re.compile(r"%s%s%s" % (regex_starter, price_regex, end_of_word), re.IGNORECASE)

RESOLUTION_PATTERN = re.compile(r"\d+(?:×|x|X)\d+")
"""Pixel resolution like 10x20 or 10×20. Units are discarded."""

regex_number = r"(%s?(?:\d+[\.\,\/\+\- ]?)+)" % regex_algebra
NUMBER_PATTERN_FAST = re.compile(r"^%s$" % regex_number)
NUMBER_PATTERN = re.compile(r"%s%s%s" % (regex_starter, regex_number, regex_stopper))
"""Signed integers and decimals, fractions and numeric IDs with interal dashes and underscores.
Numbers with starting or trailing units are not considered. Lazy decimals (.1 and 1.) are considered.
"""

ORDINAL = re.compile(r"%s([0-9]+)(st|nd|rd|th|e|er|ère|ere|nde|ème|eme)%s" % (regex_starter, end_of_word), re.IGNORECASE)

regex_hash = r"([0-9a-f]){8,}"
HASH_PATTERN_FAST = re.compile(r"^%s$" % regex_hash, re.IGNORECASE)
HASH_PATTERN = re.compile(r"%s%s%s" % (regex_starter, regex_hash, end_of_word), re.IGNORECASE)
"""Cryptographic hexadecimal hashes and fingerprints, of a min length of 8 characters."""

MULTIPLE_LINES = re.compile(r"(?: ?[\t\r\n]{2,} ?)+")
"""Detect more than 2 newlines and tab, possibly mixed with spaces"""

MULTIPLE_NEWLINES = re.compile(r"(?: ?[\t\r\n]+ ?){2,}")
"""Detect broken sequences of newlines and spaces."""

MULTIPLE_SPACES = re.compile(r"( )+")

# Physical quantities (unit numbers)

exposure_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(ev|il)s?" % regex_algebra
EXPOSURE = re.compile(r"%s%s%s" % (regex_starter, exposure_regex, end_of_word), flags=re.IGNORECASE)
"""Exposure values in EV or IL"""

pixels_regex = r"(%s)? ?([0-9]+) ?(kilo|k|mega|m|giga|g|tera|t|peta|p)?(p|px|pixels|pix)s?" % regex_algebra
PIXELS = re.compile(r"%s%s%s" % (regex_starter, pixels_regex, end_of_word), flags=re.IGNORECASE)

sensibility_regex = r"(iso|asa) ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?)|([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(iso|asa)s?"
SENSIBILITY = re.compile(r"%s%s%s" % (regex_starter, sensibility_regex, end_of_word), flags=re.IGNORECASE)
"""Photographic sensibility in ISO or ASA"""

luminance_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(Cd\/m²|Cd\/m2|Cd\/m\^2|nit|nits)" % regex_algebra
LUMINANCE = re.compile(r"%s%s%s" % (regex_starter, luminance_regex, end_of_word), flags=re.IGNORECASE)
"""Luminance/radiance in nits or Cd/m²"""

diaphragm_regex = r"f\/([0-9]+\.?[0-9]?)"
# Don't look ahead for end of word here since lenses focal lengthes may be followed by reference letters
DIAPHRAGM = re.compile(r"%s%s" % (regex_starter, diaphragm_regex), flags=re.IGNORECASE)
"""Photographic diaph aperture values like f/2.8 or f/11"""

gain_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(dB|decibel|décibel)s?" % regex_algebra
GAIN = re.compile(r"%s%s%s" % (regex_starter, gain_regex, end_of_word), flags=re.IGNORECASE)
"""Gain, attenuation and PSNR in dB"""

filesize_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(kilo|k|mega|m|giga|g|tera|t|peta|p)?i?(b|o)s?" % regex_algebra
FILE_SIZE = re.compile(r"%s%s%s" % (regex_starter, filesize_regex, end_of_word), flags=re.IGNORECASE)
"""File and memory size in bit, byte, or octet and their multiples"""

distance_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(nano|n|micro|µ|milli|m|centi|c|deci|d|deca|hecto|kilo|k|mega|giga|g)?(m|meter|mètre|metre|in|inch|inche|ft|foot|feet|\'|\'\'|’|’’|\″)s?" % regex_algebra
DISTANCE = re.compile(r"%s%s%s" % (regex_starter, distance_regex, end_of_word), flags=re.IGNORECASE)
"""Distance in meter, inch, foot and their multiples"""

percent_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?\%%" % regex_algebra
PERCENT = re.compile(r"%s%s%s" % (regex_starter, percent_regex, end_of_word))
"""Number followed by %"""

weight_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(nano|n|micro|µ|milli|m|centi|c|deci|d|deca|hecto|kilo|k|mega|giga|g)?(g|gram|gramme|lb|pound)s?" % regex_algebra
WEIGHT = re.compile(r"%s%s%s" % (regex_starter, weight_regex, end_of_word), flags=re.IGNORECASE)
"""Weight (mass) in British and SI units and their multiples"""

angle_regex = r"(%s)? ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(deg|degree|degré|degre|°|rad|radian|sr|steradian)s?" % regex_algebra
ANGLE = re.compile(r"%s%s%s" % (regex_starter, angle_regex, end_of_word), flags=re.IGNORECASE)
"""Angles in radians, degrees and steradians"""

temperature_regex = r"(%s)?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(°C|degC|degree C|celsius|K|°F|kelvin)" % regex_algebra
TEMPERATURE = re.compile(r"%s%s%s" % (regex_starter, temperature_regex, end_of_word), flags=re.IGNORECASE)
"""Temperatures in °C, °F and K"""

frequency_regex = r"(%s)?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(nano|n|micro|µ|milli|m|centi|c|deci|d|deca|hecto|kilo|k|mega|giga|g)?(Hz|hertz)" % regex_algebra
FREQUENCY = re.compile(r"%s%s%s" % (regex_starter, frequency_regex, end_of_word), flags=re.IGNORECASE)
"""Frequencies in hertz and multiples"""


TEXT_DATES = re.compile(r"([0-9]{1,2})? (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|jan|fév|mar|avr|mai|jui|jui|aou|sep|oct|nov|déc|janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre|january|february|march|april|may|june|july|august|september|october|november|december)\.?( [0-9]{1,2})?( [0-9]{2,4})(?!\:)",
                        flags=re.IGNORECASE | re.MULTILINE)
"""Find textual dates formats:

- English dates like `01 Jan 20` or `01 Jan. 2020` but avoid capturing adjacent time like `12:08`.
- French dates like `01 Jan 20` or `01 Jan. 2020` but avoid capturing adjacent time like `12:08`.

Returns:
    0 (str): 2 digits (day number or year number, depending on language)
    1 (str): month (full-form or abbreviated)
    2 (str): 2 digits (day number or year number, depending on language)
    3 (str): 4 digits (full year)
"""

BASE_64 = re.compile(r"((?:[A-Za-z0-9+\/]{4}){64,}(?:[A-Za-z0-9+\/]{2}==|[A-Za-z0-9+\/]{3}=)?)")
"""Identifies base64 encoding"""

BB_CODE = re.compile(r"\[(img|quote)[a-zA-Z0-9 =\"]*?\].*?\[\/\1\]")
"""Identifies left-over BB code markup `[img]` and `[quote]`"""

MARKUP = re.compile(r"(?:\[|\{|\<)([^\n\r]+?)(?:\]|\}|\>)")
"""Identifies left-over HTML and Markdown markup, like `<...>`, `{...}`, `[...]`"""

USER = re.compile(r"([\w\-\+\.]+)?@([\w\-\+\.]+)|(user\-?\d+)")
"""Identifies user handles or emails"""

REPEATED_CHARACTERS = re.compile(r"(.)\1{9,}")
"""Identifies any character repeated more than 9 times"""

UNFINISHED_SENTENCES = re.compile(r"(?<![?!.;:])\n\n|\r\n")
"""Identifies sentences finishing with 2 newlines characters without having ending punctuations"""

MULTIPLE_DOTS = re.compile(r"\.{2,}")
"""Identifies dots repeated more than twice"""

MULTIPLE_DASHES = re.compile(r"-{1,}")
"""Identifies dashes repeated more than once"""

MULTIPLE_QUESTIONS = re.compile(r"\?{1,}")
"""Identifies question marks repeated more than once"""

ORDINAL_FR = re.compile(r"n° ?([0-9]+)")
"""French ordinal numbers (numéros n°)"""

FRANCAIS = re.compile(r"%s(j|t|s|l|d|qu|lorsqu|quelqu|jusqu|m|c|n)\'(?=[aeiouyéèàêâîôûïüäëöh][\w\s])" % regex_starter, flags=re.IGNORECASE)
"""French contractions of pronouns and determinants"""

DASHES = re.compile(r"(?<=\w)(-|_|=)(?=\w)", re.IGNORECASE)
"""Dashes in the middle of ASCII/Latin compounded words. Will not work if accented or Unicode characters are immediately surrounding the dash."""

ALTERNATIVES = re.compile(r"(?<=[a-z])(\/)(?=[a-z])", re.IGNORECASE)
"""Slash-separated word alternatives like `and/or` `mr/mrs`"""

PLURAL_S = re.compile(r"(?<=\w{4,})s?e{0,2}s%s" % end_of_word)
"""Identify plural form of nouns (French and English), adjectives (French) and third-person present verbs (English) and second-person verbs (French) in -s."""

FEMININE_E = re.compile(r"(?<=\w{4,})e{1,2}%s" % end_of_word)
"""Identify feminine form of adjectives (French) in -e."""

DOUBLE_CONSONANTS = re.compile(r"(?<=\w{2,})([^aeiouy])\1")
"""Identify double consonants in the middle of words."""

FEMININE_TRICE = re.compile(r"(?<=\w{4,})t(rice|eur|or)%s" % end_of_word)
"""Identify French feminine nouns in -trice."""

ADVERB_MENT = re.compile(r"(?<=\w{4,})e?ment%s" % end_of_word)
"""Identify French adverbs and English nouns ending en -ment"""

SUBSTANTIVE_TION = re.compile(r"(?<=\w{4,})(t|s)ion%s" % end_of_word)
"""Identify French and English substantives formed from verbs by adding -tion and -sion"""

SUBSTANTIVE_AT = re.compile(r"(?<=\w{4,})at%s" % end_of_word)
"""Identify French and English substantives formed from other nouns by adding -at"""

PARTICIPLE_ING = re.compile(r"(?<=\w{4,})ing%s" % end_of_word)
"""Identify English substantives and present participles formed from verbs by adding -ing"""

ADJECTIVE_ED = re.compile(r"(?<=\w{4,})ed%s" % end_of_word)
"""Identify English adjectives formed from verbs by adding -ed"""

ADJECTIVE_TIF = re.compile(r"(?<=\w{2,})ti(f|v)%s" % end_of_word)
"""Identify English and French adjectives formed from verbs by adding -tif or -tive"""

SUBSTANTIVE_Y = re.compile(r"(?<=\w{3,})y%s" % end_of_word)
"""Identify English substantives ending in -y"""

VERB_IZ = re.compile(r"(?<=\w{4,})(i|y)z%s" % end_of_word)
"""Identify American verbs ending in -iz that French and Brits write in -is"""

STUFF_ER = re.compile(r"(?<=\w{4,})er%s" % end_of_word)
"""Identify French 1st group verb (infinitive) and English substantives ending in -er"""

BRITISH_OUR = re.compile(r"(?<=\w{3,})our%s" % end_of_word)
"""Identify British spelling ending in -our (colour, behaviour)."""

SUBSTANTIVE_ITY = re.compile(r"(?<=\w{4,})it(y|e)%s" % end_of_word)
"""Identify substantives in -ity (English) and -ite (French)."""

SUBSTANTIVE_IST = re.compile(r"(?<=\w{3,})is(t|m)%s" % end_of_word)
"""Identify substantives in -ist and -ism."""

SUBSTANTIVE_IQU = re.compile(r"(?<=\w{3,})i(qu|c)%s" % end_of_word)
"""Identify French substantives in -iqu"""

SUBSTANTIVE_EUR = re.compile(r"(?<=\w{3,})eur%s" % end_of_word)
"""Identify French substantives -eur"""
