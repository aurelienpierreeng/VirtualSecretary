"""
Contains global regular expression patterns re-used in the app.
You can use https://regex101.com/ to test these conveniently.

© 2023 - Aurélien Pierre
"""

import regex as re

# Internet-specific patterns

IP_PATTERN = re.compile(r"(?=^|\s|\[|\(|\{|\<)((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4}))(?=$|\s|\]|\)|\}|\>)", re.IGNORECASE)
"""IPv4 and IPv6 patterns where the whole IP is captured in the first group."""

EMAIL_PATTERN = re.compile(r"<?([0-9a-zA-Z\-\_\+\.]+?@[0-9a-zA-Z\-\_\+]+(\.[0-9a-zA-Z\_\-]{2,})+)>?", re.IGNORECASE)
"""Emails patterns like `<me@mail.com>` or `me@mail.com` where the whole address is captured in the first group."""

URL_PATTERN = re.compile(r"(?:https?\:)?\/\/([^:\/\?\#\s\\]+)(?:\:[0-9]*)?([\/]{0,1}[^?#\s\"\,\;\:>]*)(\?[a-z]+[\=\,\+\&\%\-\.a-zA-Z0-9]*)?(?=$|\s|\]|\)|\}|\>)", re.IGNORECASE)
"""URL patterns like `http(s)://domain.ext/page?q=x&r=0` or `//domain.ext/page`.

- `domain.ext` is captured as the first group,
- `/page` is the second group,
- page query parameters `?s=x&r=0` are captured in the 3rd.

URLs are captured if they are:

 - alone on their own line,
 - enclosed in {}, [], ()
 - enclosed in whitespaces.
 """

# Date/time

DATE_PATTERN = re.compile(r"((\d{2,4})(?:-|\/)(\d{2})(?:-|\/)(\d{2,4}))")
"""Dates like `2022-12-01`, `01-12-2022`, `01-12-22`, `01/12/2022`, `01/12/22` where the whole date is captured in the first group, then each group of digits is captured in the order of appearance, in the next 3 groups"""

TIME_PATTERN = re.compile(r"(\d{1,2}) ?(?:(h|H|:|am|pm|AM|PM)) ?(\d{2}|)?(?:\:(\d{2}))? ?(h|H|am|pm|AM|PM|Z|UTC)? ?((?:\+|\-)\d{1,2})?")
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

Returns:
  group[0]: 1- or 2-digits hour,
  group[1]: hour/minutes separator or half-day marker among `["h", ":", "am", "pm"]` (case-insensitive)
  group[2]: 2-digits minutes, if any, or `None`
  group[3]: 2-digits seconds, if any.
  group[4]: hour marker (`h` or `H`), half-day marker (case-insensitive `["am", "pm"]`), or time zone marker (case-sensitive `["Z", "UTC"]`)
  group[5]: 1-or 2-digits signed integer timezone shift (referred to UTC).

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
filename = r"[^\#\%\<\>\&\*\{\}\\\/\?\$\!\|\=\"\'\@\n\r\t\b ]+?"
non_filename = r"[\#\%\<\>\&\*\{\}\\\/\?\$\!\|\=\"\'\@\n\r\t\b ]"

PATH_PATTERN = re.compile(r"([A-Z]:|\.)?(\\\\|\/)(%s(\\\\|\/)?)+" % filename)
"""File path pattern like `~/file`, `/home/file`, `./file` or `C:\\windows`"""

filename = r"(?<=%s)%s" % (non_filename, filename)

IMAGE_PATTERN = re.compile(r"%s\.(bmp|jpg|jpeg|jpe|jp2|j2c|j2k|jpc|jpf|jpx|png|ico|svg|webp|heif|heic|tif|tiff|hdr|exr|ppm|pfm|nef|rw2|cr2|cr3|crw|dng|raf|arw|srf|sr2|iiq|3fr|dcr|ari|pef|x3f|erf|raw|rwz|orf)(?![\.\S]\S)" % filename, re.IGNORECASE)

CODE_PATTERN = re.compile(r"%s\.(php|m|py|sh|c|cxx|cpp|h|hxx|a|asm|awk|asp|class|java|yml|yaml|js|css|cl)(?![\.\S]\S)" % filename, re.IGNORECASE)

TEXT_PATTERN = re.compile(r"%s\.(txt|md|html|xml|xhtml|xmp|json|tex|rst|rtf)(?![\.\S]\S)" % filename, re.IGNORECASE)

DOCUMENT_PATTERN = re.compile(r"%s\.(xfc|kra|psd|ai|indd|ps|eps|pdf|xlsx|docx|pptx|doc|xls|ppt|odt|ods|odp|odg|odf|wpd)(?![\.\S]\S)" % filename, re.IGNORECASE)

ARCHIVE_PATTERN = re.compile(r"%s\.(zip|gzip|gz|tar|bz|iso|rar|img)(?![\.\S]\S)" % filename, re.IGNORECASE)

DATABASE_PATTERN = re.compile(r"%s\.(db|sql|sqlite)(?![\.\S]\S)" % filename, re.IGNORECASE)

EXECUTABLE_PATTERN = re.compile(r"%s\.(so|exe|dmg|appimage|bin|run|apk|jar|cmd|jar|workflow|action|autorun|osx|app|vb|dll|scr|bin|rpm|deb|distinfo)((?:\.[a-z0-9]+)+)?(?![a-zA-Z])" % filename, re.IGNORECASE)

SHORTCUT_PATTERN = re.compile(r"(?<=^|[\s\[\(\':])(?:(?:fn|tab|ctrl|shift|alt|altgr|maj|command|cmd|option|menu|⌘))(?: ?\+ ?(?:⌘|tab|ctrl|shift|maj|alt|altgr|command|cmd|option|menu|click|clic|up|down|left|right|top|bottom|enter|return|del|suppr|home|end|pageup|pagedown|fn|home|end|insert|numlock|scroll|drag|f1|f2|f3|f4|f5|f6|f7|f8|f9|f10|f11|f12|[a-z]))+(?=$|[\s.,?!\-:;\]\)])", flags=re.IGNORECASE)

# For some reason, merging both patterns in the same triggers infinite loop, so split it…
PRICE_US_PATTERN = re.compile(r"(?<=^|[\s\[\(\'])([+-=≠±])?((usd|eur|USD|EUR|\€|\$|\£) ?\d+(?:[.,\-]\d+)*)(k|K)?(?=$|[\s.,?!\-:;\]\)])")
PRICE_EU_PATTERN = re.compile(r"(?<=^|[\s\[\(\'])([+-=≠±])?(\d+(?:[.,\-]\d+)* ?(k|K)?(usd|eur|USD|EUR|\€|\$|\£))(?=$|[\s.,?!\-:;\]\)])")

RESOLUTION_PATTERN = re.compile(r"\d+(×|x|X)\d+")
"""Pixel resolution like 10x20 or 10×20. Units are discarded."""

NUMBER_PATTERN = re.compile(r"(?<=^|[\s\[\(\'])([\.\,\-\_\/\+\-±]?(?:\d+[\.\,\-\_\/\+\-]?)+)(?=$|[\s.,?!\-:;\]\)])")
"""Signed integers and decimals, fractions and numeric IDs with interal dashes and underscores.
Numbers with starting or trailing units are not considered. Lazy decimals (.1 and 1.) are considered.
"""

ORDINAL = re.compile(r"(?<=^|[\s\[\(\'])([0-9]+)(st|nd|rd|th|e|er|ère|ere|nde|ème|eme)(?=$|[\s.,?!\-:;\]\)])", re.IGNORECASE)

HASH_PATTERN = re.compile(r"([0-9a-f]){8,}", re.IGNORECASE)
"""Cryptographic hexadecimal hashes and fingerprints, of a min length of 8 characters."""

MULTIPLE_LINES = re.compile(r"(?:(?: ?[\t\r\n] ?){2,})+")
"""Detect more than 2 newlines and tab, possibly mixed with spaces"""

MULTIPLE_SPACES = re.compile(r"( )+")

# Physical quantities (unit numbers)

EXPOSURE = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(ev|il)s?(?=$|[\s.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Exposure values in EV or IL"""

PIXELS = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+) ?(kilo|k|mega|m|giga|g|tera|t|peta|p)?(px|pixels)s?(?=$|[\s.,?!\-:;\]\)])", flags=re.IGNORECASE)

SENSIBILITY = re.compile(r"(?<=^|[\s\[\(\'])(ISO|ASA) ?([0-9]+(?:[.,\-+\/ ][0-9]*)*?)|([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(ISO|ASA)s?(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Photographic sensibility in ISO or ASA"""

LUMINANCE = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(Cd\/m²|Cd\/m2|Cd\/m\^2|nit|nits)(?=$|[\s.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Luminance/radiance in nits or Cd/m²"""

DIAPHRAGM = re.compile(r"(?<=^|[\s\[\(\'])f\/?([0-9]+\.?[0-9]?)(?=$|[\s.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Photographic diaph aperture values like f/2.8 or f1.4"""

GAIN = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(dB|decibel|décibels)s?(?=$|[\s.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Gain, attenuation and PSNR in dB"""

FILE_SIZE = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(kilo|k|mega|m|giga|g|tera|t|peta|p)?i?(b|o)s?(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""File and memory size in bit, byte, or octet and their multiples"""

DISTANCE = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(nano|n|micro|µ|milli|m|centi|c|deci|d|deca|hecto|kilo|k|mega|giga|g)?(m|meter|mètre|metre|in|inch|inche|ft|foot|feet|\'|\'\'|’|’’|\″)s?(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Distance in meter, inch, foot and their multiples"""

PERCENT = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?\%(?=\s|$|[.,?!\-:;\]\)])")
"""Number followed by %"""

WEIGHT = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(nano|n|micro|µ|milli|m|centi|c|deci|d|deca|hecto|kilo|k|mega|giga|g)?(g|gram|gramme|lb|pound)s?(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Weight (mass) in British and SI units and their multiples"""

ANGLE = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(deg|degree|degré|degre|°|rad|radian|sr|steradian)s?(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Angles in radians, degrees and steradians"""

TEMPERATURE = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(°C|degC|degree C|celsius|K|°F)(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Temperatures in °C, °F and K"""

FREQUENCY = re.compile(r"(?<=^|[\s\[\(\'])([+\-=≠±])?([0-9]+(?:[.,\-+\/ ][0-9]*)*?) ?(nano|n|micro|µ|milli|m|centi|c|deci|d|deca|hecto|kilo|k|mega|giga|g)?(Hz|hertz)(?=\s|$|[.,?!\-:;\]\)])", flags=re.IGNORECASE)
"""Frequencies in hertz and multiples"""
