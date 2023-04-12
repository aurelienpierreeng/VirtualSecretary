"""
Contains global regular expression patterns re-used in the app.
You can use https://regex101.com/ to test these conveniently.

© 2023 - Aurélien Pierre
"""

import re

# Internet-specific patterns

IP_PATTERN = re.compile(r"from.*?((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4}))", re.IGNORECASE)
"""IPv4 and IPv6 patterns where the whole IP is captured in the first group."""

EMAIL_PATTERN = re.compile(r"<?([0-9a-zA-Z\-\_\+\.]+?@[0-9a-zA-Z\-\_\+]+(\.[0-9a-zA-Z\_\-]{2,})+)>?", re.IGNORECASE)
"""Emails patterns like `<me@mail.com>` or `me@mail.com` where the whole address is captured in the first group."""

URL_PATTERN = re.compile(r"(?>https?\:)?\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)", re.IGNORECASE)
"""URL patterns like `http(s)://domain.ext/page` or `//domain.ext/page` where `domain.ext` is captured as the first group and `/page` is the second group"""

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
