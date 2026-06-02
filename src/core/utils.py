
"""
Logging and filter finding utilities.

© 2022-2023 - Aurélien Pierre
"""

from datetime import datetime, timezone, timedelta, UTC
from dateutil.relativedelta import relativedelta

import os
import io
import errno
import pickle
import tarfile
import time
import signal
import numba
import psutil
import unicodedata
import sqlite3
from collections.abc import Iterable

from typing import TypedDict
from dateutil import parser
from .patterns import MULTIPLE_SPACES, MULTIPLE_LINES, MULTIPLE_NEWLINES, INTERNAL_NEWLINE
from .types import web_page

import numpy as np
import regex as re

filter_entry = TypedDict("filter_entry", {"path": str, "filter": str, "protocol": str })
"""Dictionnary type representating a Virtual Secretary filter

Attributes:
  path (str): absolute path of the filter path.
  filter (str): name of the filter filter, aka name of the filter itself.
  protocol (str): server protocol, matching the name of one of the [protocols][].
"""

filter_bank = dict[int, filter_entry]
"""Dictionnary type of [core.utils.filter_entry][] elements associated with their priority in the bank.

Attributes:
  key (int): priority
  value (filter_entry): filter data
"""

from enum import Enum

class filter_mode(Enum):
  """Available filter types"""

  PROCESS = "process"
  """Filter applying write, edit or move actions"""

  LEARN = "learn"
  """Filter applying machine-learning or read-only actions"""

filter_pattern = re.compile(r"^([0-9]{2})-([a-z]+)-[a-zA-Z0-9\-\_]+.py$")
learn_pattern = re.compile(r"^(LEARN)-([a-z]+)-[a-zA-Z0-9\-\_]+.py$")


def now() -> str:
  """Return current time for log lines"""
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def match_filter_name(file: str, mode: filter_mode):
  """Check if the current filter file matches the requested mode.

  Arguments:
    file (str): filter file to test
    mode (filter_mode): filter type

  Returns:
    match (re.Match.group):
  """
  if mode == filter_mode.PROCESS:
    match = filter_pattern.match(file)
  elif mode == filter_mode.LEARN:
    match = learn_pattern.match(file)
  return match


def find_filters(path: str, filters: filter_bank, mode: filter_mode) -> filter_bank:
  """Find all the filter files in directory (aka filenames matching filter name pattern)
     and append them to the dictionnary of filters based on their priority.
     If 2 similar priorities are found, the first-defined one gets precedence, the other is discarded.

    Arguments:
      path (str): the folder where to find filter files
      filters (filter_bank): the dictionnary where we will append filters found here. This dictionnary will have the integer priority of filters (order of running) set as keys. If filters with the same priority are found in the current path, former filters are overriden.
      mode (filter_mode): the type of filter.
  """
  local_filters = filters.copy()

  # Dry run only test connection and login
  if mode == "dryrun":
    return local_filters

  # Get the base priority for this path as the highest priority in the previous stage
  # This is used only for learning filters which don't have a user-set priority
  keys = list(local_filters.keys())
  priority = keys[-1] if len(keys) > 0 else 0

  if os.path.exists(path):
    for file in sorted(os.listdir(path)):
      match = match_filter_name(file, mode)
      if match:
        # Unpack the regex matching variables
        if mode == filter_mode.PROCESS:
          # Get the 2 digits prefix as the priority
          priority = int(match.groups()[0])
        elif mode == filter_mode.LEARN:
          # No digits here
          priority += 1

        protocol = match.groups()[1]
        filter_path = os.path.join(path, file)

        # Throw a warning if we already have a filter at the same priority
        if priority in filters:
          old_filter = filters[priority]["filter"]
          print("Warning : filter %s at priority %i overrides %s already defined at the same priority." %
            (file, priority, old_filter))

        # Save filter, possibly overriding anything previously defined at same priority
        local_filters[priority] = filter_entry(path=filter_path, filter=file, protocol=protocol)

  return local_filters


def lock_subfolder(lockfile: str):
  """
  Write a `.lock` text file in the subfolder being currently processed, with the PID of the current Virtual Secretary instance.

  Override the lock if it contains a PID that doesn't exist anymore on the system (Linux-only).

  Arguments:
    lockfile (str): absolute path of the target lockfile

  Todo:
    Make it work for Windows PID too.
  """
  pid = str(os.getpid())
  abort = False
  if os.path.exists(lockfile):
    # Read the PID saved into the lockfile
    with open(lockfile, "r") as f:
        saved_pid = f.read().strip()

    # Check if the PID is still running on the system.
    # If not, the lockfile is most likely a leftover of a crashed run,
    # so we ignore it and carry on.
    try:
        os.kill(int(saved_pid), 0)
    except OSError as err:
        if err.errno == errno.ESRCH:
        # ESRCH == No such process
        # Override lockfile and carry on
            abort = False
            with open(lockfile, "w") as f:
                f.write(str(os.getpid()))
        elif err.errno == errno.EPERM:
            # EPERM clearly means there's a process to deny access to
            abort = True
            pid = saved_pid
        else:
            # According to "man 2 kill" possible error values are
            # (EINVAL, EPERM, ESRCH)
            raise
    else:
        # PID found running, we don't override it
        abort = True
        pid = saved_pid
  else:
      # No lock file, capture it
      with open(lockfile, "w") as f:
          f.write(str(os.getpid()))

  return [abort, pid]


def unlock_subfolder(lockfile: str):
  """
  Remove the `.lock` file in current subfolder.

  Arguments:
    lockfile (str): absolute path of the target lockfile
  """
  if os.path.exists(lockfile):
    delete = False
    with open(lockfile, "r") as f:
      delete = (f.read() == str(os.getpid()))

    if delete:
      os.remove(lockfile)


import binascii
from typing import Iterable, MutableSequence


# ENCODING
# --------
def _modified_base64(value: str) -> bytes:
    return binascii.b2a_base64(value.encode('utf-16be')).rstrip(b'\n=').replace(b'/', b',')


def _do_b64(_in: Iterable[str], r: MutableSequence[bytes]):
    if _in:
        r.append(b'&' + _modified_base64(''.join(_in)) + b'-')
    del _in[:]


def imap_encode(value: str) -> bytes:
    """
    Encode Python string into IMAP-compliant UTF-7 bytes, as described in the RFC 3501.

    There are variations, specific to IMAP4rev1, therefore the built-in python UTF-7 codec can't be used.
    The main difference is the shift character, used to switch from ASCII to base64 encoding context.
    This is "&" in that modified UTF-7 convention, since "+" is considered as mainly used in mailbox names.
    Full description at RFC 3501, section 5.1.3.

    Code from [imap_tools/imap_utf7.py](https://github.com/ikvk/imap_tools/blob/master/imap_tools/imap_utf7.py) by ikvk under Apache 2.0 license.

    Arguments:
      value (str): IMAP mailbox path as string

    Returns:
      path (bytes): IMAP-encoded path as UTF-7
    """
    res = []
    _in = []
    for char in value:
        ord_c = ord(char)
        if 0x20 <= ord_c <= 0x25 or 0x27 <= ord_c <= 0x7e:
            _do_b64(_in, res)
            res.append(char.encode())
        elif char == '&':
            _do_b64(_in, res)
            res.append(b'&-')
        else:
            _in.append(char)
    _do_b64(_in, res)
    return b''.join(res)


# DECODING
# --------
def _modified_unbase64(value: bytearray) -> str:
    return binascii.a2b_base64(value.replace(b',', b'/') + b'===').decode('utf-16be')


def imap_decode(value: bytes) -> str:
    """
    Decode IMAP-compliant UTF-7 byte into Python string, as described in the RFC 3501.

    There are variations, specific to IMAP4rev1, therefore the built-in python UTF-7 codec can't be used.
    The main difference is the shift character, used to switch from ASCII to base64 encoding context.
    This is "&" in that modified UTF-7 convention, since "+" is considered as mainly used in mailbox names.
    Full description at RFC 3501, section 5.1.3.

    Code from [imap_tools/imap_utf7.py](https://github.com/ikvk/imap_tools/blob/master/imap_tools/imap_utf7.py) by ikvk under Apache 2.0 license.

    Arguments:
      value (bytes): IMAP-encoded path as UTF-7 modified for IMAP

    Returns:
      path (str): IMAP path encoded as Python string
    """
    res = []
    decode_arr = bytearray()
    for char in value:
        if char == ord('&') and not decode_arr:
            decode_arr.append(ord('&'))
        elif char == ord('-') and decode_arr:
            if len(decode_arr) == 1:
                res.append('&')
            else:
                res.append(_modified_unbase64(decode_arr[1:]))
            decode_arr = bytearray()
        elif decode_arr:
            decode_arr.append(char)
        else:
            res.append(chr(char))
    if decode_arr:
        res.append(_modified_unbase64(decode_arr[1:]))
    return ''.join(res)


@numba.jit(nopython=True, nogil=True)
def _unicode_to_ascii(string: str) -> str:
    # For 1:many character replacment, we will have to use slow loops
    SUBSTITUTIONS = {
        # Apostrophes
        "’": "'",
        "`": "'",
        "‘": "'",
        "ʼ": "'",
        "'": "'",
        "´": "'",
        # Accents
        # The rationale here is some people use them improperly
        # (meaning they don't at all or use the wrong ones),
        # so level down for everyone for generalization.
        # This also makes for better generalization between French and English
        "é": "e",
        "è": "e",
        "ê": "e",
        "â": "a",
        "ô": "o",
        "á": "a", # should not exist in French
        "à": "a",
        "ù": "u",
        "î": "i",
        "û": "u",
        "ï": "i",
        "ë": "e",
        "ü": "u",
        "ö": "o",
        "ç": "c",
        "": " ",
        "": " ",
        # Spaces
        "\u2002": " ",  # En space
        "\u2003": " ",  # Em space
        "\u2004": " ",  # Three-Per-Em Space
        "\u2005": " ",  # Four-Per-Em Space
        "\u2006": " ",  # Six-Per-Em Space
        "\u2007": " ",  # Figure Space
        "\u2008": " ",  # Punctuation Space
        "\u2009": " ",  # thin space
        "\u200A": " ",  # hair space
        "\u200B": " ",  # Zero Width Space
        "\u200C": " ",  # Zero Width Non-Joiner
        "\u00A0": " ",  # Unbreakable space
        "\u202f": " ",  # Narrow No-Break Space
        # Hyphens and dashes
        "\u2010": "-",  # Hyphen
        "\u2011": "-",  # Non-Breaking Hyphen
        "\u2012": "-",  # Figure Dash
        "\u2013": "-",  # En Dash
        "\u2014": "-",  # Em Dash
        "\u2015": "-",  # Horizontal Bar
        "\uFF0D": "-",  # Fullwidth Hyphen-Minus
        "\uFE63": "-",  # Small Hyphen-Minus
        "↑": " ",
        "↵": " ",
        # Decorations and fucking emojis
        "☙": " ",
        "❧": " ",
        "🔗": " ",
        "•": " ",
        "©": " ",
        "®": " ",
        "|": " ",
        "¦": " ",
        "™": " ",
        "ᵉ": "e",
        "ƒ": "f",
        "·": " ",
        "✔": " ",
        "×": "x",
        # Ligatures
        "\u0132": "IJ", # Nederlands & Flanders
        "\u0133": "ij", # Nederlands & Flanders
        "\u0152": "OE", # French
        "\u0153": "oe", # French
        "\uA7F9": "oe", # French
        "\uFB00": "ff",
        "\uFB01": "fi",
        "\uFB02": "fl",
        "\uFB03": "ffi",
        "\uFB04": "ffl",
        "\uFB05": "st", # Medieval ligature
        "\uFB06": "st", # Medieval ligature
        # Punctuation
        "\u2026": "...",
        "« "    : "\"", # This needs spaces to have been decoded before
        " »"    : "\"", # This needs spaces to have been decoded before
        "«"     : "\"",
        "»"     : "\"",
        # Fractions
        "\u00BD": "1/2",
        "\u2153": "1/3",
        "\u2154": "2/3",
        "\u00BC": "1/4",
        "\u00BE": "3/4",
        "\u2155": "1/5",
        "\u2156": "2/5",
        "\u2157": "3/5",
        "\u2158": "4/5",
        "\u2159": "1/6",
        "\u215A": "5/6",
        "\u2150": "1/7",
        "\u215B": "1/8",
        "\u215C": "3/8",
        "\u215D": "5/8",
        "\u215E": "7/8",
        "\u2151": "1/9",
        "\u2152": "1/10",
        "\u215F": "1/",
        # Arrows
        "\u2190": "<-",
        "\u2192": "->",
        "\u2194": "<->",
        "\u21D0": "<-",
        "\u21D2": "->",
        "\u21D4": "<->",
        # Newlines + space
        "\n ": "\n",
        " \n": "\n",
        "\t ": "\t",
        " \t": "\t",
        # Quotes
        "``": "\"",
        "''": "\"",
        "“": "\"",
        "”": "\"",
        # Conditional hyphen ???
        "\u00AD": "",
    }

    # Perform educated Unicode character removal with closest ASCII symbol(s)
    for key, value in SUBSTITUTIONS.items():
        string = string.replace(key, value)

    return string


def typography_undo(string:str) -> str:
    """Break correct typographic Unicode entities into dummy computer characters (ASCII) to produce computer-standard vocabulary and help word tokenizers to properly detect word boundaries.

    This is useful when parsing:

        1. **properly composed** text, like the output of LaTeX or SmartyPants[^1]/WP Scholar[^2],
        2. text typed with Dvorak-like keyboard layouts (using proper Unicode entities where needed).

    For example, the proper `…` ellipsis entity (Unicode U+2026 symbol) will be converted into 3 regular dots `...`.

    [^1]: https://daringfireball.net/projects/smartypants/
    [^2]: https://eng.aurelienpierre.com/wp-scholar/
    """
    if string and isinstance(string, str):
        string = unicodedata.normalize("NFKC", string)
        string = _unicode_to_ascii(string)
        # Blindly remove all remaining non-ASCII characters:
        # emojis, bullets, non-latin characters including Chinese, Japanese, Greek, etc.
        return string.encode("ASCII", "ignore").decode().strip()
    else:
        return ""


def clean_whitespaces(string:str) -> str:
    """Collapse repeated spaces and newlines in text."""
    # Collapse multiple newlines and spaces
    string = MULTIPLE_LINES.sub("\n\n", string, concurrent=True, timeout=60)
    string = MULTIPLE_SPACES.sub(" ", string, concurrent=True, timeout=60)
    string = MULTIPLE_NEWLINES.sub("\n\n", string, concurrent=True, timeout=60)
    string = INTERNAL_NEWLINE.sub(" ", string, concurrent=True, timeout=60)
    return string.strip()



_SURROGATE_RE = re.compile(r'[\ud800-\udfff]')


def sanitize_unicode(text) -> str:
    """
    Normalize arbitrary string-like objects into safe Python UTF-8 text.
    """

    # Fast path
    if isinstance(text, str):
        out = text

    # Bytes-like objects
    elif isinstance(text, (bytes, bytearray, memoryview)):
        out = bytes(text).decode("utf-8", errors="replace")

    # Arbitrary foreign objects
    else:
        out = str(text)

    # Rare slow-path for invalid surrogates
    if _SURROGATE_RE.search(out, concurrent=True):
        out = out.encode("utf-8", "replace").decode("utf-8")

    return out


def parse_pdf_date(s: str) -> str:
    s = s.removeprefix("D:")

    # Fix PDF timezone apostrophes
    s = re.sub(r"([+-]\d{2})'(\d{2})'?", r"\1:\2", s)

    # Strip garbage after timezone (fix broken +00:0000'00')
    s = re.sub(r"(\+\d{2}:\d{2}).*$", r"\1", s)

    return s

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12
}

UTC = timezone.utc

TZINFOS = {
    "UTC": UTC,
    "PDT": timezone(timedelta(hours=-7)),
    "PST": timezone(timedelta(hours=-8)),
    "EDT": timezone(timedelta(hours=-4)),
    "EST": timezone(timedelta(hours=-5)),
    "CDT": timezone(timedelta(hours=-5)),
    "CST": timezone(timedelta(hours=-6)),
}

def parse_human_date(text: str) -> datetime | None:
    # Matches: "11 Oct 2017"
    m = re.search(
        r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})",
        text
    )
    if not m:
        return None

    day = int(m.group(1))
    mon_str = m.group(2)[:3].lower()
    year = int(m.group(3))

    month = MONTHS.get(mon_str)
    if not month:
        return None

    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def guess_date(string: str | datetime) -> datetime | None:
    """
    Best-effort datetime parsing.

    Always returns:
        - timezone-aware UTC datetime
        - or None
    """

    if isinstance(string, datetime):
        # Already datetime
        if string.tzinfo is None:
            return string.replace(tzinfo=UTC)

        return string.astimezone(UTC)

    if not isinstance(string, str):
        return None

    # Handle PDF date formats
    if string.startswith("D:"):
        string = parse_pdf_date(string)

    try:
        date = parser.parse(string, fuzzy=True, tzinfos=TZINFOS)

    except Exception:
        return parse_human_date(string)

    if date is None:
        return None

    # Normalize timezone handling
    if date.tzinfo is None:
        date = date.replace(tzinfo=UTC)
    else:
        date = date.astimezone(UTC)

    return date


## Default files and pathes

def get_data_folder(filename: str, scheme: str, ext: str) -> str:
    """Resolve the path of a training data saved under `filename`. These are stored in `../../data/`.

    Warning:
        This does not check the existence of the file and root folder.
    """
    current_path = os.path.abspath(__file__)
    install_path = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
    models_path = os.path.join(install_path, "data")
    return os.path.abspath(os.path.join(models_path, f"{filename}.{scheme}.{ext}"))


def save_data(data: list[web_page] | sqlite3.Connection, filename: str):
    """
    Save scraped data to a compressed archive.

    The destination folder and file extension are handled automatically.

    Args:
        data:
            Data to save.

            Supported types:

            - `list[web_page]`: saved as a `.pickle.tar.gz` archive using
              Python pickling.
            - `sqlite3.Connection`: saved as a `.sql.tar.gz` archive using
              an SQLite SQL dump.

        filename:
            Base filename to use. The output extension is added
            automatically depending on the type of `data`.
    """

    if isinstance(data, list):
        scheme = "pickle"
    elif isinstance(data, sqlite3.Connection):
        scheme = "sql"
    else:
        raise TypeError("Wrong input type for data, supports only Python list or SQLite3 database")

    with tarfile.open(get_data_folder(filename, scheme, 'tar.gz'), "w:gz") as tar:
        if scheme == "pickle":
            # Pickle data first to get the exact size
            content = io.BytesIO(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))

        elif scheme == "sql":
            # Dump the existing database into the temp one
            # to avoid copy/lock/naming issues, plus it's more efficiently gzipped
            content = io.BytesIO("\n".join(data.iterdump()).encode())

        else:
            return # already raises an error above anyway
        
        # Need to pass on the buffer size explicitly, tarfile sucks
        info = tarfile.TarInfo(f"{filename}.{scheme}")
        info.size = content.getbuffer().nbytes
        tar.addfile(info, fileobj=content)


def open_data(filename: str, scheme: str = "auto") -> list[web_page] | sqlite3.Connection:
    """
    Open data stored in a tar.gz archive. We probe for `sql` and `pickle` datasets, 
    in this order, and return the first we find. 

    Args:
        filename:
            Extension-less name of the dataset (no path).
        
        scheme: 
            - `sql` for data saved as SQL dumps, 
            - `pickle` for data saved as lists of `web_page`.
            - `auto` will probe both in this order and return the first one found.

    Returns:
        - list of `web_pages` for pickle archives,
        - sqlite3.Connection for database archives. The database lives in memory and 
        will not be saved, so the caller needs to copy/dump it, and close the connection.

    If the archive does not exist, returns an empty list.
    """

    if scheme == "auto":
        schemes = ["sql", "pickle"]
    else:
        schemes = [scheme]

    for scheme in schemes:
        path = get_data_folder(filename, scheme, 'tar.gz')

        if not os.path.exists(path):
            continue

        with tarfile.open(path, "r:*") as tar:
            members = {member.name for member in tar.getmembers()}
            member = f"{filename}.{scheme}"
            if member not in members:
                raise ValueError(f"Archive {path} does not contain {member}")

            content = tar.extractfile(member)

            if scheme == "pickle":
                return pickle.loads(content.read())

            elif scheme == "sql":
                db = sqlite3.connect(":memory:")
                db.executescript(content.read().decode())
                return db

    raise ValueError(f"No .pickle or .sql data archive could be found for {filename}")
    

def get_data_mtime(filename: str, scheme: str) -> datetime | None:
    """
    Return the modification date of the tar.gz archive.

    Returns:
        datetime of the archive modification time, or None if it does not exist.
    """
    path = get_data_folder(filename, scheme, 'tar.gz')

    if not os.path.exists(path):
        return None

    return datetime.fromtimestamp(os.path.getmtime(path))


def get_models_folder(filename: str) -> str:
    """Resolve the path of a machine-learning model saved under `filename`. These are stored in `../../models/`.

    Warning:
        This does not check the existence of the file and root folder.
    """
    current_path = os.path.abspath(__file__)
    install_path = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(current_path)))
    models_path = os.path.join(install_path, "models")
    return os.path.abspath(os.path.join(models_path, filename))


def get_stopwords_file(filename: str) -> dict:
    """Get a dictionnary file containing lines of "word: frequency" stored in `../../models/`.
    By default, [core.nlp.Word2Vec][] stores a such file when the word embedding is learned.
    Manually-validated files can be used for search engine purposes, since stopwords add noise to the searches.
    """
    path = get_models_folder(filename)
    with open(path, "r") as f:
      d = dict(x.strip().split(": ", 1) for x in f)
    return d


def timeit(runs: int = 1):
    """Provide a `@timeit` decorator to profile the wall performance of a function.

    Args:
      runs: 
        how many times the function should be re-executed. Runtimes will give average and standard deviation.
    """
    def decorate(func):
        def wrapper(*args, **kwargs):
          results = []

          for _ in range(runs):
            start = time.time()
            out = func(*args, **kwargs)
            end = time.time()
            results.append(end - start)

          results = np.array(results)
          print("function %s took (%f ± %f) s, average of %i runs" % (func.__qualname__, np.mean(results), np.std(results), results.size))
          return out
        return wrapper
    return decorate

def exit_after(s: int):
    """Define a decorator `exit_after(n)` that stops a function after `n` seconds.

    Mostly intended for text parsing functions that get fed unchecked text inputs from the web.
    In that case, some really bad XML or super-long log files can make the parsing loop hang forever.
    This decorator will skip them without breaking the loop.

    Args:
        s: number of seconds

    """
    def outer(fn):
        def inner(*args, **kwargs):
            def out_of_time(signum, frame):
                raise TimeoutError

            result = None

            # Raise a TimeoutError after s seconds
            signal.signal(signal.SIGALRM, out_of_time)
            signal.alarm(s)

            try:
                result = fn(*args, **kwargs)
            except TimeoutError:
                print(f"function {fn.__name__} timed out after {s} seconds with inputs {args} {kwargs}")
            finally:
                # Reset the timer
                signal.alarm(0)

            return result
        return inner
    return outer


def get_available_ram():
    return psutil.virtual_memory().available


def get_script_ram():
    return psutil.Process(os.getpid()).memory_info().vms


def get_past_n_months(n: int) -> datetime:
    """Get the date of now minus n months"""
    return datetime.now(UTC) - relativedelta(months=n)


def get_past_n_weeks(n: int) -> datetime:
    """Get the date of now minus n weeks"""
    return datetime.now(UTC) - relativedelta(weeks=n)


def get_past_n_days(n: int) -> datetime:
    """Get the date of now minus n days"""
    return datetime.now(UTC) - relativedelta(days=n)