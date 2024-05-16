
"""
Logging and filter finding utilities.

© 2022-2023 - Aurélien Pierre
"""

from datetime import datetime, timezone, timedelta
import os
import sys
import io
import errno
import pickle
import tarfile
import time
import signal
import numba
import psutil
from collections.abc import Iterable

from typing import TypedDict
from dateutil import parser
from .patterns import MULTIPLE_SPACES, MULTIPLE_LINES, MULTIPLE_NEWLINES, INTERNAL_NEWLINE

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
"""Dictionnary type of [utils.filter_entry][] elements associated with their priority in the bank.

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


@numba.jit(nopython=True, nogil=True, cache=True)
def _unicode_to_ascii(string: str) -> str:
    # For 1:many character replacment, we will have to use slow loops
    SUBSTITUTIONS = {
        # Apostrophes
        "’": "'",
        "`": "'",
        "“": "\"",
        "”": "\"",
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
        string = _unicode_to_ascii(string)
        # Blindly remove all remaining non-ASCII characters:
        # emojis, bullets, non-latin characters including Chinese, Japanese, Greek, etc.
        return string.encode("ASCII", "ignore").decode().strip()
    else:
        return ""


def clean_whitespaces(string:str) -> str:
    # Collapse multiple newlines and spaces
    string = MULTIPLE_LINES.sub("\n\n", string, concurrent=True, timeout=60)
    string = MULTIPLE_SPACES.sub(" ", string, concurrent=True, timeout=60)
    string = MULTIPLE_NEWLINES.sub("\n\n", string, concurrent=True, timeout=60)
    string = INTERNAL_NEWLINE.sub(" ", string, concurrent=True, timeout=60)
    return string.strip()


def guess_date(string: str | datetime) -> datetime:
    """Best effort to guess a date from a string using typical date/time formats"""
    # If no timezone/offset is provided, default to UTC
    tz = timezone(timedelta(0))

    if isinstance(string, str):
        try:
            date = parser.parse(string, default=datetime.fromtimestamp(0, tz=tz), fuzzy=True)
        except Exception as e:
            print("Date parser got an error:", e)
            date = datetime.fromtimestamp(0, tz=tz)
    elif isinstance(string, datetime):
        date = string.replace(tzinfo=tz)
    else:
        date = datetime.fromtimestamp(0, tz=tz)

    return date

## Default files and pathes

def get_data_folder(filename: str) -> str:
    """Resolve the path of a training data saved under `filename`. These are stored in `../../data/`.
    The `.pickle` extension is added automatically to the filename.

    Warning:
        This does not check the existence of the file and root folder.
    """
    current_path = os.path.abspath(__file__)
    install_path = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(current_path)))
    models_path = os.path.join(install_path, "data")
    return os.path.abspath(os.path.join(models_path, filename + ".pickle.tar.gz"))


def save_data(data: list, filename: str):
    """Save scraped data to a pickle file inside a tar.gz archive in data folder. Folder and file extension are handled automatically."""
    with tarfile.open(get_data_folder(filename), "w:gz") as tar:
        content = io.BytesIO(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
        info = tarfile.TarInfo(filename + ".pickle")

        # Need to pass on the buffer size explicitly, tarfile sucks
        info.size = content.getbuffer().nbytes
        tar.addfile(info, fileobj=content)


def open_data(filename: str) -> list:
    """Open scraped data from a pickle file inside a tar.gz archive stored in data folder. Folder and file extension are handled automatically.
    An empty list is returned is the file does not exist.
    """
    path = get_data_folder(filename)
    if os.path.exists(path):
      with tarfile.open(path, "r:*") as tar:
        content = tar.extractfile(tar.getmember(filename + ".pickle"))
        dataset = pickle.loads(content.read())
    else:
       dataset = []
    return dataset


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
    By default, [core.nlp.Word2Vec.__init__][] stores a such file when the word embedding is learned.
    Manually-validated files can be used for search engine purposes, since stopwords add noise to the searches.
    """
    path = get_models_folder(filename)
    with open(path, "r") as f:
      d = dict(x.strip().split(": ", 1) for x in f)
    return d


def timeit(runs: int = 1):
    """Provide a `@timeit` decorator to profile the wall performance of a function.

    Args:
      - runs: how many times the function should be re-executed. Runtimes will give average and standard deviation.
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
          print("function %s took (%f ± %f) s, average of %i runs" % (func.__name__, np.mean(results), np.std(results), results.size))
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

    Returns:
        the output of the function or None if it timed out.
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
