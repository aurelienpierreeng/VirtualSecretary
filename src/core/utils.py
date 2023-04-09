
"""
Logging and filter finding utilities.

© 2022-2023 - Aurélien Pierre
"""

from datetime import datetime
import os
import re
import errno

from typing import TypedDict

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

filter_pattern = re.compile("^([0-9]{2})-([a-z]+)-[a-zA-Z0-9\-\_]+.py$")
learn_pattern = re.compile("^(LEARN)-([a-z]+)-[a-zA-Z0-9\-\_]+.py$")


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
