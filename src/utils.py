from datetime import datetime
import os
import re

filter_pattern = re.compile("^([0-9]{2})-(imap|caldav|carddav|mysql)-[a-zA-Z0-9\-\_]+.py$")
learn_pattern = re.compile("^(LEARN)-(imap|caldav|carddav|mysql)-[a-zA-Z0-9\-\_]+.py$")


def now():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def match_filter_name(file:str, mode:str):
  if mode == "process":
    match = filter_pattern.match(file)
  elif mode == "learn":
    match = learn_pattern.match(file)
  return match


def find_filters(path:str, filters:dict, mode:str) -> dict:
  # Find all the filter files in directory (aka filenames matching filter name pattern)
  # and append them to the dictionnary of filters based on their priority.
  # If 2 similar priorities are found, the first-defined one gets precedence, the other is discarded.
  local_filters = filters.copy()

  # Get the base priority for this path as the highest priority in the previous stage
  # This is used only for learning filters which don't have a user-set priority
  keys = list(local_filters.keys())
  priority = keys[-1] if len(keys) > 0 else 0

  if os.path.exists(path):
    for file in sorted(os.listdir(path)):
      match = match_filter_name(file, mode)
      if match:
        # Unpack the regex matching variables
        if mode == "process":
          # Get the 2 digits prefix as the priority
          priority = int(match.groups()[0])
        elif mode == "learn":
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
        local_filters[priority] = {"path": filter_path,
                                   "filter": file,
                                   "protocol": protocol }

  return local_filters


def lock_subfolder(lockfile):
  pid = str(os.getpid())
  abort = False
  if os.path.exists(lockfile):
    with open(lockfile, "r") as f:
      pid = f.read().strip()
    abort = True
  else:
    with open(lockfile, "w") as f:
      f.write(str(os.getpid()))

  return [abort, pid]


def unlock_subfolder(lockfile):
  if os.path.exists(lockfile):
    delete = False
    with open(lockfile, "r") as f:
      delete = (f.read() == str(os.getpid()))

    if delete:
      os.remove(lockfile)
