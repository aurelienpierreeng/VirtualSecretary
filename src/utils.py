from datetime import datetime
import os
import re

filter_pattern = re.compile("^([0-9]{2})-(imap|caldav|carddav|mysql)-[a-zA-Z0-9\-\_]+.py$")

def now():
  return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def find_filters(path:str, filters:dict) -> dict:
  # Find all the filter files in directory (aka filenames matching filter name pattern)
  # and append them to the dictionnary of filters based on their priority.
  # If 2 similar priorities are found, the first-defined one gets precedence, the other is discarded.
  local_filters = filters.copy()
  if os.path.exists(path):
    for file in sorted(os.listdir(path)):
      match = filter_pattern.match(file)
      if match:
        # Unpack the regex matching variables
        priority = int(match.groups()[0])
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
