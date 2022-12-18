#!/usr/bin/env python3

import argparse
import os
import utils
import re
import time
import sys

from secretary import Secretary

ts = time.time()

# Unpack the program args
parser = argparse.ArgumentParser(description='Run the whole stack of filters defined in a config directory')
parser.add_argument('path', metavar='path', type=str,
                    help='path of the config directory')
parser.add_argument('mode', metavar='mode', type=str,
                    help='`process` to run only the processing filters (prefixed with 2 digits), \n`learn` to run the learning filters (prefixed with LEARN), \n`dryrun` to only test server connections')
parser.add_argument('-s', '--single', type=str, help="path of the only config subfolder that will be processed, typically to run filters on a single email account for debugging.")

args = parser.parse_args()
path = os.path.abspath(args.path)
mode = args.mode
single = args.single

# Get the common filters if any
filters = utils.find_filters(os.path.join(path, "common"), { }, mode)

# For each email directory in the config folder,
# process the filters
subfolders = sorted(os.listdir(path))
subfolders.remove("common")

# In single mode, process only the specified subfolder
if single:
  if single in subfolders:
    subfolders = [single]
  else:
    print("The subfolder %s has not been found in %s. Abort" % (single, path))
    sys.exit(1)

for dir in subfolders:
  # Manage concurrence by disabling editing over a captured folder
  lockfile = os.path.join(os.path.join(path, dir), ".lock")
  abort, pid = utils.lock_subfolder(lockfile)

  if abort:
    print("\nThe folder %s is already captured by another running instance with PID %s. We discard it here." % (dir, pid))
    continue
  else:
    print("\nProcessing folder %s with PID %s..." % (dir, pid))

  # Get the local filters if any
  local_filters = utils.find_filters(os.path.join(path, dir), filters, mode)

  # Load the global connectors manager and execute filters
  chantal = Secretary(os.path.join(path, dir))
  chantal.filters(local_filters)
  chantal.close_connectors()

  # Release the lock only if current instance captured it
  utils.unlock_subfolder(lockfile)


print("\nGlobal execution took %.2f s. Mind this if you use cron jobs." % (time.time() - ts))
sys.exit(0)
