#!/usr/bin/env python3

import argparse
import os
import re
import time
import sys

from core.secretary import Secretary
from core import utils

ts = time.time()

# Unpack the program args
parser = argparse.ArgumentParser(description='Run the whole stack of filters defined in a config directory')
parser.add_argument('path', metavar='path', type=str,
                    help='path of the config directory')
parser.add_argument('mode', metavar='mode', type=str,
                    help='`process` to run only the processing filters (prefixed with 2 digits), \n`learn` to run the learning filters (prefixed with LEARN), \n`dryrun` to only test server connections')
parser.add_argument('-s', '--single',
                    help="path of the only config subfolder that will be processed, typically to run filters on a single email account for debugging.")
parser.add_argument('--server',
                    action='store_true',
                    help="enable server mode, more gentle on resources (slows down the computations)")
parser.add_argument('-f', "--force",
                    action='store_true',
                    help="force reprocessing items already in logs (already processed)")
parser.add_argument('-n', "--number", type=int,
                    help="override the number of items to process defined in config files and use this one")


args = parser.parse_args()
path = os.path.abspath(args.path)
mode = utils.filter_mode[args.mode.upper()]
single = args.single
server = args.server
number = args.number
force = args.force

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
  chantal = Secretary(os.path.join(path, dir), server, number, force)
  chantal.filters(local_filters)
  chantal.close_connectors()

  # Release the lock only if current instance captured it
  utils.unlock_subfolder(lockfile)


print("\nGlobal execution took %.2f s. Mind this if you use cron jobs." % (time.time() - ts))
sys.exit(0)
