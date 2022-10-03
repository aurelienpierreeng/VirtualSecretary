#!/usr/bin/env python3

import argparse
import configparser
import os
from xml.etree.ElementPath import find
import utils
import re

from mailserver import MailServer

# Unpack the program args
parser = argparse.ArgumentParser(description='')
parser.add_argument('path', metavar='url', type=str,
                    help='path of the config directory')
args = parser.parse_args()
PATH = os.path.abspath(args.path)

# Find filter scripts
filters = { }

# Get the common filters if any
filters = utils.find_filters(os.path.join(PATH, "common"), filters)

# For each email directory in the config folder,
# process the filters
for dir in sorted(os.listdir(PATH)):
  if dir != "common":
    # Unpack the servers credentials
    config_file = configparser.ConfigParser()
    config_file.read(os.path.join(PATH, dir + "/settings.ini"))

    # Start the logile
    logfile = open(os.path.join(PATH, dir + "/sync.log"), 'a')

    # Connect to the IMAP account through SSL
    imap = MailServer(config_file["imap"]["server"], config_file["imap"]["user"],
                      config_file["imap"]["password"], int(config_file["imap"]["entries"]),
                      logfile)

    # Get the local filters if any
    local_filters = utils.find_filters(os.path.join(PATH, dir), filters)

    for key in sorted(local_filters.keys()):
      filter = local_filters[key]["filter"]
      filter_path = local_filters[key]["path"]

      with open(filter_path) as f:
        print("\nExecuting filter %s :" % filter)
        logfile.write("%s : Executing filter %s\n" % (utils.now(), filter))
        code = compile(f.read(), filter_path, 'exec')
        exec(code, {"mailserver": imap, "filtername": filter_path})
        imap.close()

    imap.logout()
    logfile.close()
