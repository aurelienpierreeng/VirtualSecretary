#!/usr/bin/env python3

"""

Try all unsubscribe links from bulk emails if any. This will work only for
mailing lists that don't use double opt-out paradigm (the ones asking you to
confirm you want to unsubscribe by hitting a button).

© Aurélien Pierre - 2022

"""

import requests

GLOBAL_VARS = globals()
secretary = GLOBAL_VARS["secretary"]
imap = secretary.protocols["imap"]


def filter(email) -> bool:
  result = False

  if "Precedence" in email.header:
    if email.header["Precedence"] == "bulk":
      result = "List-Unsubscribe" in email.header

  return result


def action(email):
  # Open the unsubscribe link, hoping it's not a double opt-out shit
  links = email.header["List-Unsubscribe"].split(",")

  for link in links:
    try:
      link = link.strip("<>")
      result = requests.get(link)
      imap.logfile.write("Tried to unsubscribe from %s with no guaranty\n" % link)
    except:
      pass


imap.get_objects(imap.junk, N)
imap.run_filters(filter, action)
