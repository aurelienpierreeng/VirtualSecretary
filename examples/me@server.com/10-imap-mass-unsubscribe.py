#!/usr/bin/env python3

"""

Try all unsubscribe links from bulk emails if any. This will work only for
mailing lists that don't use double opt-out paradigm (the ones asking you to
confirm you want to unsubscribe by hitting a button).

© Aurélien Pierre - 2022

"""

import requests

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  return "List-Unsubscribe" in email


def action(email):
  # Open the unsubscribe link, hoping it's not a double opt-out shit
  links = email["List-Unsubscribe"].split(",")

  for link in links:
    try:
      link = link.strip("<>\r\n")
      result = requests.get(link)
      email.mailserver.logfile.write("Tried to unsubscribe from %s with no guaranty\n" % link)
    except:
      print(result)


imap.get_objects(imap.junk)
imap.run_filters(filter, action)
