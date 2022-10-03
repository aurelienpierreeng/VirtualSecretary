#!/usr/bin/env python3

"""

Try all unsubscribe links from bulk emails if any. This will work only for
mailing lists that don't use double opt-out paradigm (the ones asking you to
confirm you want to unsubscribe by hitting a button).

© Aurélien Pierre - 2022

"""

import requests

GLOBAL_VARS = globals()
LOCAL_VARS = locals()

mailserver = GLOBAL_VARS["mailserver"]


def filter(email) -> bool:
  result = False

  if "Precedence" in email.header:
    if email.header["Precedence"] == "bulk":
      if "List-Unsubscribe" in email.header:
        result = True

  return result


def action(email):
  # Open the unsubscribe link, hoping it's not a double opt-out shit
  links = email.header["List-Unsubscribe"].split(",")

  for link in links:
    try:
      link = link.strip("<>")
      result = requests.get(link)
      mailserver.logfile.write("Tried to unsubscribe from %s with no guaranty\n" % link)
    except:
      pass


MAILBOX = "INBOX.spam"
N = 1

mailserver.get_mailbox_emails(MAILBOX, N)
mailserver.filters(filter, action)
