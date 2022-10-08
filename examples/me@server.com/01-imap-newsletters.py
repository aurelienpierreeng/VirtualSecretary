#!/usr/bin/env python3

"""

Detect the typical emails of newsletters.

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  print(email)
  result = False

  if "Precedence" in email and "List-Unsubscribe" in email:
    result = (email["Precedence"] == "bulk")

  return result

def action(email) -> list:
  email.move("INBOX.Newsletters")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
