#!/usr/bin/env python3

"""

Detect the typical emails of newsletters.

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  return ("Precedence" in email.headers and email["Precedence"] == "bulk") or \
    ("List-Unsubscribe" in email.headers)

def action(email):
  email.move("INBOX.Newsletters")

imap.get_objects("INBOX")
imap.run_filters(filter, action, runs=-1)
