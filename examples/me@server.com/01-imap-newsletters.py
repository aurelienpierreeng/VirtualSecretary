#!/usr/bin/env python3

"""

Detect the typical email headers of newsletters.

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  result = False

  if "Precedence" in email.header and "List-Unsubscribe" in email.header:
    result = (email.header["Precedence"] == "bulk")

  return result

def action(email) -> list:
  email.move("INBOX.Newsletters")

imap.get_mailbox_emails("INBOX")
imap.run_filters(filter, action)
