#!/usr/bin/env python3

"""

Detect the typical email headers of newsletters.

© Aurélien Pierre - 2022

"""

GLOBAL_VARS = globals()
mailserver = GLOBAL_VARS["mailserver"]
filtername = GLOBAL_VARS["filtername"]

def filter(email) -> bool:
  if "Precedence" in email.header:
    return (email.header["Precedence"] == "bulk")


def action(email) -> list:
  email.move("INBOX.Newsletters")

mailserver.get_mailbox_emails("INBOX")
mailserver.filters(filter, action, filtername)
