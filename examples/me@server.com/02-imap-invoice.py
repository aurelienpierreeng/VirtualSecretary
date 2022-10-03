#!/usr/bin/env python3

"""

Find invoices in emails and move them to a dedicated folders.py

© Aurélien Pierre - 2022

"""

GLOBAL_VARS = globals()
mailserver = GLOBAL_VARS["mailserver"]
filtername = GLOBAL_VARS["filtername"]

def filter(email) -> bool:
  return "invoice" in [email.body["text/plain"].lower(), email.header["Subject"].lower()]

def action(email):
  email.move("INBOX.Invoices")

mailserver.get_mailbox_emails("INBOX")
mailserver.filters(filter, action, filtername)
