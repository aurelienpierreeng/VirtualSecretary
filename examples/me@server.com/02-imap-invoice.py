#!/usr/bin/env python3

"""

Find invoices in emails and move them to a dedicated folders.py

© Aurélien Pierre - 2022

"""

GLOBAL_VARS = globals()
secretary = GLOBAL_VARS["secretary"]
imap = secretary.protocols["imap"]

def filter(email) -> bool:
  return "invoice" in [email.body["text/plain"].lower(), email.header["Subject"].lower()]

def action(email):
  email.move("INBOX.Invoices")

imap.get_mailbox_emails("INBOX")
imap.run_filters(filter, action)
