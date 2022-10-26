#!/usr/bin/env python3

"""

Find invoices in emails and move them to a dedicated folders.py

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  return email.is_in("invoice", "Subject") or email.is_in("invoice", "Body")

def action(email) -> None:
  email.move("INBOX.Invoices")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
