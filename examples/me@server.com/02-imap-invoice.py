#!/usr/bin/env python3

"""

Find invoices in emails and move them to a dedicated folders.py

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  return "invoice" in email.get_body("plain").lower() + email["Subject"].lower()

def action(email):
  email.move("INBOX.Invoices")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
