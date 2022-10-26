#!/usr/bin/env python3

"""

Catch notifications for changes in conditions of usage for web services.

© Aurélien Pierre - 2022

"""


protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  return email.is_in("conditions", "Subject")

def action(email):
  email.move("INBOX.Services.Conditions")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
