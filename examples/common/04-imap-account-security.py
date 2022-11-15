#!/usr/bin/env python3

"""

Catch alerts regarding accounts security

© Aurélien Pierre - 2022

"""


protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  query_from = [
    "account"
    ]
  query_subject = [
    "account", "password", "security", "verify", "login",
    "compte", "mot de passe", "identifiant", "securité", "vérifier", "code"
  ]
  return email.is_in(query_from, "From") or email.is_in(query_subject, "Subject")


def action(email):
  email.move('INBOX.Services.Accounts')

imap.get_objects("INBOX")
imap.run_filters(filter, action, runs=-1)
