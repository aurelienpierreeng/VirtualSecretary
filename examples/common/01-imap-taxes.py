#!/usr/bin/env python3

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  query = ["impots.gouv.fr", "finances.gouv.fr"]
  return email.is_in(query, "From")

def action(email):
  email.mark_as_important("add")
  email.move("INBOX.Money.Taxes")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
