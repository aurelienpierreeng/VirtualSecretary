 #!/usr/bin/env python3

protocols = globals()
imap = protocols["imap"]


def filter(email) -> bool:
  query = ["confirmation", "invoice", "billing", "facture", "achat", "facture", "commande"]
  return email.is_in(query, "Subject")

def action(email):
  email.move("INBOX.Money.Invoices")

imap.get_objects("INBOX")
imap.run_filters(filter, None, runs=-1)
