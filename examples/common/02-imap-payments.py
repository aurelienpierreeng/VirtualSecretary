#!/usr/bin/env python3

"""

Catch paiements notifications from Stripe and PayPal
using their usual email addresses

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  sender = email["From"].lower()
  # Paypal switched from member@paypal.com to service@paypal.xxx mid-2017
  # Stripe switched from support@stripe.com to notification@stripe.com in April 2022
  query = ["service@paypal.", "member@paypal.com", "notifications@stripe.com", "support@stripe.com"]
  return email.is_in(query, "From")



def action(email):
  email.move("INBOX.Money.Paiements")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
