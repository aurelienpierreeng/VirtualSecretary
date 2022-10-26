#!/usr/bin/env python3

"""

Last chance of catching emails sent by automated systems and
not caught by previous filters. We store them
in a generic "Notifications" folder.

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  query = [
    "no-reply", "noreply", "no_reply", "autoreply",
    "notification",
    "ne-pas-repondre", "ne_pas_repondre", "nepasrepondre",
    "cpanel@", "wordpress@"
    ]
  return email.is_in(query, "From")


def action(email):
  email.move("INBOX.Services.Notifications")

imap.get_objects("INBOX")
imap.run_filters(filter, action, runs=-1)
