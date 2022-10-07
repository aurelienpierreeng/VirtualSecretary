#!/usr/bin/env python3

"""

Tag the emails attached to .ICS calendar invites with an RSVP label

© Aurélien Pierre - 2022

"""

import re

protocols = globals()
imap = protocols["imap"]


def filter(email) -> bool:
  # Find any attachment file with extension .ics
  calendar_pattern = re.compile(r"\S+\.ics", re.IGNORECASE)
  attachments = ", ".join(email.attachments)
  return re.match(calendar_pattern, attachments) != None

def action(email):
  email.tag("RSVP")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
