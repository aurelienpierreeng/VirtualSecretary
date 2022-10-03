#!/usr/bin/env python3

"""

Tag the emails attached to .ICS calendar invites with an RSVP label

© Aurélien Pierre - 2022

"""

import re

GLOBAL_VARS = globals()
mailserver = GLOBAL_VARS["mailserver"]
filtername = GLOBAL_VARS["filtername"]

calendar_pattern = re.compile(r"\S+\.ics")

def filter(email) -> bool:
  # Find any attachment file with extension .ics
  attachments = ", ".join(email.attachments)
  return re.match(calendar_pattern, attachments) != None

def action(email):
  email.tag("RSVP")

mailserver.get_mailbox_emails("INBOX")
mailserver.filters(filter, action, filtername)
