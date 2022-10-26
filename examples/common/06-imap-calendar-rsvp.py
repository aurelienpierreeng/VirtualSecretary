#!/usr/bin/env python3

"""

Tag the emails attached to .ICS calendar invites with an RSVP label

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]


def filter(email) -> bool:
  # Find any attachment file with extension .ics
  for attachment in email.attachments:
    if attachment:
      return attachment.endswith(".ics") or attachment.endswith(".ICS")

  return False

def action(email):
  email.tag("RSVP")
  email.move("INBOX.Calendar")

imap.get_objects("INBOX")
imap.run_filters(filter, action)
