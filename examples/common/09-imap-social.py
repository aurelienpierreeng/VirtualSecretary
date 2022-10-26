#!/usr/bin/env python3

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  sender = email["From"].lower()
  filtered = False

  if "notification@slack.com" in sender:
    email.move("INBOX.Social.Slack")
    filtered = True
  elif "no-reply@modelmayhem.com" in sender:
    email.move("INBOX.Social.ModelMayhem")
    filtered = True

  return filtered

imap.get_objects("INBOX")
imap.run_filters(filter, None)
