#!/usr/bin/env python3

import imaplib
from email.utils import formatdate, make_msgid, parsedate_to_datetime


protocols = globals()
imap = protocols["imap"]
instagram = protocols["instagram"]

def filter(elem) -> bool:
  #print(elem.to_email().as_string())
  return True

def action(elem):
  global imap
  if imap and imap.connection_inited:
    email = elem.to_email()
    imap.append("INBOX.Social.Instagram", "(Auto)", email)

#imap.get_objects("INBOX")
instagram.run_filters(filter, action)
