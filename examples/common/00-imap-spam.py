#!/usr/bin/env python3

"""

Detect spam using SpamAssasin headers and other smells

© Aurélien Pierre - 2022

"""

import os
import requests

GLOBAL_VARS = globals()
mailserver = GLOBAL_VARS["mailserver"]
filtername = GLOBAL_VARS["filtername"]


ip_blacklist_file = os.path.join(os.path.dirname(filtername), 'ip-blacklist.txt')
ip_whitelist_file = os.path.join(os.path.dirname(filtername), 'ip-whitelist.txt')

email_blacklist_file = os.path.join(os.path.dirname(filtername), 'email-blacklist.txt')
email_whitelist_file = os.path.join(os.path.dirname(filtername), 'email-whitelist.txt')


def load_lists(file:str) -> list:
  try:
    with open(file, 'r') as f:
      return f.read().split("\n")
  except:
    return []


# Load the previous blacklist file if any
ip_blacklist = load_lists(ip_blacklist_file)
ip_whitelist = load_lists(ip_whitelist_file)
email_blacklist = load_lists(email_blacklist_file)
email_whitelist = load_lists(email_whitelist_file)


def filter(email) -> bool:
  # IP is known and whitelisted : exit early
  for ip in email.ip:
    if ip in ip_whitelist:
      return False

  # Email is known and whitelisted : exit early
  if email.sender_email in email_whitelist:
    return False

  # IP is known and blacklisted : exit early
  for ip in email.ip:
    if ip in ip_blacklist:
      return True

  # Email is known and blacklisted : exist early
  if email.sender_email in email_blacklist:
    return True

  # IP is unknown : check for smells

  # SpamAssassin headers if any
  if "X-Spam-Flag" in email.header:
    if email.header["X-Spam-Flag"] == "YES":
      return True

  # Bulk emails without unsubscribe link are usually good smells.
  # This is outright illegal in Europe by the way.
  if "Precedence" in email.header:
    if (email.header["Precedence"] == "bulk" and not ("List-Unsubscribe" in email.header)):
      return True

  return False

def action(email):
  email.spam("INBOX.spam")

mailserver.get_mailbox_emails("INBOX")
mailserver.filters(filter, action, filtername, runs=-1)
