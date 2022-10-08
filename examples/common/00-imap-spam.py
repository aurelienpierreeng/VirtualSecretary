#!/usr/bin/env python3

"""

Detect spam using SpamAssasins and other smells

© Aurélien Pierre - 2022

"""

import os

protocols = globals()
secretary = locals()
imap = protocols["imap"]
filtername = secretary["filtername"]

dirname = os.path.dirname(filtername)

ip_blacklist_file = os.path.join(dirname, 'ip-blacklist.txt')
ip_whitelist_file = os.path.join(dirname, 'ip-whitelist.txt')

email_blacklist_file = os.path.join(dirname, 'email-blacklist.txt')
email_whitelist_file = os.path.join(dirname, 'email-whitelist.txt')


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
  global ip_whitelist, ip_blacklist, email_whitelist, email_blacklist

  # IP is known and whitelisted : exit early
  for ip in email.ip:
    if ip in ip_whitelist:
      return False

  # Email is known and whitelisted : exit early
  names, addresses = email.get_sender()
  for address in addresses:
    if address in email_whitelist:
      return False

  # IP is known and blacklisted : exit early
  for ip in email.ip:
    if ip in ip_blacklist:
      return True

  # Email is known and blacklisted : exist early
  for address in addresses:
    if address in email_blacklist:
      print("address found in blacklist")
      return True

  # IP is unknown : check for smells

  # SpamAssassins if any
  if "X-Spam-Flag" in email:
    if email["X-Spam-Flag"] == "YES":
      print("SpamAssassin found")
      return True

  # Bulk emails without unsubscribe link are usually good smells.
  # This is outright illegal in Europe by the way.
  if "Precedence" in email:
    if email["Precedence"] == "bulk":
      if "List-Unsubscribe" not in email:
        print("bad bulk message detected")
        return True

  return False

def action(email):
  email.spam(email.server.junk)

imap.get_objects("INBOX")
imap.run_filters(filter, action)
