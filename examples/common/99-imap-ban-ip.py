#!/usr/bin/env python3

"""

Fetch the IP of spam emails and update the IP blacklist.
Run this only if you manually checked spam and inbox folders.

© Aurélien Pierre - 2022

"""

import os
import re

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


def write_lists(file:str, l: list):
  try:
    with open(file, 'w') as f:
      f.write("\n".join(sorted(l)))
  except:
    print("Saving files failed, check you have writing permissions")


# Load the previous blacklist file if any
ip_blacklist = load_lists(ip_blacklist_file)
ip_whitelist = load_lists(ip_whitelist_file)
email_blacklist = load_lists(email_blacklist_file)
email_whitelist = load_lists(email_whitelist_file)


# Fetch the IPs from emails in spam box and add them to the blacklist
def filter(email) -> bool:
  for ip in email.ip:
    if ip not in ip_blacklist:
      ip_blacklist.append(ip)

  for address in email.sender_email:
    if address not in email_blacklist:
      email_blacklist.append(address)

  return False

def action(email):
  return

mailserver.get_mailbox_emails(email.mailserver.spam)
mailserver.filters(filter, action, filtername, runs=-1)


# Fetch the IPs from emails in main inbox and remove them from the blacklist.
# This is to account for possible false-positives and give a second chance to IPs
# that are deemed OK, provided the main inbox is manually checked.
def filter(email) -> bool:
  for ip in email.ip:
    if ip in ip_blacklist:
      ip_blacklist.remove(ip)
    if ip not in ip_whitelist:
      ip_whitelist.append(ip)

  for address in email.sender_email:
    if address in email_blacklist:
      email_blacklist.remove(address)
    if address not in email_whitelist:
      email_whitelist.append(address)

  return False

def action(email):
  return

mailserver.get_mailbox_emails("INBOX")
mailserver.filters(filter, action, filtername, runs=-1)


# Fetch again IPs from spam folder and remove anything we find
# from the whitelist. This is to account for possible false-negatives.
# This ensures that IPs/emails that are both in inbox and spam folders
# are recorded neither in black or white lists.
def filter(email) -> bool:
  for ip in email.ip:
    if ip in ip_whitelist:
      ip_whitelist.remove(ip)

  for address in email.sender_email:
    if address in email_whitelist:
      email_whitelist.remove(address)

  return False

def action(email):
  return

mailserver.get_mailbox_emails(email.mailserver.spam)
mailserver.filters(filter, action, filtername, runs=-1)


# Save the updated list files
write_lists(ip_blacklist_file, ip_blacklist)
write_lists(ip_whitelist_file, ip_whitelist)

write_lists(email_blacklist_file, email_blacklist)
write_lists(email_whitelist_file, email_whitelist)



# At this point, we have 2 lists :
# * white list : IPs that we know are good, as long as they don't send spam
# * black list : IPs that we know are bad, until some legit email is found in spam box
