#!/usr/bin/env python3

"""

Fetch the IP of spam emails and update the IP blacklist.
Run this only if you manually checked spam and inbox folders.

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
  global ip_whitelist, ip_blacklist, email_whitelist, email_blacklist

  for ip in email.ip:
    if ip not in ip_blacklist:
      if ip not in ip_whitelist:
        # IP is completely unknown, blacklist it
        ip_blacklist.append(ip)
      else:
        # IP is known in whitelist, but now it's found in spam folder.
        # It's either a false positive or the server got compromised.
        # Remove it from all lists aka "greylist" it.
        ip_whitelist.remove(ip)
    else:
      # IP is already known in blacklist, do nothing
      pass

  names, addresses = email.get_sender()

  for address in addresses:
    if address not in email_blacklist and address not in email_whitelist:
      # Email is completely unknown, blacklist it
      email_blacklist.append(address)
    elif address in email_blacklist and address in email_whitelist:
      # Email is known in both lists, remove it from everywhere aka greylist it
      email_blacklist.remove(address)
      email_whitelist.remove(address)
    else:
      # Email is known either in whitelist, blacklist

      # If email is whitelisted, it is a tricky case because valid email addresses
      # can be impersonated by spammers, contrarily to server IPs.
      # So the email address could still be valid and only used as a "tag" in email.
      # To prevent this, SPF and DKIM should be provided and checked, but many self-hosted
      # emails accounts don't properly implement those solutions.
      # In that case, give them the benefit of doubt and do nothing.

      # If email is blacklisted already, nothing to do.
      pass

  return False

print("Parsing %s for blacklisting" % imap.junk)
imap.get_objects(imap.junk)
imap.run_filters(filter, None)


# Fetch the IPs from emails in main inbox and remove them from the blacklist.
# This is to account for possible false-positives and give a second chance to IPs
# that are deemed OK, provided the main inbox is manually checked.
def filter(email) -> bool:
  global ip_whitelist, ip_blacklist, email_whitelist, email_blacklist

  names, addresses = email.get_sender()

  if addresses[0] == email.server.user:
    # This is an email sent by the current mailbox, discard it because it has
    # no IP address and sender email is ourselves.
    return False

  for ip in email.ip:
    if ip in ip_blacklist:
      # IP is known in blacklist, remove it from there.
      ip_blacklist.remove(ip)

      if ip in ip_whitelist:
        # IP is known also in whitelist, remove it also from there,
        # aka greylist it.
        ip_whitelist.remove(ip)

    elif ip not in ip_whitelist:
      # IP is entirely unknown, whitelist it
      ip_whitelist.append(ip)

  for address in addresses:
    if address in email_blacklist:
      # Email is known in blacklist, remove it from there.
      email_blacklist.remove(address)

      if address in email_whitelist:
        # Email is also known in whitelist, remove it from there too,
        # aka greylist it.
        email_whitelist.remove(address)

    elif address not in email_whitelist:
      # Email is entirely unknown, whitelist it.
      email_whitelist.append(address)

  return False

for folder in imap.folders:
  if folder != imap.junk:
    print("Parsing %s for whitelisting" % folder)
    # Restore hidden folders, delete them if needed but don't hide them.
    imap.subscribe(folder)
    imap.get_objects(folder)
    imap.run_filters(filter, None, runs=-1)


# Save the updated list files
write_lists(ip_blacklist_file, ip_blacklist)
write_lists(ip_whitelist_file, ip_whitelist)

write_lists(email_blacklist_file, email_blacklist)
write_lists(email_whitelist_file, email_whitelist)


# At this point, we have 2 lists :
# * white list : IPs that we know are good, as long as they don't send spam
# * black list : IPs that we know are bad, until some legit email is found in spam box
