#!/usr/bin/env python3

"""

Notification of leave.

© Aurélien Pierre - 2022

"""

protocols = globals()
imap = protocols["imap"]

from datetime import datetime
from dateutil import tz

# Local timezone, mandatory.
# Availaible zones are listed in https://en.wikipedia.org/wiki/List_of_tz_database_time_zones#List
# in the `TZ database name` column
local_timezone = tz.gettz('Europe/Paris')

# mandatory date format is numeric, with year, month, day
# then hour, minute and second are optional.
# tzinfo is mandatory to compare dates with internal email date.
leave_start = datetime(2022, 10, 1, hour=16, minute=0, tzinfo=local_timezone)
leave_end = datetime(2022, 10, 24, hour=8, minute=0, tzinfo=local_timezone)


def filter(email) -> bool:
  global leave_end, leave_start

  # Email date is in-between leave bounds
  return email.date > leave_start and email.date < leave_end and email.is_not_read()


def action(email):
  # Tag it so we can discard later autoresponders if needed
  from babel.dates import format_datetime
  global leave_end

  email.tag("Leave")

  global smtp
  if smtp and smtp.connection_inited:
    subject = email["Subject"]

    # Prepend the reply marker if none
    if not subject.startswith("Re:"):
      subject = "Re: " + subject

    subject = "[Absence] " + subject

    body = ""

    body += "[This is an automated answer from the mail server.]\r\n\r\n"
    body += "Your message to %s sent on %s has been properly delivered on %s.\r\n\r\n" % (email["To"], email["Date"], email["Delivery-date"])
    body += "We are out of office until %s, so it will be read later. " % format_datetime(leave_end, locale="en")
    body += "Thank you for your cooperation !"

    body += "\r\n\r\n---\r\n\r\n"

    body += "[Ceci est un message automatique du serveur d'emails]\r\n\r\n"
    body += "Votre message à %s envoyé le %s a été correctement délivré le %s.\r\n\r\n" % (email["To"], email["Date"], email["Delivery-date"])
    body += "Nous sommes absents jusqu'au %s, il sera lu plus tard. " % format_datetime(leave_end, locale="fr")
    body += "Merci pour votre coopération !"

    smtp.write_message(subject, email["From"], body, reply_to=email)

    # Send the message and copy it through IMAP in the `Sent` folder
    # This is mandatory for the next step in 32-imap-autoresponder-urgent,
    # which needs the whole thread of emails to act precisely on the one needed.
    smtp.send_message(copy_to_sent=False)

    email.server.logfile.write("%s : Leave auto-responder sent a message to %s in reply to %s\n" % (email.now(), email["From"], email["Subject"]))


now = datetime.now(local_timezone)

if now > leave_start and now < leave_end:
  print("  Leave autoresponder started")
  imap.get_objects("INBOX")
  imap.run_filters(filter, action)
else:
  print("  No active leave found")
