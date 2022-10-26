#!/usr/bin/env python3

"""

Try all unsubscribe links from bulk emails if any. This will work only for
mailing lists that don't use double opt-out paradigm (the ones asking you to
confirm you want to unsubscribe by hitting a button).

© Aurélien Pierre - 2022

"""

import requests

protocols = globals()
imap = protocols["imap"]
smtp = protocols["smtp"]


def filter(email) -> bool:
  return "List-Unsubscribe" in email.headers


def action(email):
  # Open the unsubscribe link, hoping it's not a double opt-out shit
  import requests
  import urllib.parse

  # List-Unsubscribe use either URL, emails, or both.
  # Let's find out.
  for item in email["List-Unsubscribe"].split(","):
    if item.startswith("<mailto:"):
      # We have an unsubscribing email

      # Extract the email address and the query strings if any
      link = item.strip("<mailto:>\r\n\t ")
      address, query = link.split("?")

      # Init subject and body of the email
      subject = "unsubscribe %s from your mailing-list %s" % (email["To"], email["List-ID"])

      body =  "This is in response to your email: \n\"%s\",\n" % email["Subject"]
      body += "Sent by: %s, \nOn: %s,\n" % (email["From"], email["Date"])
      body += "With ID: %s,\n" % email["Message-ID"]
      body += "Belonging to your mailing-list ID: %s.\n\n" % email["List-ID"]
      body += "Please remove %s from your mailing-list.\n\n" % email["To"]
      body += "If you did not send this email, then someone is usurping your identity.\n\n"
      body += "--\n"
      body += "This is email is sent automatically by the spam filter, please do not respond."

      # Some `<mailto:...` links define a subject and a body to input in the unsubscribing email,
      # for automated handling.
      # If any, extract them from the query string and override our default fields
      if query:
        queries = urllib.parse.parse_qs(query)
        if "subject" in queries:
          subject = queries["subject"][0]
        if "body" in queries:
          body = queries["body"][0]

      # Finally, send the unsubscribe email
      global smtp
      if smtp and smtp.connection_inited:
        smtp.write_message(subject, address, body, reply_to=email)
        smtp.send_message()
      else:
        print("No SMTP connection active, emails will not be sent. Check your SMTP server credentials")

    elif item.startswith("<http"):
      # We have an unsubscribing link
      try:
        link = item.strip("<>\r\n\t ")
        # Set a short timeout because most unsubscribe URLs in spams are bogus and reach nowhere,
        # so don't make the script hang for nothing.
        result = requests.get(link, timeout=10)
        result.raise_for_status()
      except requests.exceptions.RequestException:
        print("Could not reach unsubscribe link", link)
        pass
      else:
        email.server.logfile.write("Tried to unsubscribe from %s with no guaranty\n" % link)


imap.get_objects(imap.junk)
imap.run_filters(filter, action)
