#!/usr/bin/env python3

protocols = globals()
imap = protocols["imap"]
smtp = protocols["smtp"]


def filter(email) -> bool:
  # If the email is older than 7 days, do nothing and abort.
  # We bail early because those checks are inexpensive to compute.
  from datetime import timedelta
  if email.age() > timedelta(days=7) or not "References" in email.headers:
    return False

  # Look for the URGENT token in body.
  # Slightly more expensive to check since the body needs to be parsed entirely.
  if not email.is_in("URGENT", "Body", case_sensitive = True):
    # If not found, abort immediately
    return False

  # Try to find if the current email is a reply to an automated message we sent in filter 31-imap-autoresponder-busy.
  # Much more expensive than other checks since it needs to query the server through the network.
  email_replied_to = email.query_replied_email()
  return email_replied_to and ("X-Mailer" in email_replied_to.headers) and \
     email_replied_to["X-Mailer"] == "Virtual Secretary"


def action(email):
  # Find the parent emails, which "Message-ID" needs to be saved in "References".
  # This is done by most modern email clients to thread emails, but is not mandatory.
  # Aka the success of this relies on the behaviour of the email client.
  global smtp

  # Get the automated answer we sent in 31-imap-autoresponder-busy (parent).
  # The current email is the human answer to that.
  auto_reply = email.query_replied_email()

  # Get the original email to which 31-imap-autoresponder-busy responded automatically (grand-parent)
  original = auto_reply.query_replied_email()

  if original:
    original.mark_as_important("add")
    original.tag("Urgent")

    if original.server.std_out[0] == "OK" and smtp and smtp.connection_inited:
      # Things went well, send a confirmation email
      subject = email["Subject"] # This should already start with "Re: "

      body = ""
      body += "[Version française en bas d'email]\r\n[This is an automated answer from the mail server.]\r\n\r\n"
      body += "Your message to %s sent on %s " % (original["To"], original["Date"])
      body += "has been properly marked as urgent on the mail server. It will be processed as soon as possible.\r\n\r\n"
      body += "You may delete the automated replies, including this one, to keep a clean thread.\r\n\r\n"
      body += "Thank you for your cooperation !"

      body += "\r\n\r\n---\r\n\r\n"

      body += "[English version on top of the email]\r\n[Ceci est un message automatique du serveur d'emails]\r\n\r\n"
      body += "Votre message à %s envoyé le %s " % (original["To"], original["Date"])
      body += "a été correctement marqué urgent sur le serveur d'emails. Il sera traité dès que possible.\r\n\r\n"
      body += "Vous pouvez supprimer les messages automatiques, incluant celui-ci, pour garder un fil propre.\r\n\r\n"
      body += "Merci pour votre coopération !"

      # Send and keep a copy
      smtp.write_message(subject, email["From"], body, reply_to=email)
      smtp.send_message(copy_to_sent=True)

      email.server.logfile.write("%s : Business auto-responder sent a message to %s in reply to %s\n" % (email.now(), email["From"], email["Subject"]))

    # Remove the service email containing "URGENT"
    email.delete()

imap.get_objects("INBOX")
imap.run_filters(filter, action)
