#!/usr/bin/env python3

protocols = globals()
imap = protocols["imap"]
smtp = protocols["smtp"]

# We consider the mailbox "busy" if, among the n last messages fetched in the mailbox,
# we have a number of unread or flagged messages higher than the thresholds below.
UNREAD_THRS = 10
IMPORTANT_THRS = 5

# Get the messages and count how many are unread and important
imap.get_objects("INBOX")
mailbox_is_busy = imap.count("is_unread") > UNREAD_THRS or imap.count("is_important") > IMPORTANT_THRS


def filter(email) -> bool:
  # This filter runs at position 31, meaning we assume spam and automated messages have already
  # been filtered out of the way and all the remaining messages are sent by humans.

  from datetime import timedelta
  global mailbox_is_busy

  # If the email has been read already or is older than 1 day or has been already handled by the leave autoresponder,
  # do nothing and bail early
  if email.age() > timedelta(days=1) or email.is_read() or not mailbox_is_busy or email.has_tag("Leave"):
    return False

  # Emails with "URGENT" in body are treated in filter 32-imap-autoresponder-urgent
  # and may already be an answer to an auto-email sent here.
  return not email.is_in("URGENT", "Body", case_sensitive=True)


def action(email):
  global smtp
  if smtp and smtp.connection_inited:
    subject = email["Subject"]

    # Prepend the reply marker if none
    if not subject.startswith("Re:"):
      subject = "Re: " + subject

    body = ""

    body += "[Version française en bas d'email]\r\n[This is an automated answer from the mail server.]\r\n\r\n"
    body += "Your message to %s sent on %s has been properly delivered on %s.\r\n\r\n" % (email["To"], email["Date"], email["Delivery-date"])
    body += "However, this mailbox has got a lot of new emails recently, and "
    body += "yours may not be read before some time.\r\n\r\n"
    body += "If your message requires time-sensitive action or answer, please reply \"URGENT\" to this email. "
    body += "The system will then put your previous email on top of the waiting queue and send you back a confirmation in the next 15 minutes. "
    body += "If you don't get a confirmation in the next 15 minutes, the operation would have failed.\r\n\r\n"
    body += "Thank you for your cooperation !"

    body += "\r\n\r\n---\r\n\r\n"

    body += "[English version on top of the email]\r\n[Ceci est un message automatique du serveur d'emails]\r\n\r\n"
    body += "Votre message à %s envoyé le %s a été correctement délivré le %s.\r\n\r\n" % (email["To"], email["Date"], email["Delivery-date"])
    body += "Cependant, cette messagerie a reçu de nombreux emails récemment, et "
    body += "le vôtre pourrait ne pas être lu avant un certain temps.\r\n\r\n"
    body += "Si votre message requiert une action ou une réponse urgente, merci de répondre \"URGENT\" à cet email. "
    body += "Le système mettra alors votre précédent email en haut de la pile en attente et vous enverra une confirmation dans les 15 minutes. "
    body += "Si vous ne recevez pas de confirmation dans les prochaines 15 minutes, l'opération aura échoué.\r\n\r\n"
    body += "Merci pour votre coopération !"

    smtp.write_message(subject, email["From"], body, reply_to=email)

    # Send the message and copy it through IMAP in the `Sent` folder
    # This is mandatory for the next step in 32-imap-autoresponder-urgent,
    # which needs the whole thread of emails to act precisely on the one needed.
    smtp.send_message(copy_to_sent=True)

    email.server.logfile.write("%s : Business auto-responder sent a message to %s in reply to %s\n" % (email.now(), email["From"], email["Subject"]))


imap.run_filters(filter, action)
