import imaplib
import utils
import email
import re
import pickle
import os
import hashlib
import html
import time
import connectors

from datetime import datetime

ip_pattern = re.compile(r"\[(\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3})\]")
email_pattern = re.compile(r"([0-9a-zA-Z\-\_\+]+?@[0-9a-zA-Z\-\_\+]+?\.[a-zA-Z]{2,})")
url_pattern = re.compile(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)")
uid_pattern = re.compile(r"UID ([0-9]+) ")
flags_pattern = re.compile(r"\(FLAGS \((.*)\)")
xml_pattern = re.compile(r"<.*?>", re.DOTALL)
style_pattern = re.compile(r"(<head.*?>)(.*?)(</head>)", re.DOTALL)
img_pattern = re.compile(r"<img.*?\/?>", re.DOTALL)
multiple_spaces = re.compile(r"\s{2,}", re.UNICODE)
multiple_line_breaks = re.compile(r"\s{2,}")

class EMail(connectors.Content):

  ##
  ## Data parsing / decoding
  ##

  def parse_uid(self, raw):
    self.uid = uid_pattern.search(raw).groups()[0]

  def parse_flags(self, raw):
    self.flags = flags_pattern.search(raw).groups()[0]

  def parse_ips(self):
    result = ip_pattern.search(self.msg.as_string())
    self.ip = result.groups() if result is not None else None

  def parse_email(self):
    result = email_pattern.search(self.header["From"])
    self.sender_email = result.groups() if result is not None else None

  def parse_urls(self):
    # Output a list of all URLs found in email body.
    # Each result in the list is a tuple (domain, page), for example :
    # `google.com/index.php` is broken into `('google.com', '/index.php')`
    # `google.com/` is broken into `('google.com', '/')`
    # `google.com/login.php?id=xxx` is broken into `('google.com', '/login.php')`
    self.urls = []

    try:
      self.urls.append(url_pattern.findall(self.body["text/plain"]))
    except:
      print("Could not parse urls in email body (plain text) for %s :" % self.header["Subject"])
      print(self.body["text/plain"])

    try:
      self.urls.append(url_pattern.findall(self.body["text/html"]))
    except:
      print("Could not parse urls in email body (HTML) for %s" % self.header["Subject"])

    # Flatten the list 2D list
    self.urls = [item for sublist in self.urls for item in sublist]


  def parse_headers(self):
    self.header = { }

    for elem in ["Message-ID", "Subject", "Date", "From", "To",               # Mandatory fields
                 "Return-Path", "Envelope-to", "Delivery-date", "Reply-To",   # Optional
                 "MIME-Version", "Content-Type", "Content-Transfer-Encoding", # Content hints
                 "DKIM-Signature",                                            # Mailbox authentification
                 "Received",                                                  # Network route followed
                 "List-Unsubscribe", "List-Unsubscribe-Post", "List-ID", "Feedback-ID", "Precedence", # Newsletter and bulk sendings
                 "X-Mailer", "X-Csa-Complaints", "X-Mailin-Campaign", "X-Mailin-Client",              # Custom stuff from bulk sendings
                 "X-Spam-Status", "X-Spam-Score", "X-Spam-Bar", "X-Ham-Report", "X-Spam-Flag",        # SpamAssassin headers
                 ]:

      if elem in self.msg:
        self.header[elem] = str(email.header.make_header(email.header.decode_header(self.msg.get(elem))))

    # Sanitize things for those weird malformed spam emails
    if "Message-ID" not in self.header:
      self.header["Message-ID"] = ""

    if "Subject" not in self.header:
      self.header["Subject"] = ""


  def parse_body(self):
    # Extract email body
    self.body = { "text/plain": "", "text/html" : "" }
    charset = "utf-8"

    if self.msg.is_multipart():
      # Emails with attachments have multiple parts, we iterate over them to find the body
      for part in self.msg.walk():
        is_attachment = "attachment" in str(part.get("Content-Disposition"))

        if is_attachment:
          self.attachments.append(part.get_filename())

        body_type = part.get_content_type()
        if body_type in ["text/plain", "text/html"] and not is_attachment:
            self.body[body_type] = part.get_payload(decode=True)
            charset = part.get_content_charset()
    else:
      body_type = self.msg.get_content_type()
      if body_type in ["text/plain", "text/html"]:
        self.body[body_type] = self.msg.get_payload(decode=True)
        charset = self.msg.get_content_charset()
      else:
        print("Warning : email content type ", self.msg.get_content_type(), "is not supported")

    charset = "utf-8" if not charset else charset

    # Decode plain text
    try:
      self.body["text/plain"] = self.body["text/plain"].decode(charset)
    except:
      try:
        # If charset is not specified or detected and UTF-8 failed, next best guess is Windows BS
        self.body["text/plain"] = self.body["text/plain"].decode("windows-1251")
      except:
        pass

    # Decode HTML
    try:
      self.body["text/html"] = self.body["text/html"].decode(charset)
    except:
      try:
        # If charset is not specified or detected and UTF-8 failed, next best guess is Windows BS
        self.body["text/html"] = self.body["text/html"].decode("windows-1251")
      except:
        pass

    # Manage fallbacks if one version of the email is absent

    if self.body["text/html"] == "" and isinstance(self.body["text/plain"], str):
      self.body["text/html"] = self.body["text/plain"]

    if self.body["text/plain"] == "" and isinstance(self.body["text/html"], str):
      # For HTML emails not providing plain-text version, remove markup
      self.body["text/plain"] = self.body["text/html"]

      # Remove HTML/XML markup
      self.body["text/plain"] = re.sub(style_pattern, '', self.body["text/plain"])
      self.body["text/plain"] = re.sub(img_pattern, '', self.body["text/plain"])
      self.body["text/plain"] = re.sub(xml_pattern, '', self.body["text/plain"])

      # Decode HTML entities
      self.body["text/plain"] = html.unescape(self.body["text/plain"])

      # Collapse multiple whitespaces
      self.body["text/plain"] = re.sub(multiple_spaces, '\n', self.body["text/plain"])


  ##
  ## IMAP ACTIONS
  ##

  def tag(self, keyword:str):
    # Add a tag.
    result = self.server.uid('STORE', self.uid, '+FLAGS', keyword)

    if result[0] == "OK":
      self.server.logfile.write("%s : Tag `%s` added to email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    keyword,
                                                                                    self.uid,
                                                                                    self.header["Subject"],
                                                                                    self.header["From"],
                                                                                    self.header["Date"]))

    self.server.std_out = result

  def untag(self, keyword):
    # Remove a tag.
    result = self.server.uid('STORE', self.uid, '-FLAGS', keyword)

    if result[0] == "OK":
      self.server.logfile.write("%s : Tag `%s` removed from email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    keyword,
                                                                                    self.uid,
                                                                                    self.header["Subject"],
                                                                                    self.header["From"],
                                                                                    self.header["Date"]))

    self.server.std_out = result

  def delete(self):
    # Delete an email directly without using the trash bin.
    # Use a move to trash folder to get a last chance at reviewing what will be deleted.
    result = self.server.uid('STORE', self.uid, '+FLAGS', "(\\Deleted)")

    if result[0] == "OK":
      self.server.logfile.write("%s : Deteled email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    self.uid,
                                                                                    self.header["Subject"],
                                                                                    self.header["From"],
                                                                                    self.header["Date"]))

      # delete the email from the list of emails in the server object to avoid further manipulation
      self.server.emails.remove(self)

    self.server.std_out = result

  def spam(self, spam_folder="INBOX.spam"):
    # Mark an email as spam using Thunderbird tags and move it to the spam/junk folder
    self.server.uid('STORE', self.uid, '-FLAGS', 'NonJunk')
    self.server.uid('STORE', self.uid, '+FLAGS', 'Junk')
    self.server.uid('STORE', self.uid, '-FLAGS', '(\\Seen)')

    result = self.server.uid('COPY', self.uid, spam_folder)

    if result[0] == "OK":
      result = self.server.uid('STORE', self.uid, '+FLAGS', '(\\Deleted)')

      if result[0] == "OK":
        self.server.logfile.write("%s : Spam email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                          self.uid,
                                                                                          self.header["Subject"],
                                                                                          self.header["From"],
                                                                                          self.header["Date"]))

    self.server.std_out = result

  def move(self, folder:str):
    # create the folder and update the list of folders
    if folder not in self.server.folders:
      result = self.server.create(folder)

    result = self.server.subscribe(folder)

    if result[0] == "OK":
      self.server.logfile.write("%s : Folder`%s` created in INBOX\n" % (utils.now(), folder))

    self.server.get_imap_folders()

    result = self.server.uid('COPY', self.uid, folder)

    if result[0] == "OK":
      result = self.server.uid('STORE', self.uid, '+FLAGS', '(\\Deleted)')
      if result[0] == "OK":
        self.server.logfile.write("%s : Moved email (UID %s) `%s` from `%s` received on %s to %s\n" % (utils.now(),
                                                                                                          self.uid,
                                                                                                          self.header["Subject"],
                                                                                                          self.header["From"],
                                                                                                          self.header["Date"],
                                                                                                          folder))
        # delete the email from the list of emails in the server object to avoid further manipulation
        # because the emails list is tied to a particular mailbox folder.
        self.server.emails.remove(self)

    self.server.std_out = result


  def mark_as_important(self, mode:str):
    # Flag or unflag an email as important
    tag = "(\\Flagged)"
    if mode == "add":
      self.tag(tag)
    elif mode == "remove":
      self.untag(tag)


  def mark_as_read(self, mode:str):
    # Flag or unflag an email as read (seen)
    tag = "(\\Seen)"
    if mode == "add":
      self.tag(tag)
    elif mode == "remove":
      self.untag(tag)


  def mark_as_answered(self, mode:str):
    # Flag or unflag an email as answered
    # Note :
    #   if you answer programmatically, you need to manually pass the Message-ID header of the original email
    #   to the In-Reply-To and References headers of the answer to get threaded messages. In-Reply-To gets only
    #   the immediate previous email, References get the whole thread.

    tag = "(\\Answered)"
    if mode == "add":
      self.tag(tag)
    elif mode == "remove":
      self.untag(tag)


  ##
  ## Checks
  ##

  def is_read(self) -> bool:
    # Email has been opened and read
    return "\\Seen" in self.flags

  def is_recent(self) -> bool:
    # This session is the first one to get this email. Doesn't mean user read it.
    # Note : this flag cannot be set by client, only by server. It's read-only app-wise.
    return "\\Recent" in self.flags

  def is_draft(self) -> bool:
    # This email is maked as draft
    return "\\Draft" in self.flags

  def is_answered(self) -> bool:
    # This email has been answered
    return "\\Answered" in self.flags

  def is_important(self) -> bool:
    # This email has been flagged as important
    return "\\Flagged" in self.flags


  ##
  ## Utils
  ##


  def now(self):
    # Helper to get access to date/time from within the email object when writing filters
    return utils.now()


  def __str__(self) -> str:
    # Print an email
    return """
=====================================================================================
ID\t:\t%s
SUBJECT\t:\t%s
FROM\t:\t%s
ON\t:\t%s
UID\t:\t%s
FLAGS\t:\t%s
Attachments : %s
-------------------------------------------------------------------------------------
%s""" % (self.header["Message-ID"],
        self.header["Subject"],
        self.header["From"],
        self.header["Date"],
        self.uid,
        self.flags,
        self.attachments,
        (self.body[:300] + '..') if len(self.body) > 300 else self.data) # shorten body if needed

    # Support the global std for compatibility in filters
    self.server.std_out = ["OK", ]

  def create_hash(self):
    # IMAP UID are linked to a particular mailbox. When we move
    # an email to another mailbox (folder), the UID is changed.
    # This can't be used to log emails in a truly unique way,
    # so we need to create our own hash.

    # Convert the email date to Unix timestamp
    date_time = email.utils.parsedate_to_datetime(self.header["Date"])
    timestamp = str(int(datetime.timestamp(date_time)))

    # Hash the Received header, which is the most unique
    # field as it contains the server route of the email along with
    # all the times.
    hash = hashlib.md5(self.header["Received"].encode())

    # Our final hash is the Unix timestamp for easy finding and Received hash
    self.hash = timestamp + "-" + hash.hexdigest()

  def __init__(self, raw_message:list, server) -> None:
    # Position of the email in the server list
    super().__init__(raw_message, server)

    # Raw message as fetched by IMAP
    email_content = raw_message[0].decode()
    self.parse_uid(email_content)
    self.parse_flags(email_content)

    # Decoded message
    self.msg = email.message_from_bytes(raw_message[1])

    # Extract and decode header fields in a dictionnary
    self.parse_headers()
    self.parse_ips()
    self.parse_email()
    self.create_hash()

    # Extract and decode body
    self.attachments = []
    self.parse_body()
    self.parse_urls()

    # Post URL pattern for regex filtering
    self.url_pattern = url_pattern
    self.email_pattern = email_pattern
