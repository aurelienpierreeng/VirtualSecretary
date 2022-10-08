import imaplib
import quopri
import utils
import email
import re
import pickle
import os
import hashlib
import html
import time
import connectors

from email import policy

from datetime import datetime

ip_pattern = re.compile(r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]")
email_pattern = re.compile(r"([0-9a-zA-Z\-\_\+\.]+?@[0-9a-zA-Z\-\_\+]+?\.[a-zA-Z]{2,})")
url_pattern = re.compile(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)")
uid_pattern = re.compile(r"UID ([0-9]+) ")
flags_pattern = re.compile(r"\(FLAGS \((.*)\)")


class EMail(connectors.Content):

  ##
  ## Data parsing / decoding
  ##

  def parse_uid(self, raw):
    self.uid = uid_pattern.search(raw).groups()[0]

  def parse_flags(self, raw):
    self.flags = flags_pattern.search(raw).groups()[0]

  def parse_ips(self):
    # Get the whole network route taken by the email
    network_route = self.msg.get_all("Received")
    network_route = "; ".join(network_route)
    self.ips = ip_pattern.findall(network_route)

    # Remove localhost
    self.ips = [ip for ip in self.ips if not ip.startswith("127.")]

  @property
  def ip(self):
    # Lazily parse IPs in email only when/if the property is used
    if self.ips == []:
      self.parse_ips()
    return self.ips


  def get_sender(self) -> list[list, list]:
    emails = email_pattern.findall(self["From"])
    names = re.findall(r"\"(.+)?\"", self["From"])
    out = [names, emails]
    return out


  def parse_urls(self, input:str):
    # Output a list of all URLs found in email body.
    # Each result in the list is a tuple (domain, page), for example :
    # `google.com/index.php` is broken into `('google.com', '/index.php')`
    # `google.com/` is broken into `('google.com', '/')`
    # `google.com/login.php?id=xxx` is broken into `('google.com', '/login.php')`

    try:
      self.urls.append(url_pattern.findall(input))
    except:
      print("Could not parse urls in email body :")
      print(input)

    # Flatten the list 2D list
    self.urls = [item for sublist in self.urls for item in sublist]


  def __getitem__(self, key):
    # Getting key from the class is dispatched directly to email.EmailMessage properties
    return self.msg.get(key)

  xml_pattern = re.compile(r"<.*?>", re.DOTALL)
  style_pattern = re.compile(r"(<style.*?>)(.*?)(</style>)", re.DOTALL)
  img_pattern = re.compile(r"<img.*?\/?>", re.DOTALL)
  multiple_spaces = re.compile(r"[\t ]{2,}", re.UNICODE)
  multiple_lines = re.compile(r"\s{2,}", re.UNICODE)

  def remove_html(self, input:str):
    # Fetch urls now because they will be removed with markup
    self.parse_urls(input)

    # Remove HTML/XML markup
    output = re.sub(self.style_pattern, '', input)
    output = re.sub(self.img_pattern, '', output)
    output = re.sub(self.xml_pattern, ' ', output)

    # Decode HTML entities
    output = html.unescape(output)

    # Collapse multiple whitespaces
    output = re.sub(self.multiple_spaces, ' ', output)
    output = re.sub(self.multiple_lines, '\n', output)

    return output

  def get_body(self, preferencelist=('related', 'html', 'plain')):
    body = self.msg.get_body(preferencelist)

    # For emails providing HTML only, build a plain text version from
    # the HTML one by removing markup
    build_plain = False
    if not body and preferencelist == "plain":
      body = self.msg.get_body("html")
      build_plain = True

    if body:
      charset = body.get_content_charset()
      encoding = body["Content-Transfer-Encoding"]
      content = body.get_content()

      if build_plain:
        content = self.remove_html(content)
      else:
        # Fetch urls now because they will be removed with markup
        self.parse_urls(content)

        # Even if it's plain text, there might be html entities/markup in it
        # so clean it.
        content = re.sub(self.xml_pattern, "", content)
        content = html.unescape(content)

      if encoding == "quoted-printable" and charset != "utf-8":
        # That should be handled because it's prone to errors,
        # but affects only spam messages thus low-priority.
        pass

    return content

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
                                                                                    self["Subject"],
                                                                                    self["From"],
                                                                                    self["Date"]))

    self.server.std_out = result

  def untag(self, keyword):
    # Remove a tag.
    result = self.server.uid('STORE', self.uid, '-FLAGS', keyword)

    if result[0] == "OK":
      self.server.logfile.write("%s : Tag `%s` removed from email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    keyword,
                                                                                    self.uid,
                                                                                    self["Subject"],
                                                                                    self["From"],
                                                                                    self["Date"]))

    self.server.std_out = result

  def delete(self):
    # Delete an email directly without using the trash bin.
    # Use a move to trash folder to get a last chance at reviewing what will be deleted.
    result = self.server.uid('STORE', self.uid, '+FLAGS', "(\\Deleted)")

    if result[0] == "OK":
      self.server.logfile.write("%s : Deteled email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    self.uid,
                                                                                    self["Subject"],
                                                                                    self["From"],
                                                                                    self["Date"]))

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
                                                                                          self["Subject"],
                                                                                          self["From"],
                                                                                          self["Date"]))

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
                                                                                                          self["Subject"],
                                                                                                          self["From"],
                                                                                                          self["Date"],
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
    #   if you answer programmatically, you need to manually pass the Message-ID of the original email
    #   to the In-Reply-To and Referencess of the answer to get threaded messages. In-Reply-To gets only
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
%s""" % (self["Message-ID"],
        self["Subject"],
        self["From"],
        self["Date"],
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
    date_time = email.utils.parsedate_to_datetime(self["Date"])
    timestamp = str(int(datetime.timestamp(date_time)))

    # Hash the Received, which is the most unique
    # field as it contains the server route of the email along with
    # all the times.
    hash = hashlib.md5(self["Received"].encode())

    # Our final hash is the Unix timestamp for easy finding and Received hash
    self.hash = timestamp + "-" + hash.hexdigest()

  def __init__(self, raw_message:list, server) -> None:
    # Position of the email in the server list
    super().__init__(raw_message, server)

    self.urls = []
    self.ips = []
    self.attachments = []

    # Raw message as fetched by IMAP, decode IMAP-specifics
    email_content = raw_message[0].decode()
    self.parse_uid(email_content)
    self.parse_flags(email_content)

    # Decode RFC822 email body
    self.msg = email.message_from_bytes(raw_message[1], policy=email.policy.default)
    self.create_hash()