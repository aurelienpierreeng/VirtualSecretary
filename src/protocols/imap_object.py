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

from datetime import datetime, timedelta, timezone

ip_pattern = re.compile(r"\[(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\]")
email_pattern = re.compile(r"<?([0-9a-zA-Z\-\_\+\.]+?@[0-9a-zA-Z\-\_\+]+(\.[0-9a-zA-Z\_\-]{2,})+)>?")
url_pattern = re.compile(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)")
uid_pattern = re.compile(r"UID ([0-9]+) ")
flags_pattern = re.compile(r"FLAGS \((.*)?\)")


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
    if "Received" in self.headers:
      # There will be no "Received" header for emails sent by the current mailbox
      network_route = self.msg.get_all("Received")
      network_route = "; ".join(network_route)
      self.ips = ip_pattern.findall(network_route)

      # Remove localhost
      self.ips = [ip for ip in self.ips if not ip.startswith("127.")]
    else:
      self.ips = []

  @property
  def ip(self):
    # Lazily parse IPs in email only when/if the property is used
    if self.ips == []:
      self.parse_ips()
    return self.ips


  @property
  def attachments(self) -> list[str]:
    # Just the filenames
    return [attachment.get_filename() for attachment in self.msg.iter_attachments()]

  @property
  def headers(self) -> list[str]:
    # Return the currently declared headers keys (not values)
    return self.msg.keys()


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

  def is_in(self, query_list, field:str, case_sensitive:bool=False, mode:str="any"):
    # Check if any or all of the elements in the query_list is in the email field
    # The field can by any RFC header or "Body".
    value = None
    if field == "Body":
      value = self.get_body() if case_sensitive else self.get_body().lower()
    elif field in self.headers:
      value = self[field] if case_sensitive else self[field].lower()

    if not value:
      # Value is empty or None -> abort early
      return False

    if not isinstance(query_list, list):
      # If a single query element is given, make it a list for uniform handling in the following
      query_list = [query_list]

    if mode == "any":
      return any(query_item in value for query_item in query_list)
    elif mode == "all":
      return all(query_item in value for query_item in query_list)
    else:
      raise ValueError("Non-implementad mode `%s` used" % mode)


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
      self.server.objects.remove(self)

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

  def create_folder(self, folder:str):
    # create the folder, recursively if needed (create parent then children)
    target = ""
    for level in folder.split('.'):
      target += level

      if target not in self.server.folders:
        print(target)
        result = self.server.create(target)

        if result[0] == "OK":
          print("Folder `%s` created\n" % target)
          self.server.logfile.write("%s : Folder `%s` created\n" % (utils.now(), target))
          self.server.subscribe(target)
        else:
          print("Failed to create folder `%s`\n" % target)
          self.server.logfile.write("%s : Failed to create folder `%s`\n" % (utils.now(), target))

      target += "."

    # Update the list of server folders
    self.server.get_imap_folders()


  def move(self, folder:str):
    self.create_folder(folder)
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
        self.server.objects.remove(self)

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

  def is_unread(self) -> bool:
    # Email has not been opened and read
    return "\\Seen" not in self.flags

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

  def age(self):
    # Compute the age of an email at the time of evaluration
    sending_date = email.utils.parsedate_to_datetime(self["Date"])
    current_date = datetime.now(timezone.utc)
    delta = (current_date - sending_date)
    return delta


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
    try:
      date_time = email.utils.parsedate_to_datetime(self["Date"])
      timestamp = str(int(datetime.timestamp(date_time)))
    except:
      # Some poorly-encoded emails (spams) are not properly decoded by the email package
      # and the date gets embedded in subject
      timestamp = "0"

    # Hash the Received header, which is the most unique
    # field as it contains the server route of the email along with
    # all the times.
    # If no received header (for emails sent from the current mailbox),
    # find other fields supposed to be unique.
    hashable = ""
    if "Received" in self.headers:
      hashable = self["Received"]
    elif "Message-ID" in self.headers:
      hashable = self["Message-ID"]
    else:
      # Last chance for malformed emails. "Message-ID" is technically not mandatory,
      # but it has become de-facto standard to thread emails.
      hashable = self["Date"] + self["From"] + self["To"]

    hash = hashlib.md5(hashable.encode())

    # Our final hash is the Unix timestamp for easy finding and Received hash
    self.hash = timestamp + "-" + hash.hexdigest()


  def query_referenced_emails(self) -> list:
    # Fetch the list of all emails referenced in the present message,
    # aka the whole email thread in wich the current email belongs.
    # The list is sorted from newest to oldest.
    thread = []
    if "References" in self.headers:
      for id in self["References"].split(" "):
        # Query all emails having Message-ID header matching an item in references
        data = [b'']
        i = 0

        while not data[0] and i < len(self.server.folders):
          # Iterate over all mailboxes until we find the email
          self.server.select(self.server.folders[i])
          typ, data = self.server.uid('SEARCH', '(HEADER Message-ID "%s")' % id)
          i += 1

        if data[0]:
          # We found something.
          # data[0] usually contains one single UID but it may happen in broken mailboxes that we have several
          for uid in data[0].decode().split(" "):
            message = self.server.get_email(uid, mailbox=self.server.folders[i - 1])

            # Double-check that we have the right email
            if message.uid != uid or message["Message-ID"] != id:
              raise Exception("The email fetched is not the one requested")
            else:
              thread.append(message)

      # Select the original mailbox again for the next operations
      self.server.select(self.server.mailbox)

    # Output the list of emails from most recent to most ancient
    thread.reverse()
    return thread


  def query_replied_email(self):
    # Fetch the email being replied to by the current email.
    replied = None

    if "In-Reply-To" in self.headers:
      id = self["In-Reply-To"]
      data = [b'']
      i = 0

      while not data[0] and i < len(self.server.folders):
        # Iterate over all mailboxes until we find the email
        self.server.select(self.server.folders[i])
        typ, data = self.server.uid('SEARCH', '(HEADER Message-ID "%s")' % id)
        i += 1

      # Select the original mailbox again for the next operations
      self.server.select(self.server.mailbox)

      if data[0]:
        print(id, data)
        # We found something.
        # data[0] usually contains one single UID but it may happen that we have more
        uids = data[0].decode().split(" ")

        # returns only the most recent email if several
        replied = self.server.get_email(uids[-1], mailbox=self.server.folders[i - 1])

        # Double-check that we have the right email
        if replied.uid != uids[-1] or replied["Message-ID"] != id:
          raise Exception("The email fetched is not the one requested")

    return replied


  def __init__(self, raw_message:list, server) -> None:
    # Position of the email in the server list
    super().__init__(raw_message, server)

    self.urls = []
    self.ips = []

    # Raw message as fetched by IMAP, decode IMAP headers
    try:
      email_content = raw_message[0].decode()
      self.parse_uid(email_content)
      self.parse_flags(email_content)
    except:
      self.flags = ""
      self.uid = ""
      print("Decoding headers failed for : %s" % raw_message[0])

    # Decode RFC822 email body
    self.msg = email.message_from_bytes(raw_message[1], policy=email.policy.default)

    try:
      self.create_hash()
    except:
      print("Can't hash the email", self["Subject"], "on", self["Date"], "from", self["From"], "to", self["To"])
