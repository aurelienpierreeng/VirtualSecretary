import imaplib
import utils
import email
import re
import pickle
import os
import hashlib
import html
import time

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

class EMail(object):

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
    result = self.mailserver.uid('STORE', self.uid, '+FLAGS', keyword)

    if result[0] == "OK":
      self.mailserver.logfile.write("%s : Tag `%s` added to email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    keyword,
                                                                                    self.uid,
                                                                                    self.header["Subject"],
                                                                                    self.header["From"],
                                                                                    self.header["Date"]))

    self.mailserver.std_out = result

  def untag(self, keyword):
    # Remove a tag.
    result = self.mailserver.uid('STORE', self.uid, '-FLAGS', keyword)

    if result[0] == "OK":
      self.mailserver.logfile.write("%s : Tag `%s` removed from email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    keyword,
                                                                                    self.uid,
                                                                                    self.header["Subject"],
                                                                                    self.header["From"],
                                                                                    self.header["Date"]))

    self.mailserver.std_out = result

  def delete(self):
    # Delete an email directly without using the trash bin.
    # Use a move to trash folder to get a last chance at reviewing what will be deleted.
    result = self.mailserver.uid('STORE', self.uid, '+FLAGS', "(\\Deleted)")

    if result[0] == "OK":
      self.mailserver.logfile.write("%s : Deteled email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    self.uid,
                                                                                    self.header["Subject"],
                                                                                    self.header["From"],
                                                                                    self.header["Date"]))

      # delete the email from the list of emails in the MailServer object to avoid further manipulation
      self.mailserver.emails.remove(self)

    self.mailserver.std_out = result

  def spam(self, spam_folder="INBOX.spam"):
    # Mark an email as spam using Thunderbird tags and move it to the spam/junk folder
    self.mailserver.uid('STORE', self.uid, '-FLAGS', 'NonJunk')
    self.mailserver.uid('STORE', self.uid, '-FLAGS', '(\\Seen)')

    result = self.mailserver.uid('COPY', self.uid, spam_folder)

    if result[0] == "OK":
      result = self.mailserver.uid('STORE', self.uid, '+FLAGS', '(\\Deleted)')
      if result[0] == "OK":
        self.mailserver.logfile.write("%s : Spam email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                          self.uid,
                                                                                          self.header["Subject"],
                                                                                          self.header["From"],
                                                                                          self.header["Date"]))

    self.mailserver.std_out = result

  def move(self, folder:str):
    # create the folder and update the list of folders
    if folder not in self.mailserver.folders:
      result = self.mailserver.create(folder)

    result = self.mailserver.subscribe(folder)

    if result[0] == "OK":
      self.mailserver.logfile.write("%s : Folder`%s` created in INBOX\n" % (utils.now(), folder))

    self.mailserver.get_imap_folders()

    result = self.mailserver.uid('COPY', self.uid, folder)

    if result[0] == "OK":
      result = self.mailserver.uid('STORE', self.uid, '+FLAGS', '(\\Deleted)')
      if result[0] == "OK":
        self.mailserver.logfile.write("%s : Moved email (UID %s) `%s` from `%s` received on %s to %s\n" % (utils.now(),
                                                                                                          self.uid,
                                                                                                          self.header["Subject"],
                                                                                                          self.header["From"],
                                                                                                          self.header["Date"],
                                                                                                          folder))
        # delete the email from the list of emails in the MailServer object to avoid further manipulation
        # because the emails list is tied to a particular mailbox folder.
        self.mailserver.emails.remove(self)

    self.mailserver.std_out = result


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
    self.mailserver.std_out = ["OK", ]

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

  def __init__(self, raw_message:list, index:int, mailserver) -> None:
    # Position of the email in the mailserver list
    self.index = index
    self.mailserver = mailserver

    # Raw message as fetched by IMAP
    email_content = raw_message[0][0].decode()
    self.parse_uid(email_content)
    self.parse_flags(email_content)

    # Decoded message
    self.msg = email.message_from_bytes(raw_message[0][1])

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


class MailServer(imaplib.IMAP4_SSL):
  def get_imap_folders(self):
    # List inbox subfolders as plain text, to be reused by filter definitions

    mail_list = self.list()
    self.folders = []

    if(mail_list[0] == "OK"):
      for elem in mail_list[1]:
        entry = elem.decode().split('"."')
        flags = entry[0].strip("' ")
        folder = entry[1].strip("' ")

        if "\\Archive" in flags:
          self.archive = folder
        if "\\Sent" in flags:
          self.sent = folder
        if "\\Trash" in flags:
          self.trash = folder
        if "\\Junk" in flags:
          self.junk = folder

        self.folders.append(folder)

      self.logfile.write("%s : Found %i inbox subfolders : %s\n" % (utils.now(), len(self.folders), ", ".join(self.folders)))
    else:
      self.logfile.write("%s : Impossible to get the list of inbox folders\n" % (utils.now()))

    self.std_out = mail_list

  def get_mailbox_emails(self, mailbox:str, n_messages=-1):
    # List the n-th last emails in mailbox
    if not self.connection_inited:
      print("We do not have an active IMAP connection. Ensure your IMAP credentials are defined in `settings.ini`")
      return

    # If no explicit number of messages is passed, use the one from `settings.ini`
    if n_messages == -1:
      n_messages = self.n_messages

    if mailbox in self.folders:
      status, messages = self.select(mailbox)
      num_messages = int(messages[0])
      self.logfile.write("%s : Reached mailbox %s : %i emails found, loading only the first %i\n" % (utils.now(), mailbox, num_messages, n_messages))

      self.emails = []
      messages_queue = []

      # Network loop
      ts = time.time()
      for i in range(max(num_messages - n_messages + 1, 1), num_messages + 1):
        try:
          res, msg = self.fetch(str(i), "(FLAGS RFC822 UID)")
          messages_queue.append(msg)
        except:
          print("Could not get email %i, it may have been deleted on server by another application in the meantime." % i)
      print("  - IMAP\ttook %.3f s\tto query\t%i emails from %s" % (time.time() - ts, len(messages_queue), mailbox))

      # Process loop
      ts = time.time()
      for msg in messages_queue:
        self.emails.append(EMail(msg, i, self))
      print("  - Parsing\ttook %.3f s\tto parse\t%i emails" % (time.time() - ts, len(self.emails)))

      # TODO: split process and network loops in 2 threads ?

      self.std_out = [status, messages]

    else:
      self.logfile.write("%s : Impossible to get the mailbox %s : no such folder on server\n" % (utils.now(), mailbox))


  def __init_log_dict(self, log:dict):
    # Make sure the top-level dict keys are inited
    # for our current server/username pair
    if self.server not in log:
      log[self.server] = { self.user : { } }

    if self.user not in log[self.server]:
      log[self.server][self.user] = { }


  def __update_log_dict(self, email:EMail, log:dict, field:str, enable_logging:bool, action_run=True):
    if(enable_logging and action_run):
      try:
        # Update existing log entry for the current uid
        log[self.server][self.user][email.hash][field] += 1
      except:
        # Create a new log entry for the current uid
        log[self.server][self.user][email.hash] = { field : 1 }

    if(not action_run):
      try:
        # Update existing log entry for the current uid
        log[self.server][self.user][email.hash][field] += 0
      except:
        # Create a new log entry for the current uid
        log[self.server][self.user][email.hash] = { field : 0 }



  def run_filters(self, filter, action, runs=1):
    # Run the function `filter` and execute the function `action` if the filtering condition is met
    # * `filter` needs to return a boolean encoding the success of the filter.
    # * `action` needs to return a list where the [0] element contains "OK" if the operation succeeded,
    #    and the [1] element is user-defined stuff.
    # * `filter` and `action` take an `mailserver.EMail` instance as input
    # * `runs` defines how many times the emails need to be processed. -1 means no limit.
    # * `filter_file` is the full path to the filter Python script

    # Define the log file as an hidden file inheriting the filter filename
    directory = os.path.dirname(self.secretary.filtername)
    basename = os.path.basename(self.secretary.filtername)
    filter_logfile = os.path.join(directory, "." + basename + ".log")

    # Init a brand-new log dict
    log = { self.server : # server
            { self.user : # username
              {
                # email.hash : { PROPERTIES }
              }
            }
          }

    # Enable logging and runs limitations if required
    enable_logging = (runs != -1) and isinstance(runs, int)

    # Open the logfile if any
    if enable_logging and os.path.exists(filter_logfile):
      with open(filter_logfile, "rb") as f:
        log = dict(pickle.load(f))
        #print("DB found : %s" % f)
        #print(log)
    else:
      #print("%s not found" % log)
      pass

    self.__init_log_dict(log)

    ts = time.time()

    for email in self.emails:
      # Disable the filters if the number of allowed runs is already exceeded
      if enable_logging and email.hash in log[self.server][self.user]:
        # We have a log entry for this hash.
        #print("%s found in DB" % email.hash)
        filter_on = (log[self.server][self.user][email.hash]["processed"] < runs)
      else:
        # We don't have a log entry for this hash or we don't limit runs
        #print("%s not found in DB" % email.hash)
        filter_on = True

      # Run the actual filter
      if filter_on and filter:
        try:
          # User wrote good filters.
          self.std_out = [ "", ]
          filter_on = filter(email)
          #print(filter_on)

          if not isinstance(filter_on, bool):
            filter_on = False
            print("The filter does not return a boolean, the behaviour is ambiguous. Filtering is canceled.")
            raise TypeError("The filter does not return a boolean, the behaviour is ambiguous. Filtering is canceled.")
        except:
          # User tried to filter non-existing fields or messed-up somewhere.
          filter_on = False

      # Run the action
      if filter_on:
        try:
          # The action should update self.std_out internally. If not, init here as a success.
          # Success and errors matter only for email write operations
          self.std_out = [ "", ]
          action(email)

          if self.std_out[0] == "OK":
            # Log success
            print("Filter application successful on", email.header["Subject"], "from", email.header["From"])
            self.__update_log_dict(email, log, "processed", enable_logging)
          else:
            # Log error
            print("Filter application failed on", email.header["Subject"], "from", email.header["From"])
            self.__update_log_dict(email, log, "errored", enable_logging)

        except:
          # Log error
          print("Filter application failed on", email.header["Subject"], "from", email.header["From"])
          self.__update_log_dict(email, log, "errored", enable_logging)
      else:
        # No action run but we still store email ID in DB
        self.__update_log_dict(email, log, "processed", enable_logging, action_run=False)

    # Actually delete with IMAP the emails marked with the tag `\DELETED`
    # We only need to run it once per email loop/mailbox.
    self.expunge()

    # Close the current mailbox.
    # We need it here in case a filter starts more than one loop over different mailboxes
    self.close()

    print("  - Filtering\ttook %.3f s\tto filter\t%i emails" % (time.time() - ts, len(self.emails)))


    # Dump the log dict to a byte file for efficiency
    with open(filter_logfile, "wb") as f:
      pickle.dump(log, f)
      #print(log)
      #print("%s written" % filter_logfile)


  def init_connection(self, params:dict):
    # High-level method to login to a server in one shot
    self.n_messages = int(params["entries"])
    self.server = params["server"]
    self.user = params["user"]

    # Init the SSL connection to the server
    self.logfile.write("%s : Trying to login to %s with username %s\n" % (utils.now(), self.server, self.user))
    try:
      imaplib.IMAP4_SSL.__init__(self, host=self.server)
    except:
      print("We can't reach the server %s. Check your network connection." % self.server)

    try:
      self.std_out = self.login(self.user, params["password"])
      self.logfile.write("%s : Connection to %s : %s\n" % (utils.now(), self.server, self.std_out[0]))
    except:
      print("We can't login to %s with username %s. Check your credentials" % (self.server, self.user))

    self.get_imap_folders()

    self.connection_inited = True


  def close_connection(self):
    # High-level method to logout from a server
    self.logout()
    self.connection_inited = False


  def __init__(self, logfile, secretary) -> None:
    # This is the global logfile, `sync.log`
    # Not to be confused with the local filter file.
    self.logfile = logfile

    # Init an output pipe to globally fetch IMAP commands output
    # Each internal IMAP method will post its output on it, so
    # users don't have to return the out code in the filters
    self.std_out = [ ]

    # Store a link to the secretary connectors
    self.secretary = secretary

    self.connection_inited = False
