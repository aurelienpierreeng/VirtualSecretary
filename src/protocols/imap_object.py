import imaplib
import utils
import email
import re
import hashlib
import html
import connectors
import spf
import dkim
from dns import resolver

from email import policy

from datetime import datetime, timedelta, timezone

# IPv4 and IPv6
ip_pattern = re.compile(r"from.*?((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4}))", re.IGNORECASE)
domain_pattern = re.compile(r"from ((?:[A-Za-z0-9\-_]{0,61}\.)+[a-z]{2,})", re.IGNORECASE)
email_pattern = re.compile(r"<?([0-9a-zA-Z\-\_\+\.]+?@[0-9a-zA-Z\-\_\+]+(\.[0-9a-zA-Z\_\-]{2,})+)>?", re.IGNORECASE)
url_pattern = re.compile(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)", re.IGNORECASE)
uid_pattern = re.compile(r"UID ([0-9]+)")
flags_pattern = re.compile(r"FLAGS \((.*?)\)")


class EMail(connectors.Content):

  ##
  ## Data parsing / decoding
  ##

  def parse_uid(self, raw):
    self.uid = uid_pattern.search(raw).groups()[0]

  def parse_flags(self, raw):
    self.flags = flags_pattern.search(raw).groups()[0]

  def parse_ips(self):
    # Get the IPs of the whole network route taken by the email
    if self.has_header("Received"):
      # There will be no "Received" header for emails sent by the current mailbox
      # Exclude the most recent "Received" header because it will be our server.
      network_route = self.msg.get_all("received")[1:-1]
      network_route = "\n".join(network_route)
      self.ips = ip_pattern.findall(network_route)

      # Remove localhost and local network IPs
      self.ips = [ip for ip in self.ips if not (ip.startswith("127.") or ip.startswith("fe80::") or ip.startswith("192."))]
    else:
      self.ips = []

  @property
  def ip(self):
    # Lazily parse IPs in email only when/if the property is used
    if self.ips == [] or not self.ips:
      self.parse_ips()
    return self.ips

  def parse_domains(self):
    # Get the domains of the whole network route taken by the email
    if self.has_header("Received"):
      # There will be no "Received" header for emails sent by the current mailbox
      # Exclude the most recent "Received" header because it will be our server.
      network_route = self.msg.get_all("received")[1:-1]
      network_route = "\n".join(network_route)
      self.domains = domain_pattern.findall(network_route)

       # Remove localhost
      self.domains = [ domain for domain in self.domains if "localhost" not in domain ]
    else:
      self.domains = []

  @property
  def domain(self):
    # Lazily parse IPs in email only when/if the property is used
    if self.domains == [] or not self.domains:
      self.parse_domains()
    return self.domains

  @property
  def attachments(self) -> list[str]:
    # Just the filenames
    return [attachment.get_filename() for attachment in self.msg.iter_attachments()]

  @property
  def headers(self) -> list[str]:
    # Return the currently declared headers keys (not values) in lowercase.
    # That's noticeably important for headers like "Message-ID" which can be aliased "Message-Id"
    return [key.lower() for key in self.msg.keys()]

  def has_header(self, header: str) -> bool:
    check = header.lower() in self.headers
    return check

  def get_sender(self) -> list[list, list]:
    emails = email_pattern.findall(self["From"])
    names = re.findall(r"\"(.+)?\"", self["From"])
    out = [names, [email[0] for email in emails if email[0]]]
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


  def __getitem__(self, key: str):
    # Getting key from the class is dispatched directly to email.EmailMessage properties
    return self.msg.get(key.lower())

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
    if field.lower() == "body":
      value = self.get_body() if case_sensitive else self.get_body().lower()
    elif self.has_header(field):
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

    result = self.server.uid('COPY', self.uid, self.server.encode_imap_folder(spam_folder))

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
    self.server.create_folder(folder)
    result = self.server.uid('COPY', self.uid, self.server.encode_imap_folder(folder))

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

  def is_mailing_list(self) -> bool:
    # This email has the typical mailing-list tags. Warning: this is not standard and not systematically used.
    # Mailing list are sent by humans to a group of other humans.
    has_list_unsubscribe = self.has_header("List-Unsubscribe") # note that we don't check if it's a valid unsubscribe link
    has_precedence_list = self.has_header("Precedence") and self["Precedence"] == "list"
    return has_list_unsubscribe and has_precedence_list

  def is_newsletter(self) -> bool:
    # This email has the typical newsletter tags. Warning: this is not standard and not systematically used.
    # Newsletters are sent by bots to a group of humans.
    has_list_id = self.has_header("List-ID")
    has_precedence_bulk = self.has_header("Precedence") and self["Precedence"] == "bulk"
    has_feedback_id = self.has_header("Feedback-ID") or self.has_header("X-Feedback-ID")
    has_csa_complaints = self.has_header("X-CSA-Complaints")
    has_list_unsubscribe = self.has_header("List-Unsubscribe") # note that we don't check if it's a valid unsubscribe link
    return has_list_unsubscribe and (has_list_id or has_precedence_bulk or has_feedback_id or has_csa_complaints)

  def has_tag(self, tag:str) -> bool:
    return tag in self.flags

  ##
  ## Authenticity checks
  ##

  def spf_pass(self) -> int:
    # Check if any of the servers listed by IP in the "Received" header is authorized
    # by the mail server to send emails on behalf of the email address used as "From".
    # See https://www.rfc-editor.org/rfc/rfc7208
    # Output a reputation score :
    scores = { "none":    0,    # no SPF records were retrieved from the DNS.
               "neutral": 1,    # the ADMD has explicitly stated that it is not asserting whether the IP address is authorized.
               "pass":    2,    # explicit statement that the client is authorized to inject mail with the given identity.
               "temperror":  0, # the SPF verifier encountered a transient (generally DNS) error while performing the check.
               "permerror": -1, # the domain's published records could not be correctly interpreted.
               "softfail":  -1, # weak statement by the publishing ADMD that the host is probably not authorized.
               "fail": -2       # explicit statement that the client is not authorized to use the domain in the given identity.
              }

    names, addresses = self.get_sender()

    # Convert all DNS domains from the email route to IPs
    # Useful for emails sent on the local network where no IP is written in headers
    ips = []
    for domain in self.domain:
      try:
        for x in resolver.resolve(domain, 'A'):
          ips.append(x.to_text())
      except:
        pass

    # Append the list of IPs mentionned explicitely in headers
    # Reverse it so the bottom "Received" are treated first since they are the most probably
    # linked to SPF records and each entry starts a network request.
    ips += self.ip
    ips.reverse()

    # Check all addresses for all IPs in the email route until we find a match.
    # We consider it a fail only on explicite fail, e.g. the SPF record explicitely prohibits
    # all found IPs to send emails on behalf of the address used.
    # Non-existing or wrongly-set SPF records are treated as a success.
    spf_score = -2
    for email in addresses:
      email_domain = email.split("@")[1]
      for ip in set(ips):
        try:
          for x in resolver.resolve(email_domain, 'MX'):
            spf_status = spf.check2(i=ip, s=email, h=x.to_text())[0]

            # Record the highest reputation score of the list of IPs
            score = scores[spf_status]
            spf_score = score if score > spf_score else spf_score

            # If we got a success, abort immediately
            if spf_score == 2:
              return spf_score
        except:
          pass

    return spf_score

  def dkim_pass(self):
    # Return a reputation score :
    #  0 if no DKIM signature
    #  1 if the DKIM signature is valid
    # -1 if the DKIM signature is invalid. That's because many spammers
    # forge a fake Google DKIM signature hoping to past by the spam filters
    # that only check for the header presence without actually validating it.

    if self.has_header("DKIM-Signature"):
      dkim_score = -1
      dk = dkim.DKIM(message=self.raw)
      signatures = self.msg.get_all("DKIM-Signature")
      for i in range(len(signatures)):
        # Sometimes there are several DKIM Signature when the message
        # transits through several servers. We need to check them all.
        try:
          output = dk.verify(i)
        except:
          # Invalid encoding or something happened
          pass
        else:
          if output and dkim_score < 1:
            # Valid DKIM signature - Abort now
            dkim_score = 1
            return dkim_score
    else:
      dkim_score = 0

    return dkim_score

  def authenticity_score(self) -> int:
    # Returns :
    # == 0 : neutral, no explicit authentification is defined on DNS or no rule could be found
    #  > 0 : expliticitely authenticated
    # == 3 : maximal authenticity (valid SPF and valid DKIM)
    #  < 0 : spoofed, either or both SPF and DKIM explicitely failed
    spf_score = int(self.spf_pass())
    dkim_score = int(self.dkim_pass())
    total = spf_score + dkim_score
    print(self["From"], spf_score, "+", dkim_score, "=", total)
    return total

  def is_authentic(self) -> bool:
    # Check SPF and DKIM to validate that the email is authentic,
    # aka not spoofed. That's enough to detect most spams.
    return self.authenticity_score() >= 0

  ##
  ## Utils
  ##

  def age(self):
    # Compute the age of an email at the time of evaluation
    current_date = datetime.now(timezone.utc)
    delta = (current_date - self.date)
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
      timestamp = str(int(datetime.timestamp(self.date)))
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
    if self.has_header("Received"):
      hashable = self["Received"]
    elif self.has_header("Message-ID"):
      hashable = self["Message-ID"]
    elif self.has_header("From") and self.has_header("To"):
      # Last chance for malformed emails. "Message-ID" is technically not mandatory,
      # but it has become de-facto standard to thread emails.
      hashable = self.date.strftime("%d/%m/%Y %H:%M:%S") + self["From"] + self["To"]
    else:
      hashable = self.get_body("plain")

    hash = ""

    try:
      hash = hashlib.md5(hashable.encode()).hexdigest()
    except:
      print("Can't hash the email", self["Subject"], "on", self["Date"], "from", self["From"], "to", self["To"])

    # Our final hash is the Unix timestamp for easy finding and Received hash
    self.hash = timestamp + "-" + hash


  def query_referenced_emails(self) -> list:
    # Fetch the list of all emails referenced in the present message,
    # aka the whole email thread in wich the current email belongs.
    # The list is sorted from newest to oldest.
    thread = []
    if self.has_header("References"):
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

    if self.has_header("In-Reply-To"):
      id = self["In-Reply-To"]
      data = [b'']
      i = 0

      while not data[0] and i < len(self.server.folders):
        # Iterate over all mailboxes until we find the email
        self.server.select(self.server.encode_imap_folder(self.server.folders[i]))
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

  def get_date(self):
    # Extract some temporal info. Problem is bad emails don't have a proper Date header,
    # or it is badly encoded, so we need to deal with that.

    self.date = None

    if self.has_header("Date") and self["Date"]:
      try:
        # Try the date of sending
        self.date = email.utils.parsedate_to_datetime(self["Date"])
      except Exception as e:
        print(e)

        # If we have a date header but the parsing failed, try again to parse
        # usual-yet-non-standard formats.
        date_eu = re.match(r"([0-9]{2})\-([0-9]{2})\-([0-9]{4})", self["Date"])
        date_us = re.match(r"([0-9]{4})\-([0-9]{2})\-([0-9]{2})", self["Date"])

        if date_eu:
          month = int(date_eu.groups()[0])
          day = int(date_eu.groups()[1])
          year = int(date_eu.groups()[2])
          self.date = datetime(year, month, day, tzinfo=timezone.utc)
        elif date_us:
          day = int(date_us.groups()[2])
          month = int(date_us.groups()[1])
          year = int(date_us.groups()[0])
          self.date = datetime(year, month, day, tzinfo=timezone.utc)

    if not self.date and self.has_header("Delivery-date") and self["Delivery-date"]:
        # If we don't find one, use the incoming date. This one should be put by our server.
        try:
          self.date = email.utils.parsedate_to_datetime(self["Delivery-date"])
        except Exception as e:
          print(e)

    if not self.date:
      # If everything else failed, set 1970.
      self.date = datetime.fromtimestamp(0, timezone.utc)


  def __init__(self, raw_message:list, server) -> None:
    # Position of the email in the server list
    super().__init__(raw_message, server)

    self.urls = []
    self.ips = []
    self.domains = []

    # Raw message as fetched by IMAP, decode IMAP headers
    try:
      email_content = raw_message[0].decode()
    except Exception as e:
      print("Decoding headers failed for : %s" % raw_message[0])
      print(e)

    if email_content:
      self.parse_uid(email_content)
      self.parse_flags(email_content)

    # Decode RFC822 email body
    # No exception handling here, let it fail. Email validity should be checked at server level
    self.msg = email.message_from_bytes(raw_message[1], policy=policy.default)
    self.raw = raw_message[1]

    # Get "a" date for the email
    self.get_date()

    # The hash uses the date defined above, so we need to create it after
    self.create_hash()
