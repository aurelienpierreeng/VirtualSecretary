from __future__ import annotations
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
from email.utils import parseaddr

from datetime import datetime, timedelta, timezone

# IPv4 and IPv6
ip_pattern = re.compile(r"from.*?((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4}))", re.IGNORECASE)
domain_pattern = re.compile(r"from ((?:[a-z0-9\-_]{0,61}\.)+[a-z]{2,})", re.IGNORECASE)
email_pattern = re.compile(r"<?([0-9a-zA-Z\-\_\+\.]+?@[0-9a-zA-Z\-\_\+]+(\.[0-9a-zA-Z\_\-]{2,})+)>?", re.IGNORECASE)
url_pattern = re.compile(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)", re.IGNORECASE)
uid_pattern = re.compile(r"UID ([0-9]+)")
flags_pattern = re.compile(r"FLAGS \((.*?)\)")


class EMail(connectors.Content):

  ##
  ## Data parsing / decoding
  ##

  def parse_uid(self, raw):
    self.uid =  uid_pattern.search(raw).groups()[0]

  def parse_flags(self, raw):
    self.flags = flags_pattern.search(raw).groups()[0]

  def __get_domain(self, string: str) -> str:
    # Find the DNS domain in the form `something.com`
    domain_pattern = r"^((?:[a-z0-9\-_]{0,61}\.)+[a-z]{2,})"
    result = re.findall(domain_pattern, string, re.IGNORECASE)

    if len(result) == 0:
      # Nothing found.
      if "localhost" in string:
        # Check if it's localhost
        result = ["localhost"]
      else:
        # Maybe the domain is referenced directly by IPv4 or IPv6
        ip_pattern = r"^\[((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4})).*?\]"
        result = re.findall(ip_pattern, string, re.IGNORECASE)

        # Detect local IPs and rename them localhost
        result = [r if (not (r.startswith("127.") or r.startswith("192.") or r.startswith("fe80::"))) else "localhost" for r in result]

    return result

  def __get_ip(self, string: str) -> str:
    ip_pattern = r"\(.*?\[((?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4}))\].*?\)"
    result = re.findall(ip_pattern, string, re.IGNORECASE)

    # Remove local IPs from results
    return [r for r in result if not (r.startswith("127.") or r.startswith("192.") or r.startswith("fe80::"))]

  def __get_envelope(self, string: str) ->str:
    envelope_pattern = r"\(.*?envelope-from <(.+?@.+?)>\)"
    result = re.findall(envelope_pattern, string, re.IGNORECASE)
    return result

  def __sanitize_domains(self, route: dict) -> dict:
    ip_pattern = r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:fe80::)?(?:[0-9a-fA-F]{1,4}:){3}[0-9a-fA-F]{1,4})"

    if len(route["from"]["domain"]) > 0 and len(route["from"]["ip"]) == 0:
      if re.match(ip_pattern, route["from"]["domain"][0]):
        # The domain is actually declared by its IP, we need to switch fields
        route["from"]["ip"] = route["from"]["domain"]
        route["from"]["domain"] = []

    return route

  def parse_network_route(self):
    # Get the IPs of the whole network route taken by the email
    self.route = []

    if self.has_header("received"):
      # There will be no "Received" header for emails sent by the current mailbox
      # Exclude the most recent "Received" header because it will be our server.
      network_route = self.msg.get_all("received")

      for step in network_route:
        # Clean up space characters
        step = step.replace("\t", "\n")

        # Split sender, receipient and options
        parts = {
          "for": "".join(re.findall(r"^for (.+?);", step, re.MULTILINE)),
          "from": "".join(re.findall(r"^from (.+?)\s(?:for|with|by)", step, re.MULTILINE)),
          "by": "".join(re.findall(r"^by (.+?)\s(?:from|with|for)", step, re.MULTILINE)),
        }

        step_dict = {
          "from": {
            "ip": self.__get_ip(parts["from"]),
            "domain":  self.__get_domain(parts["from"])
            },
          "by": {
            "ip": self.__get_ip(parts["by"]),
            "domain": self.__get_domain(parts["by"])
            },
          "envelope-from" : self.__get_envelope(step)
        }

        step_dict = self.__sanitize_domains(step_dict)
        self.route.append(step_dict)

      self.route.reverse()

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
    """Check if the case-insensitive `header` exists in the email headers.

    Arguments:
      header (str): the RFC 822 email header.

    Returns:
      (bool): presence of the header
    """
    return header.lower() in self.headers

  def get_sender(self) -> list[list, list]:
    """Get the full list of senders of the email, using the `From` header, splitting their name (if any) apart from their address.

    Returns:
      (list[list, list]): `list[0]` contains the list of names, rarely used, `list[1]` is the list of email addresses.
    """
    emails = email_pattern.findall(self["From"])
    names = re.findall(r"\"(.+)?\"", self["From"])
    return [names, [email[0] for email in emails if email[0]]]


  def parse_urls(self, input:str) -> list[tuple]:
    """Update `self.urls` with a list of all URLs found in `input`, split as `(domain, page)` tuples.

    Examples:
      Each result in the list is a tuple (domain, page), for example :

      - `google.com/index.php` is broken into `('google.com', '/index.php')`
      - `google.com/` is broken into `('google.com', '/')`
      - `google.com/login.php?id=xxx` is broken into `('google.com', '/login.php')`

    """
    try:
      self.urls.append(url_pattern.findall(input))
    except:
      print("Could not parse urls in email body :")
      print(input)

    # Flatten the list 2D list
    self.urls = [item for sublist in self.urls for item in sublist]


  def __getitem__(self, key: str):
    # Getting key from the class is dispatched directly to email.EmailMessage properties
    # Return empty string instead of None object for direct concatenation
    value = self.msg.get(key.lower())
    return value if value else ""

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

  def get_body(self, preferencelist=('related', 'html', 'plain')) -> str:
    """Get the body of the email.

    Arguments:
      preferencelist (tuple | str): sequence of candidate properties in which to pick the email body, by order of priority. If set to `"plain"`, return either the plain-text variant of the email if any, or build one by removing (x)HTML markup from the HTML variant if no plain-text variant is available.

    Note:
      Emails using `quoted-printable` transfer encoding but not UTF-8 charset are not handled. This weird combination has been met only in spam messages written in Russian, so far, and should not affect legit emails.
    """
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
        return ""

    return content

  def is_in(self, query_list: list[str] | str, field:str, case_sensitive:bool=False, mode:str="any") -> bool:
    """Check if any or all of the elements in the `query_list` is in the email `field`.

    Arguments:
      query_list (list[str] | str): list of keywords or unique keyword to find in `field`.
      field (str): any RFC 822 header or `"body"`.
      case_sensitive (str): `True` if the search should be case-sensitive. This has no effect if `field` is a RFC 822 header, it only applies to the email body.
      mode (str): `"any"` if any element in `query_list` should be found in `field` to return `True`. `"all"` if all elements in `query_list` should be found in `field` to return `True`.

    Returns:
      (bool): `True` if any or all elements (depending on `mode`) of `query_list` have been found in `field`.
    """
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
      raise ValueError("Non-implemented mode `%s` used" % mode)


  ##
  ## IMAP ACTIONS
  ##

  def tag(self, keyword:str):
    """Add any arbitrary IMAP tag (aka label), standard or not, to the current email.

    Warning:
      In Mozilla Thunderbird, labels/tags need to be configured first in the preferences (by mapping the label string to a color) to properly appear in the GUI. Otherwise, any undefined tag will be identified as "Important" (associated with red), no matter its actual string.

      Horde, Roundcube and Nextcloud mail (based on Horde) treat those properly.
    """
    result = self.server.uid('STORE', self.uid, '+FLAGS', keyword)

    if result[0] == "OK":
      self.server.logfile.write("%s : Tag `%s` added to email (UID %s) `%s` from `%s` received on %s\n" % (utils.now(),
                                                                                    keyword,
                                                                                    self.uid,
                                                                                    self["Subject"],
                                                                                    self["From"],
                                                                                    self["Date"]))

    self.server.std_out = result

  def untag(self, keyword: str):
    """Remove any arbitrary IMAP tag (aka label), standard or not, to the current email."""
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
    """Delete the current email directly without using the trash bin. It will not be recoverable.

    Use [EMail.move][protocols.imap_object.EMail.move] to move the email to the trash folder to get a last chance at reviewing what will be deleted.

    Note:
      As per IMAP standard, this only add the `\\Deleted` flag to the current email. Emails will be actually deleted when the `expunge` server command is launched, which is done automatically at the end of [Server.run_filters][protocols.imap_server.Server.run_filters].
    """
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
    """Mark the current email as spam, adding Mozilla Thunderbird `Junk` flag, and move it to the spam/junk folder."""
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
    """Move the current email to the target `folder`, that will be created recursively if it does not exist. `folder` will be internally encoded to IMAP-custom UTF-7 with [Server.encode_imap_folder][protocols.imap_server.Server.encode_imap_folder].
    """
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
    """Flag or unflag an email as important

    Arguments:
      mode (str): `add` to add the `\\Flagged` IMAP tag to the current email, `remove` to remove it.

    """
    tag = "(\\Flagged)"
    if mode == "add":
      self.tag(tag)
    elif mode == "remove":
      self.untag(tag)


  def mark_as_read(self, mode:str):
    """Flag or unflag an email as read (seen).

    Arguments:
      mode (str): `add` to add the `\\Seen` IMAP tag to the current email, `remove` to remove it.

    """
    tag = "(\\Seen)"
    if mode == "add":
      self.tag(tag)
    elif mode == "remove":
      self.untag(tag)


  def mark_as_answered(self, mode:str):
    """Flag or unflag an email as answered.

    Arguments:
      mode (str): `add` to add the `\\Answered` IMAP tag to the current email, `remove` to remove it.

    Note :
      if you answer programmatically, you need to manually pass the Message-ID of the original email to the In-Reply-To and Referencess of the answer to get threaded messages. In-Reply-To gets only the immediate previous email, References get the whole thread.
    """
    tag = "(\\Answered)"
    if mode == "add":
      self.tag(tag)
    elif mode == "remove":
      self.untag(tag)


  ##
  ## Checks
  ##

  def is_read(self) -> bool:
    """Check if this email has been opened and read."""
    return "\\Seen" in self.flags

  def is_unread(self) -> bool:
    """Check if this email has not been yet opened and read."""
    return "\\Seen" not in self.flags

  def is_recent(self) -> bool:
    """Check if this session is the first one to get this email. It doesn't mean user read it.

    Note:
      this flag cannot be set by client, only by server. It's read-only app-wise.
    """
    return "\\Recent" in self.flags

  def is_draft(self) -> bool:
    """Check if this email is maked as draft."""
    return "\\Draft" in self.flags

  def is_answered(self) -> bool:
    """Check if this email has been answered."""
    return "\\Answered" in self.flags

  def is_important(self) -> bool:
    """Check if this email has been flagged as important."""
    return "\\Flagged" in self.flags

  def is_mailing_list(self) -> bool:
    """Check if this email has the typical mailing-list headers.

    Warning:
      The headers checked for hints here are not standard and not systematically used.
    """
    has_list_unsubscribe = self.has_header("List-Unsubscribe") # note that we don't check if it's a valid unsubscribe link
    has_precedence_list = self.has_header("Precedence") and self["Precedence"] == "list"
    return has_list_unsubscribe and has_precedence_list

  def is_newsletter(self) -> bool:
    """Check if this email has the typical newsletter headers.

    Warning:
      The headers checked for hints here are not standard and not systematically used.
    """
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
    """Check if any of the servers listed in the `Received` email headers is authorized by the DNS SPF rules to send emails on behalf of the email address set in `Return-Path`.

    Returns:
      score:
        - `= 0`: neutral result, no explicit success or fail, or server configuration could not be retrieved/interpreted.
        - `> 0`: success, server is explicitly authorized or SPF rules are deliberately permissive.
        - `< 0`: fail, server is unauthorized.
        - `= 2`: explicit success, server is authorized.
        - `= -2`: explicit fail, server is forbidden, the email is a deliberate spoofing attempt.

    Note:
      The `Return-Path` header is set by any proper mail client to the mailbox collecting bounces (notice of undelivered emails), and, while it is optional, the [RFC 4408](https://www.ietf.org/rfc/rfc4408.txt) states that it is the one from which the SPF domain will be inferred. In practice, it is missing only in certain spam messages, so its absence is treated as an explicit fail.

    Warning:
      Emails older than 6 months will at least get a score of `0` and will therefore never fail the SPF check. This is because DNS configuration may have changed since the email was sent, and it could have been valid at the time of sending.
    """
    # See https://www.rfc-editor.org/rfc/rfc7208
    # Output a reputation score :
    scores = { "none":    0,    # no SPF records were retrieved from the DNS.
               "neutral": 1,    # the ADMD has explicitly stated that it is not asserting whether the IP address is authorized.
               "pass":    2,    # explicit statement that the client is authorized to inject mail with the given identity.
               "temperror":  0, # the SPF verifier encountered a transient (generally DNS) error while performing the check.
               "permerror":  0, # the domain's published records could not be correctly interpreted.
               "softfail":  -1, # weak statement by the publishing ADMD that the host is probably not authorized.
               "fail": -2       # explicit statement that the client is not authorized to use the domain in the given identity.
              }

    spf_score = -2
    fast_fail = False

    if self.has_header("Return-Path"):
      name, email_address = parseaddr(self["Return-Path"])

      if email_address:
        email_domain = email_address.split("@")[1]

        # Build the list of mail servers from the DNS record of the sender domain
        try:
          mx = resolver.resolve(email_domain, 'MX')
        except:
          fast_fail = True
      else:
        fast_fail = True
    else:
      fast_fail = True

    if fast_fail:
      # No Return-Path or invalid address in there means no SPF. Fail immediately.
      # Typically indicates spoofing.
      return spf_score

    for step in self.route:
      # Get the sender email from the Return-Path

      # Build the list of IPs to test, starting with explicitely mentionned IPs
      ips = step["from"]["ip"]

      # Add IPs from DNS domains on top
      for domain in step["from"]["domain"]:
        if domain != "localhost":
          try:
            for x in resolver.resolve(domain, 'A'):
              ips.append(x.to_text())
          except:
            pass

      # Try each IP until we find a match
      for x in mx:
        # Ensure we have a set of unique IPs
        for ip in set(ips):
          try:
            spf_status = spf.check2(i=ip, s=email_address, h=x.to_text())[0]

            # Record the highest reputation score of the list of IPs
            score = scores[spf_status]
            spf_score = score if score > spf_score else spf_score

            # If we got a success, abort immediately
            if spf_score == 2:
              return spf_score
          except:
            pass

    # Return score if email is more recent than 6 months or not detected spammy.
    # Otherwise return neutral score. This is because server DKIM/ARC keys, MX and SPF may have changed.
    return 0 if self.age() > timedelta(days=30 * 6) and spf_score < 0 else spf_score

  def dkim_pass(self) -> int:
    """Check the authenticity of the DKIM signature.

    Note:
      The DKIM signature uses an asymetric key scheme, where the private key is set on the SMTP server and the public key is set in DNS records of the mailserver. The signature is a cryptographic hash of the email headers (not their content). A valid signature means the private key used to hash headers matches the public key in the DNS records AND the headers have not been tampered with since sending.

    Returns:
      score:
        - `= 0`: there is no DKIM signature.
        - `= 1`: the DKIM signature is valid but outdated. This means the public key in DNS records has been updated since they email was sent.
        - `= 2`: the DKIM signature is valid and up-to-date.
        - `= -2`: the DKIM signature is invalid. Either the headers have been tampered or the DKIM signature is entirely forged (happens a lot in spam emails).

    Warning:
      Emails older than 6 months will at least get a score of `0` and will therefore never fail the DKIM check. This is because DNS configuration (public key) may have changed since the email was sent, and it could have been valid at the time of sending.
    """

    if self.has_header("DKIM-Signature"):
      dkim_score = -2
      dk = dkim.DKIM(message=self.raw)
      signatures = self.msg.get_all("DKIM-Signature")
      for i in range(len(signatures)):
        # Sometimes there are several DKIM Signature when the message
        # transits through several servers. We need to check them all.
        try:
          output = dk.verify(i)
          # print("DKIM success on %i-th element" % i)
        except Exception as e:
          # Invalid encoding or something happened
          # print("DKIM verify failed : ", type(e).__name__, e)

          if ("value is past" in str(e)) and (dkim_score < 1):
            # The DKIM signature is valid but expired. Don't penalize users
            # because their IT guys sleep at their desk.
            dkim_score = 1
        else:
          if output and dkim_score < 2:
            # Valid DKIM signature - Abort now
            dkim_score = 2
            return dkim_score
    else:
      dkim_score = 0

    # Return score if email is more recent than 6 months or not detected spammy.
    # Otherwise return neutral score. This is because server DKIM/ARC keys, MX and SPF may have changed.
    return 0 if self.age() > timedelta(days=30 * 6) and dkim_score < 0 else dkim_score

  def arc_pass(self) -> int:
    """Check the authenticity of the ARC signature.

    Note:
      The ARC signature is still experimental and not widely used. When an email is forwarded, by an user or through a mailing list, its DKIM signature will be invalidated and the email will appear forged/tampered. ARC authentifies the intermediate servers and aims at solving this issue.

    Returns:
      score:
        - `= 0`: there is no ARC signature,
        - `= 2`: the ARC signature is valid
        - `=-2`: the ARC signature is invalid. Typically, it means the signature has been forged.
    """
    if self.has_header("ARC-Message-Signature") and \
        self.has_header("ARC-Seal") and \
          self.has_header("ARC-Authentication-Results"):
      arc_score = -2

      try:
        cv, results, comment = dkim.arc_verify(self.raw)
      except Exception as e:
        # Invalid encoding or something happened
        # print("ARC verify failed : ", type(e).__name__, e)
        pass
      else:
        if comment == "success" and arc_score < 2:
          # Valid ARC signature - Abort now
          arc_score = 2
          return arc_score
    else:
      arc_score = 0

    # Return score if email is more recent than 6 months or not detected spammy.
    # Otherwise return neutral score. This is because server DKIM/ARC keys, MX and SPF may have changed.
    return 0 if self.age() > timedelta(days=30 * 6) and arc_score < 0 else arc_score

  def authenticity_score(self) -> int:
    """Compute the score of authenticity of the email, summing the results of [EMail.spf_pass][protocols.imap_object.EMail.spf_pass], [EMail.dkim_pass][protocols.imap_object.EMail.dkim_pass] and [EMail.arc_pass][protocols.imap_object.EMail.arc_pass]. The weighting is designed such that one valid check compensates one fail.

    Returns:
      score:
        - `== 0`: neutral, no explicit authentification is defined on DNS or no rule could be found
        - `> 0`: explicitly authenticated by at least one method,
        - `== 6`: maximal authenticity (valid SPF, DKIM and ARC)
        - `< 0`: spoofed, at least one of SPF or DKIM or ARC failed and
    """
    spf_score = int(self.spf_pass())
    dkim_score = int(self.dkim_pass())
    arc_score = int(self.arc_pass())
    total = spf_score + dkim_score + arc_score
    print(self["From"], arc_score, "+", spf_score, "+", dkim_score, "=", total)
    return total

  def is_authentic(self) -> bool:
    """Helper function for [EMail.authenticity_score][protocols.imap_object.EMail.authenticity_score], checking if at least one authentication method succeeded.

    Returns:
      True if [EMail.authenticity_score][protocols.imap_object.EMail.authenticity_score] returns a score greater or equal to zero.
    """
    return self.authenticity_score() >= 0

  ##
  ## Utils
  ##

  def age(self) -> timedelta:
    """Compute the age of an email at the time of evaluation

    Returns:
      time difference between current time and sending time of the email
    """
    current_date = datetime.now(timezone.utc)
    delta = (current_date - self.date)
    return delta


  def now(self) -> str:
    """Helper to get access to date/time from within the email object when writing filters"""
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


  def query_referenced_emails(self) -> list[EMail]:
    """Fetch the list of all emails referenced in the present message, aka the whole email thread in wich the current email belongs.

    The list is sorted from newest to oldest. Queries emails having a `Message-ID` header matching the ones contained in the `References` header of the current email.

    Returns:
      All emails referenced.
    """
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


  def query_replied_email(self) -> EMail:
    """Fetch the email being replied to by the current email.

    Returns:
      The email being replied to.

    """
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


  def __init__(self, raw_message:list, server):
    # Position of the email in the server list
    super().__init__(raw_message, server)

    self.urls = []
    """`list(tuple(str))` List of URLs found in email body."""

    self.ips = []
    """`list(str)` List of IPs found in the server delivery route (in `Received` headers)"""

    self.domains = []
    """`list(str)` List of domains found in the server delivery route (in `Received` headers)"""

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
    self.raw = raw_message[1]
    self.msg : email.message.EmailMessage = email.message_from_bytes(self.raw, policy=policy.default)
    """`(email.message.EmailMessage)` standard Python email object"""

    # Get "a" date for the email
    self.get_date()

    # The hash uses the date defined above, so we need to create it after
    self.create_hash()

    # Get the message route
    self.parse_network_route()
