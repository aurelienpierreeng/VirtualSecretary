# Writing filters

## Email filters

### Structure

Filters named like `xx-imap-name.py` are triggered by incoming email fetched from IMAP. A basic email filter can be found in `examples/common/03-imap-invoices.py`. Here is the structure :

```python
protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  query = ["confirmation", "invoice", "billing", "facture", "achat", "facture", "commande"]
  return email.is_in(query, "Subject")

def action(email):
  email.move("INBOX.Money.Invoices")

imap.get_objects("INBOX")
imap.run_filters(filter, None)
```

Let's analyse how it's written:

```python
protocols = globals()
imap = protocols["imap"]
```

is mandatory to get the mail server from the main script `main.py`. The mail server contains a live SSL connection to the IMAP server which fetches the login credentials from the `settings.ini` files. You can copy-paste these as-is in all filters. Every server protocol implemented will be available from the `protocols` dictionary. Note that getting the servers trough `globals` is necessary only from outside the `action()` and `filter()` functions, to be able to call `imap.get_objects()` and `imap.run_filters()`. From within the functions, they are referenced from `email.server` and can be accessed locally.

```python
def filter(email) -> bool:
  query = ["confirmation", "invoice", "billing", "facture", "achat", "facture", "commande"]
  return email.is_in(query, "Subject")

def action(email):
  email.move("INBOX.Money.Invoices")
```

are the filtering template functions. As input argument, they take an `email` instance of the `EMail` instance as defined in `/src/protocols/imap_object.py`. The `filter` function performs the test over the email content and outputs a boolean, `True` if the conditions are met, `False` otherwise. If `True`, the `action` function is executed on the email, otherwise nothing happens. Here, the filter checks if the subject of the email contains any word in the `query` list and the action taken is to move the email to the `Money/Invoices` IMAP folder.

---

Note : IMAP subfolders can be separated either with `/`, as in `INBOX/Money/Invoices` (format used by Gmail and Hotmail) or with `.`, as in `INBOX.Money.Invoices` (format used by most open-source servers). You may use one or the other when writing filters, the one you prefer, the program will detect automatically which one your mail server uses and will re-encode subfolders properly. Spaces and accentuated characters are also supported in IMAP folders.

---

```python
imap.get_objects("INBOX")
imap.run_filters(filter, action)
```

finally opens a mailbox, here set to `INBOX` (the main inbox), and then passes on the `filter` and `action` methods to the mail loop. These will then be run sequentially on the `n` latests emails in `INBOX`. By default, the number of emails loaded from the mailbox is set to whatever is set as `entries =` in `settings.ini` under the `imap` section. It is possible to override the global setting by calling `imap.get_mailbox_emails("INBOX", n_messages=20)`.

The line `imap.run_filters(filter, action)` can be called as-is and will apply the filters only once by email, meaning if you execute the main script more than once, emails already processed will be ignored. This is done by keeping a database of already-processed emails in an hidden file, identifying emails by an unique MD5 hash and timestamp, and recording the number of runs for each filter. This is especially useful to manually take over auto spam filters : auto filters run just once, giving the user the opportunity to correct, but don't re-apply on top of user changes. To bypass this limit, you can call `imap.run_filters(filter, action, runs=-1)` or set `runs =` to the max number of runs you want.

It is possible to call the filters on more than one mailbox in the same filter file, like:

```python
imap.get_objects("INBOX")
imap.run_filters(filter, action, filtername)

imap.get_objects("INBOX.spam")
imap.run_filters(filter, action, filtername)
```

or you can even loop over all the folders found on your IMAP server, excluding the spam folder:

```python
for folder in imap.folders:
  if folder != imap.junk:
    imap.get_objects(folder)
    imap.run_filters(filter, action)
```


### Filtering fields

The `/src/mailserver.py` script decodes and parses emails content, and provides direct access to the different fields through the `EMail` class. Here are the available fields that you can query to write your filters:

* `Email` is an object that stores the content of an email and provides parsing methods running on-demand or in background when reading an email from the IMAP server:
  * `Email.msg` is a property that instanciate an `EmailMessage` object from the standard Python package [`email`](https://docs.python.org/3/library/email.message.html). Most of the methods in `Email` are helpers and proxies on top of this object. All the methods of `EmailMessage` can be accessed from there.
  * `Email.headers` is the list of header fields found in the message. It is a proxy for `EmailMessage.keys()` from the standard Python package [`email`](https://docs.python.org/3/library/email.message.html). The header fields can then be accessed as a Python dictionnary:
    * Basic fields:
      * `EMail["From"]`: sender of the email, either an single email like `j.doe@server.com` or a name/email pair like `"John Doe" <j.doe@server.com>`. For convenience, the email addresses and names are parsed and provided as lists in the method `names, addresses = EMail.get_sender()` as `['j.doe@server.com']` no matter how it is declared in `EMails["From"]`,
      * `EMail["Date"]`: date of sending of the email,
      * `EMail["Subject"]`: subject of the email, may be set to empty string,
      * `EMail["To"]`: recipient of the email, mandatory but mildly irrelevant here,
      * `EMail["Message-ID"]`: an unique ID set by the SMTP server that sent the email. It should be defined but some spam emails don't have one.
    * Special fields:
      * `EMail["Return-Path"]`: usually used by newsletters and auto mailing, it sets the email address to notify when an email bounces (can't be delivered). Mailing-lists software can then run scripts that fetch bounces from the email capturing them and remove the bouncing address from their lists.
      * `EMail["Reply-To"]`: the preferred email address to be used if you intend on replying to this email (typically the same as the one which sent it, but not necessarily),
      * `Email["In-Reply-To"`: if the current email is a reply to another email, this header stores the `Message-ID` of that particular email.
      * `Email["References"]`: if the current email is a reply to another email, this header stores the `Message-ID` of that particular email (aka duplicates the `In-Reply-To` field) but also all the `Message-ID` of the previous emails in the thread. This allows email clients to follow threads.
      * `EMail["Delivery-date"]`: date and time of delivery on your IMAP server.
    * Mass-mailing (newsletters or spam) fields :
      * `EMail["Precedence"]`: set to `bulk` if the email was sent through a mass-mailing system, good hint to detect newsletters,
      * `EMail["List-Unsubscribe"]` : for bulk emails, this must to be set to either an unique link to follow to unsubscribe from the mailing list, or an email address to which send some email to opt-out the mailing list. Many spam emails don't define it, which is a good clue.
      * `EMail["List-Unsubscribe-Post"]` : defines the method to unsubscribe from the mailing list. Should be set to `List-Unsubscribe=One-Click`, which means you only need to visit the URL set in `EMail["List-Unsubscribe"]` to get removed from the mailing list. Unfortunately, most spammers use a double opt-out technique where you need to visit the link AND click on a confirmation button, which means you can't script a mass-unsubscribe filter.
      * `EMail["List-ID"]`: the mailing-list of that current email.
    * Custom fields: user agents are allowed to define their own custom fields, starting with `X`:
      * `EMail["X-Mailer"]`: user agent sending the email,
      * `EMail["X-Spam-Status"]`, `EMail["X-Spam-Score"]`, `EMail["X-Spam-Bar"]`, `EMail["X-Ham-Report"]`, `EMail["X-Spam-Flag"]`: SpamAssassin headers giving clues on whether the message is spam or not.
    * Any header entry found in email will create a field here, accessible through the header name as a key in the dictionnary. The above fields are given as examples to get started, more can be found in the [RFC822](https://www.w3.org/Protocols/rfc822/3_Lexical.html#z1).
  * `EMail.flags` is a flat string of characters containing all the IMAP flags, standard and user-defined, like `\Seen` if the message has been read or `Junk` if Thunderbird detected it as spam, or any custom flag you set. User-defined flags are treated as labels or tags by most email clients.
  * `EMail.ips` is a list of all the IP of the servers through which your email transited before getting into your mailbox. It is parsed from the `Received` headers, which contains the whole server route taken by the email.
  * `EMail.urls` is a list of tuples containing all URLs in the email body, both from the plain text and HTML versions. The tuples split URLs as `(domain, page)` like :
    * `https://google.com/index.php` is broken into `('google.com', '/index.php')`
    * `https://google.com/` is broken into `('google.com', '/')`
    * `https://google.com/login.php?id=xxx` is broken into `('google.com', '/login.php')`
  * `EMail.uid` is the unique ID set by the IMAP server. There is a caveat though : it is relative to a particular mailbox and doesn't follow emails through mailboxes, meaning if you move an email in another folder, this UID will change. Since this is very annoying, the `Email` class provides another unique ID :
  * `EMail.hash` is a truly unique ID for each email which addresses the problems of the UID above. It is generated from the email headers content and will therefore follow an email through folder moves.
  * `EMail.attachments` is a list of the file names of all the attachments. It does not contain the actual files, but allows to detect the files extension.
  * Regular expressions:
    * `EMail.url_pattern` is the regex used to extract URLs from the body,
    * `EMail.email_pattern` is the regex used to extract the email from the `From`.
    * In your filters, to extract all emails from the body, you could then reuse them like `EMail.email_pattern.findall(EMail.email.get_body())`.
  * Tests and checks : those are helper functions performing usual tests over the properties of the email:
    * `EMail.is_read() -> bool`: check if the email has been read, by looking for the standard tag `\Seen`,
    * `Email.is_unread() -> bool`: same as above, but returns `True` if the email has **not** been read,
    * `EMail.is_recent() -> bool`: check if the email has the standard tag `\Recent` which means no other email client got it yet,
    * `EMail.is_draft() -> bool`: check if the email has the standard tag `\Draft` which means the email has been written but not sent yet,
    * `EMail.is_answered() -> bool`: check if the email has the standard tag `\Answered` which means you sent a reply to it,
    * `EMail.is_important() -> bool`: check if the email has the standard tag `\Flagged` which means it has been labelled as important.
    * `EMail.has_tag(tag: str) -> bool`: check if `tag` is contained in the flags `EMail.flags`.
    * `EMail.is_in(query_list, field: str, case_sensitive: bool=False, mode: str="any")`: search if the elements in query list can be found in `field`. Field can by any of keys of the email headers (see above), like `'From'` or the email body (using all HTML and plain text versions) using the key `'Body'`. The default `mode='any'` returns `True` if any of the elements in `query_list` is found in the field, otherwise `mode='all'` returns `True` if all elements in `query_list` are found in the field. If using the default `case_sensitive=False` mode, all elements in `query_list` should be in lower case. This method will perform the necessary checks on the validity of the field and will return `False` if the field is not present or empty in the current email.
  * `Email.get_body(preferencelist=('related', 'html', 'plain'))`: parse and output the body. The optional argument `preferencelist` defines what part of the body is output ; if not specified, all parts are used. If no `html` body is found, the `plain` body is used instead. If no `plain` body is found, one is automatically generated from the `html` body by removing all HTML markup.
  * `Email.age()`: returns the age of the email as a time delta compared to current time (using the `Email["Date"]` header), as a [`datetime.timedelta` type](https://docs.python.org/3/library/datetime.html#timedelta-objects). This can be used to check if an email is older than a certain delta, with `Email.age() > datetime.timedelta(days=7)` returns `True` if the email is more than 7 days old. The `age` is accurate down to the second and supports time zones.

### Email actions

The `EMail` class references the `MailServer` instance containing the active IMAP connection in `EMail.server`. The `MailServer` class is inherited from the `imaplib.IMAP4_SSL` class that comes with the standard Python module `imaplib`. This can be used to call directly the underlying `imaplib` methods from the `Email` instance and therefore from your filters. But the `EMail` class provides proxy methods and implements various usual actions so you can reuse them efficiently in your filter code :

* `print(EMail)` will print a condensed version of the email, including subject, date, sender, flags, UID, etc. Useful for debugging.
* **Tagging** :
  * `EMail.tag(keyword:str)` : allows to add any tag (also named keyword or flag or label) to your emails. Note that Thunderbird has a quirk here : any tag you add programmatically here will need to be added also in Thunderbird GUI in order to appear in Thunderbird. NextCloud Mail (and apparently Horde, which provides the base libs for NextCloud Mail) detects them automatically and shows them without any fuss.
  * `EMail.untag(keyword:str)` : allows to remove any tag from your emails.
  * `EMail.mark_as_important(mode:str)` : add or remove the `\Flagged` standard flag which is interpreted by most clients as an "important" label. The `mode` needs be set to `"add"` or `"remove"`. It is an alias of `EMail.tag("\\Flagged")` and `EMail.untag("\\Flagged")`.
  * `EMail.mark_as_read(mode:str)` : add or remove the `\Seen` standard flag which is interpreted by clients as "read". The `mode` is set as the previous. It is also an alias of `EMail.tag` and `EMail.untag`.
  * `EMail.mark_as_answered(mode:str)` : add or remove the `\Answered` standard flag. The `mode` is set as the previous. It is again an alias of `EMail.tag` and `EMail.untag`.
* **Moving** : `EMail.move(folder:str)` : move the email to the specified folder, like `INBOX`. If the folder does not exist, it will be automatically created. Folders can be hierarchical, for example `INBOX.Money.Taxes`, in which case the parent folders will be created too if needed. Note that folder names are case-sensitive. To see what folders are available in your mail account, look at the first lines of the `sync.log` in your email subfolder : they will be listed.
* **Deleting** : `EMail.delete()` : entirely removes an email from your mailbox. This is without recuperation and will not use the trash bin. If you want to use the trash been, you need to move the email to trash, for example with `EMail.move("Email.server.trash)`.
* **Mark as spam** : `EMail.spam(spam_folder="INBOX.spam")` : add the `Junk` flag to the email (like Thunderbird) and move the email to the spam folder. Check that it's the right folder for your mail server.

### Email threads

* `Email.query_replied_email()` finds the email to which the current email replies (referenced by `Message-ID` in the `In-Reply-To` header), if any, by looking for its `Message-ID` header in all folders in the IMAP server. It returns another `Email` object that can be manipulated the same way as the current one (read, parsed, moved, tagged, etc.), or `None` if nothing was found.
* `Email.query_referenced_emails()` finds all the emails from the thread in which the current email belongs (referenced by `Message-ID` in the `References` header), by looking for their `Message-ID` header in all folders in the IMAP server. This can take some time since all queries go through the network. It returns a list of `Email` objects or an empty list if nothing was found.

### Sending emails

If you set an SMTP server in your `settings.ini` file, you can use it to conditionnaly send emails as a response to events triggered by incoming emails. Set your SMTP credentials like such:

```ini
[smtp]
server = server.com
user = me@server.com
password = xxxx
```

You will need to update the servers at the beginning of the filter:

```python
protocols = globals()
imap = protocols["imap"]
smtp = protocols["smtp"]
```

Then, from within the filter `action()` method, you can send an email if the filtering condition is met:

```python
def action(email):
  global smtp # needed because smtp is declared outside the scope of the function

  # ensure we have a valid SMTP connection
  if smtp and smtp.connection_inited:

    # Init the outgoing email subject from the incoming email subject
    subject = "Re: " + email["Subject"]

    # Dummy body content for a confirmation of receipt
    body = "[This is an automated answer from the mail server.]\r\n\r\n"
    body += "Your message was delivered on %s\r\n" % email["Delivery-date"]
    body += "Thanks !"

    # Init the email object, fetching receipient into current email
    smtp.write_message(subject, email["From"], body, reply_to=email)

    # Send email by SMTP and copy it to the IMAP "Sent" folder for archiving
    smtp.send_message(copy_to_sent=True)
```

The `smtp` server class `Server` inherits from the class `smtplib.SMTP_SSL` from the built-in Python package [`smtplib`](https://docs.python.org/3/library/smtplib.html). It has the following addition methods:

* `Server.write_message(subject: str, to: str, content: str, reply_to=None)`: create an `message.EmailMessage` object from the built-in Python package `email` and store it into the `Server.msg` property. Subject, receipient and content are automatically set, and more headers can be added directly through the package `email.message` using the property `Server.msg`. If the written message replies to another message, the message replied to can be passed as a `message.EmailMessage` object to the optional argument `reply_to`. The relevant headers (`In-Reply-To` and `References`) will be automatically extracted from the message being replied to for correct support of email threading in clients.
* `Server.send_message(copy_to_sent : bool = False)`: send the message through SMTP. If a valid IMAP connection is open and `copy_to_sent` is set `True`, the email will also be copied in the default `Sent` folder of the IMAP server.

You will find a basic "leave of absence" autoresponder in `examples/common/30-imap-autoresponder-absence.py`.
