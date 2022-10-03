# Virtual Secretary

## What is the Virtual Secretary ?

Imagine a world where :

* emails get tagged "urgent" if your agenda records an appointment in less than 48h with the email sender,
* terribly long emails get tagged "call me" if you have the phone number of their senders in your contact book,
* emails with `.ics` calendar events invites get tagged "RSVP"
* emails get sorted in folders by the CardDAV category in which their sender is, so your college alumni don't get mixed up with your family,
* emails sent by people into your customer SQL database automatically get a "client" label,
* emails sent by bulk-emailing systems get moved out of the way to their own folder, for you to read when you have time,
* etc.

The Virtual Secretary connects to :

* your IMAP email server (incoming emails),
* your SMTP email server (outgoing emails),
* your CardDAV contacts server,
* your CalDAV agendas server,
* your MySQL/MariaDB databases server.

It then provides you with an high-level Python interface to write mail filters that can cross-check data from emails, contacts, agenda and databases and define actions like (un)tagging/moving/deleting emails, updating contacts or appointments data, etc. It can also define auto-responders, email digests, etc.

Aside of the simple filter use cases, it lets you use the full extend of Python scripting to use machine-learning modules, regular expressions pattern matching, or even write your own data connectors.

It comes with a large range of example filters that can be used as-is or as templates, including a ready-to-use anti-spam system using IP blacklist/whitelist and neural network AI that can be trained against your own spam box.

## How it works

The `main.py` scripts needs to be called with the path to a configuration directory, like `python main.py ~/secretary/config`. Say you have 2 email addresses, `me@domain.com` and `pro@domain.com`, you need to create a folder for each email plus one `common` folder in your `config` directory. Then populate them with your filters, named like `00-protocol-filter.py` and your credentials into a `settings.ini` configuration file. This gives you :

* `/config`:
  * `/common`:
    * `00-imap-spam.py`
    * `01-imap-urgent.py`
  * `/me`:
    * `settings.ini`
    * `03-imap-family.py`
    * `04-imap-banks.px`
  * `/pro`:
    * `settings.ini`
    * `03-imap-colleagues.py`
    * `04-imap-clients.py`

The filters put in `/common` folder will be used for all email addresses. The filters put in the email folder will be private to this email account. The `/common` folder needs to have this exact name, but the email folders can be named whatever you like and will be run in alphabetical order.

Filters will be processed in the order defined by their 2-digits priority prefix, from `00` to `99`. If the same priority level is found in both the global `common` folder and a local email folder, the global filter is overriden by the local one with a warning issued in terminal. Priorities are shared between protocols.

The script will process all the filters it finds in all the folders it finds, so you just need to create folders and filters.

The protocol that triggers the filter needs to be written in the filter name just after the priority. The available choices are `imap`, `carddav`, `caldav`, `mysql`. Even though filters can connect different protocols, one of them needs to be the input signal that will trigger the whole filtering, so for example, the `imap` trigger will launch the filters for each email in some mailbox and then dispatch events to other accounts.

The `settings.ini` need to contain the login credentials for at least the trigger protocol of each filter, like :

```ini
[imap]
user = me@server.com
password = XXXXXXXX
server = mail.server.com
entries = 20

[smtp]
user = me@server.com
password = XXXXXXXX
server = smtp.server.com
```

Filters are defined like such: in a `01-imap-invoice.py` file, write :

```python
#!/usr/bin/env python3

GLOBAL_VARS = globals()
mailserver = GLOBAL_VARS["mailserver"]
filtername = GLOBAL_VARS["filtername"]

def filter(email) -> bool:
  return "invoice" in email.body["text/plain"].lower() or "invoice" in email.header["Subject"].lower()

def action(email):
  email.move("INBOX.Invoices")

mailserver.get_mailbox_emails("INBOX", n_messages=20)
mailserver.filters(filter, action, filternamer, runs=1)
```

This very basic filter will fetch the 20 most recent emails (`n_messages=20`) from the mailbox `INBOX` (the base IMAP folder), look for the case-insensitive keyword "Invoice" in the email body and subject, and if found, will move the email to the "Invoices" IMAP folder. This folder will be automatically created if it does not exist already. The filter will be run at most one time (`runs=1`) for each email, which means that if you manually move the emails back to `INBOX` after the filter is applied, the next run will not move them again to "Invoices". The history of runs is stored in an hidden log file named `.01-imap-invoice.py.log` where emails are identified globally for all the folders in the email account, and the history is kept even if you change the content of the filter in the future.

You may find several example filters in the `/examples` folder, along with the source code.

When you run the `main.py` script, it will create a `sync.log` file into each email folder logging all operations applied on emails, connection statuses and available IMAP folders. This will help writing and debugging your filters. Errors and exceptions of the program are logged in the standard output of the terminal you used to launch the main script.

## Installing

Download the repository.
