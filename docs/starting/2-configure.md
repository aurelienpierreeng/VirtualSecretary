# Configure

**Virtual Secretary** is designed to require as little programming as possible while still retaining the full potential of Python should you need it. The configuration is done by using declarative `settings.ini` files and templated Python filters.

An example configuration is provided in the repository in the `examples` folder, you can copy it somewhere and start editing it.

*Note : only the IMAP emails connectors are implemented for now*.

## Configuration structure

### Folders tree

Let's say your configuration base folder is called `config` for simplicity. It should contain at least a subfolder `common`, for global filters, and an email subfolder that you can call whatever you want. You may add as many subfolders as you want and name them as you want, only the `common` on is mandatory and subfolders will be processed in alphabetical order.

Each subfolder (except `common`) should contain a `settings.ini` file containing the credentials (server, login, password) for all your IMAP, SMTP, CardDAV, CalDAV and MySQL accounts. Each folder can be set for at most one account for each protocol, if you need to configure a "one to many" or a "many to one" kind of situation, you have to manage each input -> output combination in a separate subfolder. `settings.ini` files should look like that :

```ini
[imap]
user = me@server.com
password = XXXXXXXX
server = mail.server.com
entries = 20
```

### Filter naming convention

The subfolders contain the filters, which are Python files with `.py` extension. The names of the filter files contain processing instructions given to the main program, therefore they need to follow a specified format.

There are 2 kinds of filters:

1. Typical filters, performing read-write operations in real-time, which filenames are prefixed with 2 digits declaring their execution priority (`00` runs first, `99` runs last),
2. Learning filters, performing read-only operation that can be deferred at convenient times, which filenames are prefixed with `LEARN`. They don't get priorities.

The typical filters are run by calling `$ python src/main.py config process` and the learning filters are run `$ python src/main.py config learn`. This gives you the opportunity to run heavy data parsing operation at times where your mailbox is not busy, without destroying real-time responsivity. Learning filters have no explicit priorities, they will be processed by alphabetical order. Note that nothing prevents you from using learning filters as processing filters, and the other way around, they ultimately run the same — it's up to you to follow the scripting hygiene you find suitable and to apply some discipline. *I suggest respecting the intended use*.

After the prefix comes, separated by a dash `-`, comes the name of the trigger protocol (`imap`, `carddav`, `caldav`). The trigger is the kind of data that will start the filtering process, for example the `imap` protocol will loop the filters over emails, `carddav` will loop the filters over vCard contacts, etc.

Finally, the mnemonic name of the filter, separated again by a dash `-` from the previous. This is some meaningful name that will help you sort your filters for admin purposes. They accept letters (upper and lower case), digits, dashes and underscores.


### Summary

That should leave you with something like :

* `./config/`:
    * `/common/`:
        * `00-imap-spam.py`
        * `01-imap-urgent.py`
        * `LEARN-imap-ban-ip.py`
    * `/me/`:
        * `settings.ini`
        * `03-imap-family.py`
        * `04-imap-banks.px`
    * `/pro/`:
        * `settings.ini`
        * `03-imap-colleagues.py`
        * `04-imap-clients.py`
* `./src/`
    * `core` and `protocols` modules (not covered in this section).

## Connectors

### Emails

#### Gmail

For Gmail, you need to enable the [IMAP transfer](https://support.google.com/mail/answer/7126229) and use an [app-specific password](https://support.google.com/accounts/answer/185833).

Gmail IMAP uses TLS on the default port (993) and Gmail SMTP uses TLS on the default port too (465), so you only need to configure user/password/server.

#### Outlook and Office365

As of 2022, Outlook users are the free accounts and Office365 are the paying accounts. Both support IMAP over TLS with the default port (993), so it's easy.

However, Outlook (free) SMTP does **not** support TLS but only STARTTLS. This is not supported in *Virtual Secretary* because it's not secure enough. You can use a different SMTP provider though, like Gmail or anything self-hosted.

Office365 SMTP supports TLS on port 587 (non-standard), so you need to add `port = 587` in the `[smtp]` section of your `settings.ini` configuration file.

## Testing and debugging filters

When your filters and configuration files are written, you can start the main script with the path to the config file as an argument:
```bash
$ python src/main.py /path/to/config process
```

This results in something like:
```bash
$ python src/main.py config/ process

Executing filter 00-imap-spam.py :
  - IMAP        took 6.689 s    to query        50 emails from INBOX
  - Parsing     took 0.536 s    to parse        50 emails
  - Filtering   took 0.041 s    to filter       50 emails

Executing filter 01-imap-newsletters.py :
  - IMAP        took 5.826 s    to query        50 emails from INBOX
  - Parsing     took 0.522 s    to parse        50 emails
  - Filtering   took 0.042 s    to filter       50 emails

Executing filter 06-imap-calendar-rsvp.py :
  - IMAP        took 5.972 s    to query        50 emails from INBOX
  - Parsing     took 0.619 s    to parse        50 emails
  - Filtering   took 0.044 s    to filter       50 emails

Global execution took 27.59 s. Mind this if you use cron jobs.
```

In each subfolder, a `sync.log` plain-text file will keep track of every successful action made on emails. The output of the terminal, when you run the main script, will output all errors and exceptions met during the execution of the program, as well as a log of filters run and runtimes.

The history database of all filters is stored as an hidden file along with their filter. Say you run a filter which name is `00-imap-spam.py`, its database is stored in the same subfolder and named `.00-imap-spam.py.log`. In case of a problem, you can safely remove this file but this will then re-run the filter for all emails so beware of the consequences if your filters are run more than once (or any specified number of times). Those logs are not plain-text but Python dictionaries saved as pickle format (binary dump of the memory). This makes them impossible to read outside Python but very efficient to read/write and reasonably small.
