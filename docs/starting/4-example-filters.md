# Example filters

The `examples` folder contains fully-functional filters that can run as soon as you edit the `settings.ini` file with your credentials in your `config` directory. I will present them here.

## Anti-spam filter

The elephant in the room, consists of 2 filters stored in `/common/` :

* `00-imap-spam.py`,
* `LEARN-imap-ban-ip.py`.

It actually starts with `LEARN-imap-ban-ip.py`. It assumes both your spam and inbox folders have been manually cleaned, so they actually represent what you consider (un)desirable mail. This filter does not apply any action, instead it simply records the emails and IP address of spam mail and puts them in a blacklist file, then it does the same in your inbox and puts them in a whitelist file. Emails and IP addresses that can be found in both spam and inbox are removed from both lists. If you manually move emails between spam and inbox folders, the lists are updated at the next run.

The `00-imap-spam.py` filter then reuses those lists to immediately accept whitelisted senders and immediately mark as spam the blacklisted senders. For unknown senders (that are in neither lists), we then proceed to looking for SpamAssassins and check if it is a bulk email sent without a `List-Unsubscribe` (which is illegal in Europe and a reliable clue).

The `00-imap-spam.py` can be extended by posting the sending IPs (`email.ip`) and URLs found in the body (`email.url`) to a reputation server, using GET or POST requests through HTTP with the Python modules `pyCurl` or `requests`.

## Detect calendar event invites

The `examples/me@server.com/06-imap-rsvp.py` shows how to add an `RSVP` labels to emails containing invitations to calendar events. Those are generally attachments having `.ics` format. The filter flattens the list of attachments and checks if a filename ending in `.ics` is there. Though the use of regular expressions is slightly overkill in this context, it is a good example of how they can be used in filters.

## Mass-unsubscribe from all spam mailing lists

The `examples/me@server.com/05-imap-mass-unsubscribe.py` tries to unsubscribe in bulk from all the newsletters in the spam box. This is done by following the link that *should* be in the `"List-Unsubscribe"`. Problem is, many spams don't provide one, and even for those which do, a lot of mailing lists use "double opt-out" mechanisms where you need to load the page AND click a button (which is ironic since they usually got your emails from leaks and didn't ask you to double opt-in). This will work for mailing-lists using "One-Click Unsubscribe".

This can be extended by sending an email to the address in `"List-Unsubscribe"` once SMTP support is added.

## Move newsletters away

Newsletters (known as `bulk` emails) are typically content that doesn't require your immediate attention and doesn't need to stay in your main inbox. The `examples/me@server.com/01-imap-newsletters.py` filter uses the `"Precedence"` to detect such emails and moves them to a dedicated `Newsletters` folder. This needs to run after the anti-spam detection.

## Move invoices away

Again, invoices sent by email are for your archives and future reference, but they don't require your immediate attention (unless someone else ordered). The `examples/me@server.com/02-imap-invoices.py` looks for the keyword "invoice" into the email subject and body. This is a good example of simple keyword-based content filters

## Detect questions in emails

TODO. Emails containing questions may be inquiries from clients and need to be answered at some point.

### Dependencies

```bash
pip install -U pip setuptools wheel
pip install -U scikit-learn nlkt
pip install -U 'spacy[cuda117,transformers,lookups]'
python -m spacy download en_core_web_trf
python -m spacy download fr_dep_news_trf
```

See https://spacy.io/usage to install your language support.

## Sync Google Agenda with any CalDAV calendar

TODO. Many services support Google Agenda API (REST) without supporting standard CalDAV servers. There are nasty ways of sharing read-only calendars between Google and CalDAV servers, using private .ics links, but these are read-only and not secure (everyone knowing the private link can access the agenda). Also, Google doesn't check your availability by looking into third-party calendars, so people or services that can only access Google Agenda may double-book you if you already have an appointment written on another calendar. To make things worse, Google Agenda doesn't provide a ready CalDAV API, this needs to be manually activated into the developers console, along with creating an OAuth token, and the CalDAV API is v2 only. Annoying and not even future-proof (Google may discontinue CalDAV support when they want).

The most simple way to address this issue is by duplicating events and appointments between a reference CalDAV server and Google Agenda bidirectionally:

* Connect to Google Agenda through their native Rest API (with authentification),
* Connect to the CaldDAV server through CalDAV v3 protocol (with authentification),
* On first reading:
    * Google -> CalDAV : copy events integrally, including emails, locations, etc.
        * leave a token in the duplicated event, stating that it was originally a Google event,
        * save the timestamp of the last edit on Google Agenda in the CalDAV duplicate,
    * CalDAV -> Google : copy obfuscated events, discarding any personal data, just to lock the time slot for availability status,
        * leave a token in the duplicated event, stating that it was originally a CalDAV event,
        * save the timestamp of the last edit on the CalDAV server in the Google Agenda duplicate,
* On later readings: on both origins, for each event, if the event is a duplicate, check if the last edit timestamp stored in the duplicate matches the timestamp of the original. If timestamps don't match, overwrite the duplicate or delete it if the original is not found.


See https://google-calendar-simple-api.readthedocs.io/en/latest/getting_started.html
