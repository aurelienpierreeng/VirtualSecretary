# Home

## Presentation

The Virtual Secretary is a Python framework that provides connectors to your emails, CardDAV address books, Instagram posts & comments (_and more to come, like CalDAV events and MySQL entries_) in order to write custom filters and actions to automate your digital office workflows.

It aims at making your office life less stressful, by pre-filtering all the inputs you need to manage to sort the important/urgent information out of the notifications and other less important information that you can read later. To pre-filter inputs, it relies on cross-checking information between different sources accessible through the connectors.

The framework provides an high-level Python API allowing you to efficiently write simple and advanced email filters, unleashing the full power of the Python programming language, along with its powerful ecosystem of packages for data mining, machine learning, regular expression search, etc.

The Virtual Secretary is standalone: it doesn't need any intermediate piece of software, but connects directly to servers.

## Features

Internally, it provides low-level features exposed through a nice programming interface allowing to write clean and elegant filters:

* connection to __IMAP and SMTP mail servers__ through SSL and StartTLS, allowing to fetch, parse, delete and send threaded emails (tested with Hotmail/Live, Gmail, [Dovecot](https://www.dovecot.org/) and [Exim](https://www.exim.org/)),
* connection to __Instagram__ through their OAuth2 and Rest API _(read-only)_,
* __email authentication__ (spam detection) through [SPF](https://en.wikipedia.org/wiki/Sender_Policy_Framework), [DKIM](https://en.wikipedia.org/wiki/DomainKeys_Identified_Mail) and [ARC](https://en.wikipedia.org/wiki/Authenticated_Received_Chain),
* connection to __CardDAV address book servers__ (tested with [SabreDAV](https://sabre.io/), used by [NextCloud](https://nextcloud.com/)  and [Owncloud](https://owncloud.com/) servers) _(read-only)_,
* __natural language processing with machine-learning classifier__ ([SVM](https://en.wikipedia.org/wiki/Support_vector_machine) on top of [word2vec](https://en.wikipedia.org/wiki/Word2vec)), allowing to train your own AI against your own emails in your own language and perform automatic tagging and sorting based on content,
* filters can be processed on a desktop or on a server, on demand or as a Cron job. A locking mechanism prevents more than one instance to process each mailbox. AI classifiers can be trained locally on desktop and sent to run on the server,
* an overridable internal logging mechanism prevents emails from being processed more than once, so automatic actions that are manually reverted are not performed again on the next run.

## Some applications

The Virtual Secretary can, for example :

* move emails into IMAP folders or tag them based on their sender and content,
* detect spoofed emails and send them to the spam box,
* mirror your Instagram comments into an IMAP folder, respecting their original date, author, and threading, and attaching the original media,
* create advanced autoresponders, based on timeframes and number of unread messages in your mailbox,

## Quick demos

Filters are written in full Python and entirely modular. The quick demos after demonstrate how filters are coded.

### Detect spams and remove them

We use spoofing detection through SPF, DKIM and ARC, then check if the sender is already known in the CardDAV server, and finally use SpamAssassin headers if any.

```python
protocols = globals()
imap = protocols["imap"]
carddav = protocols["carddav"] if "carddav" in protocols else None

def filter(email) -> bool:
  # If email is spoofed or authentication is forged, exit immediately
  if not email.is_authentic():
    return True

  names, addresses = email.get_sender()

  # Email is in the address book : exit early
  if carddav.connection_inited:
    for address in addresses:
      if address in carddav.emails:
        return False

  # SpamAssassin headers if any
  if email.has_header("X-Spam-Flag"):
    if email["X-Spam-Flag"] == "YES":
      return True

  return False

def action(email):
  email.spam(email.server.junk)

imap.get_objects("INBOX")
imap.run_filters(filter, action)
```

### Sort emails in folders by CardDAV category

Assuming your CardDAV contacts have a category (like "client", "family", "friends"), we can move their emails directly to a corresponding sub-folder of a `People` parent folder (which will be created on-the-fly if it does not exist), or simply move them to `People` if they have no category:

```python
protocols = globals()
imap = protocols["imap"]
carddav = protocols["carddav"]

def filter(email) -> bool:
  global carddav
  names, addresses = email.get_sender()

  # Search if the sender is in the CardDAV address book
  if len(addresses) > 0:
    results = carddav.search_by_email(addresses[0])

    if results:
      for vcard in results:
        # Current sender can be found in multiple VCards on server, try them all
        if len(vcard["categories"]) > 0:
          # If more than one category is found, just take the first
          category = vcard["categories"][0]
          folder_tree = ["INBOX", "People", category]
          target_path = email.server.build_subfolder_name(folder_tree)
          email.move(target_path)
          return True

      # We have no category but this contact is known
      folder_tree = ["INBOX", "People"]
      target_path = email.server.build_subfolder_name(folder_tree)
      email.move(target_path)
      return True

  return False

imap.get_objects("INBOX")
imap.run_filters(filter, None)
```

### Sort Github emails by organization/project

Github can generate a lot of notifications, but since all emails subjects start with `[organization/repository]`, we can use this to sort all emails into `Github/Organization/Repository` folders. Sub-folders are created on-the-fly if they don't already exist. We will unleash here the power of regular expressions:

```python

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  if email.is_in("notifications@github.com", "From"):
    import re
    # find orga and repo in "[orga/repo] blablabla"
    match = re.search(r"\[(.*?)\/(.*?)\]", email["Subject"])

    if match:
      # Replace . by _ in orga names containing .org
      # because . is IMAP subfolder separator on Dovecot and Gmail.
      organisation = match.groups()[0].replace(".", "_")
      repository = match.groups()[1].replace(".", "_")
      folder_tree = ["INBOX", "Github", organisation, repository]
      target_path = email.server.build_subfolder_name(folder_tree)
      email.move(target_path)
      return True
    else:
      email.move("INBOX.Github")
      return True

  return False

imap.get_objects("INBOX")
imap.run_filters(filter, None)
```

### Train an AI to classify emails for you

Assuming you have manually sorted your emails into folders and sub-folders, in a way that makes sense, we can try to find patterns in the content of those emails like this:

```python
import pickle
import os

protocols = globals()
imap = protocols["imap"]
emails_set = []

def filter(email) -> bool:
  global emails_set, folder
  # Create a tuple with the content (subject and body) of the email and its foldername as label for training
  # You can use any other meaningful textual property in place of the foldername.
  emails_set.append((email["Subject"] + "\r\n\r\n" + \
    email.get_body(preferencelist='plain'), folder))
  return False

# Create the database of emails
for folder in imap.folders:
  imap.get_objects(folder, n_messages=500)
  imap.run_filters(filter, None)

# Train the word embedding and SVM classifier against the database
training_set = [imap.nlp.Data(post[0], post[1]) for post in emails_set]
embedding_set = [post[0] for post in emails_set]
model = imap.nlp.Classifier(training_set, embedding_set, 'classifier.joblib', True)
```

To use the trained model in a filter that will move emails in the relevant folder for you, it's as simple as:

```python

protocols = globals()
imap = protocols["imap"]

def filter(email) -> bool:
  global imap
  model = imap.nlp.Classifier([], [], 'classifier.joblib', False)
  content = email["Subject"] + "\r\n\r\n" + email.get_body(preferencelist='plain')
  result = model.prob_classify_many([content]))

  if result[1] > 0.5:
    # Do something only if we found a label with at least 50 % confidence
    # The label is directly the folder/subfolder
    email.move(result[0])

  return False

imap.get_objects("INBOX")
imap.run_filters(filter, None)

```

The model will be stored in a file named `classifier.joblib` that can be saved and shared between computers. On a training sample of 6500 emails mixing both French and English, I get a predictive accuracy between 87 and 90 %.


## Extensible by design

Protocols are managed through an abstract class. To implement your own connector for protocol `xyz`, you only need to inherit the `Server` and `Content` abstract classes from `src/connectors.py`, then put your children classes in a file named `xyz_server.py`, into the `src/protocols` folder. It will then be automatically loaded by the framework and will be accessible from the filters through:

```python
protocols = globals()
xyz = protocols["xyz"]
```
