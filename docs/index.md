# Home

## Presentation

<figure style="float:left;" markdown>
![](assets/secretary.png){ align=left width="300" }
  <figcaption markdown>Secretary icons by [UltimateArm](https://www.flaticon.com/authors/ultimatearm)</figcaption>
</figure>

 The Virtual Secretary is a Python framework allowing to write custom filters and actions to __automate your digital office workflows__ and
 improve __information processing and retrieval__. It provides connectors to synchronize information across:

* your emails,
* your contacts (address books),
* your Instagram posts & comments,
* _and more in the future_:
    * your agenda events,
    * your database entries.


> __Secretary__: _a person who works in an office, working for another person, dealing with mail and phone calls, keeping records, arranging meetings with people, etc._ ([Oxford dictionnary](https://www.oxfordlearnersdictionaries.com/definition/english/secretary?q=secretary))


Human secretaries were able to use their knowledge, agreed-upon rules and their best judgment to assess whether any unplanned event (visitor, phone call, mail) was worth disturbing your planning and workflow. To cut on expenses, they have been replaced by computers and software, which send you notifications for everything, without discernment. The result is you are disturbed all the time, you have to process huge amounts of info, so you only gained more stress and more work.

The Virtual Secretary aims at __making your ultra-connected office life less stressful__, by cross-checking and pre-filtering information for you, such that you get a digest of only the important/urgent information, and can deal with less important information later, at your own pace.

The framework provides an high-level Python API allowing you to efficiently write simple and advanced email filters, unleashing the full power of the Python programming language, along with its powerful ecosystem of packages for data mining, machine learning, regular expression search, etc.

The Virtual Secretary works standalone: it doesn't need any intermediate piece of software, but connects directly to servers. It can therefore be used on any typical mailbox supporting IMAP v4 (Gmail, Outlook/Office365, self-hosted and custom servers). It can work alongside your usual desktop mail client (Microsoft Outlook, Mozilla Thunderbird, web clients like Zimbra, Horde, SoGo, etc.).

## Features

Internally, it provides low-level features exposed through a nice programming interface allowing to write clean and elegant filters:

* connection to __IMAP and SMTP mail servers__ through SSL and StartTLS (tested with Hotmail/Live, Gmail, [Dovecot](https://www.dovecot.org/) and [Exim](https://www.exim.org/)):
    - fetch, parse, and inject email content, headers and attachments into arbitrary Python scripts,
    - tag and sort/move emails into IMAP folders based on manual filters (Python scripts) or AI classifiers,
    - send automatic replies (auto-responder).
* __HTML, PDF and XML__ crawling, parsing and indexing with OCR :
    - crawl websites recursively, for content and links,
    - easily create natural language corpora to train specialized AI language models that speak your business slang,
    - create specialized search engines aggregating internal and external resources you need for your job (reference implementation of a server-side interface: [Chantal](https://chantal.aurelienpierre.com)),
* connection to __Instagram__ through their OAuth2 and Rest API _(read-only)_,
  - duplicate comments into your mailbox to centralize,
* __email authentication__ (spam detection) through [SPF](https://en.wikipedia.org/wiki/Sender_Policy_Framework), [DKIM](https://en.wikipedia.org/wiki/DomainKeys_Identified_Mail) and [ARC](https://en.wikipedia.org/wiki/Authenticated_Received_Chain),
    - basic SPF and DKIM checks get rid of 90% of spam emails without relying on flimsy machine learning. Go figure why mail providers don't do it…
* connection to __CardDAV address book servers__ (tested with [SabreDAV](https://sabre.io/), used by [NextCloud](https://nextcloud.com/)  and [Owncloud](https://owncloud.com/) servers) _(read-only)_,
    - reuse your contact info to label and sort emails
* __Machine-learning language classifier__ ([SVM](https://en.wikipedia.org/wiki/Support_vector_machine) and decision trees on top of [word2vec](https://en.wikipedia.org/wiki/Word2vec)), allowing to train your own AI against your own emails in your own language and perform automatic tagging and sorting based on content,
    - perform topic recognition in emails or documents,
    - find relevant resources from a query (search engine) or an email body (auto-responder).
* __works on server or desktop__, on demand or as a Cron job. A locking mechanism prevents more than one instance to process each mailbox. AI classifiers can be trained locally on desktop and sent to run on the server,
* an overridable internal logging mechanism prevents emails from being processed more than once, so automatic actions that are manually reverted are not performed again on the next run.

## Examples of applications

The Virtual Secretary can, for example :

* move emails into IMAP folders or tag them based on their sender and content,
* detect spoofed emails and send them to the spam box,
* mirror your Instagram comments into an IMAP folder, respecting their original date, author, and threading, and attaching the original media,
* create advanced autoresponders, based on timeframes and number of unread messages in your mailbox,
* detect the language of an email.

In the future, it will be able to :

* add an "urgent" tag to emails sent by people with whom you have an appointment scheduled in the next 48 hours,
* add a "client" tag to emails sent by someone who bought your product (connecting to the MySQL database of your Prestashop or WordPress WooCommerce website).

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

Assuming your CardDAV contacts have a category (like "client", "family", "friends"), we can move their emails directly to a corresponding sub-folder of a `People` parent folder, or simply move them to `People` if they have no category:

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

!!! note
    The true power of the Virtual Secretary here is mail folders and subfolders will be created dynamically for each contact category, meaning that:

    * you don't need to know beforehand what the contact categories will be,
    * you can add more contact categories anytime in the future, with no additional work on the filter,
    * you don't need to manually map one filter with one folder, as with conventional mail filters.

### Sort Github emails by organization/project

Github can generate a lot of notifications, but since all emails subjects start with `[organization/repository]`, we can use this to sort all emails into `Github/Organization/Repository` folders and dynamically replace `Organization` and `Repository` by their actual value from each email. We will unleash here the power of regular expressions:

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

!!! note
    This is not possible to achieve with conventional email filters because they don't allow to use the content to dynamically change the action performed on the email. So you would have to write one filter per repository, like `if email.subject contains("orga/repo") then move(email, to="Github/Orga/Repo"`, in addition of manually creating each `Github/Orga/Repo` subfolder.

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

!!! note
    Training your own AI makes sure it uses your language(s) and it knows the specific vocabulary of your particular business (including slang, trademarks and company names). This training is done on your own computer or server, not on a third-party cloud, which solves one of the major data privacy concerns of AI in its current [SaaS](https://en.wikipedia.org/wiki/Software_as_a_service) approach.


## Extensible by design

Protocols are managed through an abstract class. To implement your own connector for protocol `xyz`, you only need to inherit the `Server` and `Content` abstract classes from `src/connectors.py`, then put your children classes in a file named `xyz_server.py`, into the `src/protocols` folder. It will then be automatically loaded by the framework and will be accessible from the filters through:

```python
protocols = globals()
xyz = protocols["xyz"]
```
