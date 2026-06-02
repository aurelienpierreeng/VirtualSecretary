# Home

## Presentation

<figure style="float:left;" markdown>
![](assets/secretary.png){ align=left width="300" }
  <figcaption markdown>Secretary icons by [UltimateArm](https://www.flaticon.com/authors/ultimatearm)</figcaption>
</figure>

The Virtual Secretary is a Python framework allowing to write filters and actions to __automate your digital office workflows__ and
improve __information processing and retrieval__ by creating __relationships__ between information.

It provides connectors to aggregate and synchronize information across:

* your __emails__ (IMAP)
* your __contacts/address book__ (CardDAV),
* your __Instagram posts & comments__ (oAuth),
* typical __websites and forums__ (HTML and PDF, with optional/automatic OCR),
* AJAX-driven/Rest-driven websites: 
    * __Github__ issues, pull requests, commits, discussions,
    * __Stack Exchange__ forums,
    * __YouTube__ channels,
* __local PDF and text documents__ (automatic OCR if needed),
* _and more in the future_:
    * your agenda events (CalDAV),
    * your database entries (SQL).

With all this data, it allows you to easily create information pipelines to:

* __train an AI language model__, privately, on your own computer and on your own data, then, from it:
    * train an AI classifier (tagging documents with topics or categories),
    * create a semantic search-engine, for intranet and/or internet knowledge base, on your own infrastructure,
* __write custom filters and actions on events__. Some examples:
    * arbitrarily tag or flag incoming emails, straight on server (so all clients get the tags):
        * based on keyword detection and regex in subject, body, attachments, sender, etc.
        * based on AI classifier output (topic, category, etc.)
    * automatically sort incoming emails into (dynamically-created) IMAP folders:
        * mirroring your address book contact categories: 
        > contacts in the "family" category have their emails moved to "family" folder, same for any other category: folders are dynamically created based on the sender's category, so you only need to manage that on your CardDAV address book,
        * using a self-trained AI classifier where IMAP folders are used as tags (needs prior manually-sorted emails whithin topical folders to be trained),
        * using Github `organization/repository` detection in email subject (regex),
    * detect spam emails from a mix of:
        * SPF and DKIM checks (very reliable, yet few email providers do it…),
        * SpamAssassin headers,
        * your self-trained AI classifier,
    * program auto-responders:
        * put up vacation notices that start and end automatically at specified dates,
        * notify your correspondents that they are n-th in your backlog queue, based on the count of unread emails in your box,
        * detect questions and topics in incoming emails and send back automatic responses with the most relevant pages from your search engine,
    * mirror/duplicate notifications from third-party services on your mailbox:
        * when a new Instagram comment comes in, mirror it on a dedicated mailbox folder, respecting threading hierarchy and authors,

!!! note
    The AI language models used by Virtual Secretary are memory- and power-efficient methods from before 2015, they will run with reasonable runtimes on servers and old desktop computers, and don't use GPU. The AI layer uses Python modules interfacing with parallelized C code, so the heavy-lifting is done by compiled code.
    
    Virtual Secretary uses no LLM and provides no chat bot.

---

> __Secretary__: _a person who works in an office, working for another person, dealing with mail and phone calls, keeping records, arranging meetings with people, etc._ ([Oxford dictionnary](https://www.oxfordlearnersdictionaries.com/definition/english/secretary?q=secretary))


Human secretaries were able to use their knowledge, agreed-upon rules and their best judgment to assess whether any unplanned event (visitor, phone call, mail) was worth disturbing your planning and workflow. To cut on expenses, they have been replaced by computers and software, which send you notifications for everything, without discernment. The result is you are disturbed all the time, you have to process huge amounts of info, so you only gained more stress and more work.

The Virtual Secretary aims at __making your ultra-connected office life less stressful__, by cross-checking and pre-filtering information for you, such that you get a digest of only the important/urgent information, and can deal with less important information later, at your own pace.

The framework provides an high-level Python API allowing you to efficiently write simple and advanced email filters, unleashing the full power of the Python programming language, along with its powerful ecosystem of packages for data mining, machine learning, regular expression search, etc.

The Virtual Secretary works standalone: it doesn't need any intermediate piece of software, but connects directly to servers. It can therefore be used on any typical mailbox supporting IMAP v4 (Gmail, Outlook/Office365, self-hosted and custom servers). It can work alongside your usual desktop mail client (Microsoft Outlook, Mozilla Thunderbird, web clients like Zimbra, Horde, SoGo, etc.).

---

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
* __works on server or desktop__, on demand or as a Cron job. A locking mechanism prevents more than one instance to process each mailbox. AI classifiers can be trained locally on desktop and sent to run read-only on the server,
* an overridable internal logging mechanism prevents emails from being processed more than once, so automatic actions that are manually reverted are not performed again on the next run.

---

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
  # Uses SPF, DKIM and ARC.
  if not email.is_authentic():
    return True

  names, addresses = email.get_sender()

  # Sender is in the address book : exit early
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


### Build a semantic search-engine for your website

"Semantic" means the search-engine understands synonyms and possibly translations, otherwise basic information retrieval systems simply rely on exact keywords, which is quite limiting when users are not experts using the exact technical slang.

```python

from core import crawler, utils, deduplicator, nlp, batching, database, types, language, search

dataset_name = "ansel"

# Open a temp database to save the pages
tmp_db = database.create_temp_db()

# Instanciate a tokenizer that will split sentences into single words
# and remove English stopwords
tokenizer = nlp.Tokenizer(replacements=language.REPLACEMENTS,
                          abbreviations=language.ABBREVIATIONS,
                          stopwords=language.STOPWORDS_DICT["english"])

#######################################################################
# 1. Acquire data from the web
#######################################################################

# Crawl the website content
with crawler.Crawler(delay=1.) as cr:
  output = cr.get_website_from_sitemap("https://ansel.photos",
                                        "en",
                                        sitemap="/en/sitemap.xml",
                                        markup=("div", {"id": "content-body"}),
                                        category="reference",
                                        internal_links="external",
                                        mine_pdf=True)
# Dump the pages into the database
database.populate_db(tmp_db, output)

# Cleanup and prepare crawled data: extract dates and guess language
batching.batch_parse_web_page(tmp_db, tokenizer)

# Deduplicate pages
dedup = deduplicator.Deduplicator()
dedup(tmp_db)

# Tokenize the whole corpus
batching.batch_tokenize(tmd_db, tokenizer, only_none=False)

# Compress and save the database for later reuse
database.compress_db(tmp_db)
utils.save_data(tmp_db, dataset_name)

#######################################################################
# 2. Train the language model
#######################################################################

# Train an n-gram-aware tokenizer: split words but keep "New York City" as one single token.
# Extract only English content for reliable training
corpus = database.SQLitePageCorpus(db,
                                   "SELECT tokenized FROM pages WHERE lang IN ('en')",
                                   max_depth=1) # list of strings

tokenizer.train_ngrams(corpus,
                       " a an the "  # articles; we never care about these in MWEs
                       " for of with without at from to in on by "  # prepositions; incomplete on purpose, to minimize FNs
                       " and or "  # conjunctions; incomplete on purpose, to minimize FNs
                       " del della of von der die das van " # foreign
                      )

# Save the trained tokenizer to the disk, to be able to reuse it in the future
tokenizer.save("my-tokenizer") 
# load it from disk in the future with `nlp.Tokenizer.load("my-tokenizer")`

# Stem words for generality, with n-grams detection
batching.batch_stem(tmp_db, tokenizer)

# Train Word2Vec model only on English documents for proper semantics
corpus = database.SQLitePageCorpus(db,
                                   "SELECT stemmed FROM pages WHERE lang IN ('en')",
                                   max_depth=0) # list of list of strings

w2v = nlp.Word2Vec(corpus, 
                   "word2vec-public", 
                   vector_size=496, epochs=40, window=31, min_count=10, sample=1e-4, ns_exponent=-0.5, negative=5,
                   tokenizer=tokenizer)
# this will be automatically saved to disk.
# load it from disk in the future with `nlp.Word2Vec.load_model("word2vec-public")`

#######################################################################
# 3. Build the search engine
#######################################################################

# Build the permament database, indexed by URL as primary key
db = database.create_db("engine.db")
database.import_pages(source_db=tmp_db, destination_db=db)
database.compress_db(db)
database.delete_tmp_db(tmp_db) # we will not need the temp DB anymore

# Instanciate the search engine object, with expensive variable pre-computing
engine = search.Indexer(db, "engine", w2v, principal_components=2)
# this will be automatically saved to disk
# load it from disk in the future with `search.Indexer.load("engine", db)`

database.close_db(db)
```

All the above is fairly computationally-expensive, but:

1. in 45 lines of code, you just built a semantic search-engine from scratch, from any website, on your computer,
2. then everything is pre-computed into 2 artifacts saved on disk:
    - `engine.db`: the SQLite database of web pages (or any other text content),
    - `engine.joblib`: the pre-computed search indexer instance.

So, at runtime (possibly on server), you need those 2 pre-computed artifacts in read-only mode:

```python

from core import database, search

## Open pre-computed artifacts from disk in read-only mode:

# The database contains all the expensive data that may not fully fit in RAM,
# open it in read-only mode for performance and security.
db = database.open_db("engine.db", mode="ro") 

# The engine contains only lightweight loaders and managers,
# it will read the database when needed.
engine = search.Indexer.load("engine", db) 

# Run the user query
user_query = "How to install Ansel on Mac OS ?"
tokenized_query = engine.tokenize_query(user_query)
results = engine.rank(db, tokenized_query, search.search_methods.AI)
print(results[0:50])

# results is an ordered list: rank, url, similarity score,
# by descending order of relevance.
# In case you need more data on the page results, you will
# need to fetch them from the database, which is indexed by URL.

# Get title, date, excerpt and url again for all URLs in the first
# 50 results.
urls = [url for rank, url, score in results[0:50]]
sql_placeholder = ", ".join(["?" for _ in urls])
cursor = db.execute(
    f"SELECT title, date, excerpt, url FROM pages WHERE url IN ({sql_placeholder})",
    urls
)

# Create a list of Python dict from the results and do whatever you want with it...
# They are still ordered like results, by descending relevance
full_results = [
  { 
    "title": row[0], 
    "date": row[1], 
    "excerpt": row[2], 
    "url": row[3] 
  } 
  for row in cursor.fetchall()
]

db.close()
```

!!! warning
    In practice, you will want to train the language model on a much larger dataset than the one you are going to index in the search engine, so the language model can acquire a larger vocabulary and learn synonyms. See the [full tutorial on building a search index](/starting/7-build-your-own-search-engine.md).

!!! note
    All the stages (crawling data, training the tokenizer and the language model, building the search engine index and serving the actual queries) are independent and communicate through SQLite databases saved on disk, which means:

    - all the stages can be deployed as micro-services on different hardware, having different performance and running on different timelines. Crawling needs a lot of runtime (due to servers thresholding bots) but low performance, training language models needs a couple of hours here and there, but powerful hardware, and the language model doesn't need to be updated everytime new pages are inserted into the index,
    - updating datasets means inserting new rows into existing databases, which can be done incrementally so website crawling can only crawl new pages (since the previour crawl), and pages that go 404 or 403 are not lost in your knowledge base,
    - merging different, pre-filtered, data sources into a final index database is easy,
    - datasets can optionally be controlled and cleaned using [DB Browser for SQLite](https://sqlitebrowser.org/), providing a spreadsheet-like UI and allowing to run custom queries,
    - updating the server-side search engine means uploading 2 updated artifacts through FTP and overwritting the previous. They contain everything they need (page index, language model, tokenizer, etc.),

!!! tip
    A search engine index of 256k pages, with a language model vectorizing on 496 dimensions, produces a 8.6 GB database and uses 1.6 GB of RAM at runtime.
    
    On a laptop from 2018, using a 8 × Intel® Xeon® CPU E3-1505M v6 @ 3.00GHz: 
    
    - in a server-like situation (Flask debug server, capped at 2 threads): the indexer loads in ~5 s and returns a search result in under 300 ms,
    - in a script-like situation: the indexer loads in ~1.5s and returns a search result in under 75 ms,
    - building the full pipeline, using 520k documents for the language model:
        - requires at most 12 GB of RAM (can be reduced by using fewer cores),
        - takes around 6 h of computation (without crawling the sources).

## Extensible by design

Protocols are managed through an abstract class. To implement your own connector for protocol `xyz`, you only need to inherit the `Server` and `Content` abstract classes from `src/core/connectors.py`, then put your children classes in a file named `xyz_server.py`, into the `src/protocols` folder. It will then be automatically loaded by the framework and will be accessible from the filters through:

```python
protocols = globals()
xyz = protocols["xyz"]
```
