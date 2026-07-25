# Natural Language Processing

## Training a language model

The [Word2Vec][core.nlp.Word2Vec] model is trained from pre-tokenized documents. Each document is a list of sentences, and each sentence is a list of tokens. The [Tokenizer][core.nlp.Tokenizer] object is therefore part of the training pipeline and is saved inside the model so production code can tokenize new text the same way.

The corpus is streamed straight out of the canonical database the [crawler](6-crawling-pages.md) fills — the `tokenized` column is written at crawl time, so there is nothing to pre-process and nothing to hold in RAM. [`database.SQLitePageCorpus`][core.database.SQLitePageCorpus] is a lazy iterable that drives the SQL cursor one row at a time, and gensim consumes it as a streaming corpus (one pass to build the vocabulary, then one per epoch). Create a user script:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import nlp, database

tokenizer = nlp.Tokenizer()
db = database.open_db("corpus.db", mode="ro")

# Stream the pre-tokenized sentences of every FR/EN page (max_depth=0 keeps the
# per-sentence nesting Word2Vec expects). The language filter is applied inside
# the SQL query, i.e. during streaming — no whole-corpus copy in memory.
documents = database.SQLitePageCorpus(
    db,
    "SELECT tokenized FROM pages WHERE lang IN ('fr', 'en')",
    max_depth=0,
)

w2v = nlp.Word2Vec(
    documents,
    "word2vec-ansel",
    tokenizer=tokenizer,
    vector_size=300,
    epochs=200,
    window=15,
    min_count=32,
    sample=0.0005,
)

print(w2v.wv.most_similar(w2v.tokenizer.normalize_token("free", "en")))
db.close()
```

This saves a `word2vec-ansel` model into `VirtualSecretary/models`. To retrieve it later:

```python
from core import nlp

w2v = nlp.Word2Vec.load_model("word2vec-ansel")
```

## Training an AI classifier

[Classifier][core.nlp.Classifier] learns labels from `Data(text, label)` samples, using an existing [Word2Vec][core.nlp.Word2Vec] model to vectorize the text. The label can be any meaningful class: a topic, a document type, or an IMAP folder name.

This example trains a classifier on the [NPS Chat](http://faculty.nps.edu/cmartell/npschat.htm) corpus:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
import nltk
from core import nlp

nltk.download("nps_chat")

tokenizer = nlp.Tokenizer()
posts = list(nltk.corpus.nps_chat.xml_posts())

documents = [
    tokenizer.tokenize_document_per_sentence(post.text)
    for post in posts
]

w2v = nlp.Word2Vec(
    documents,
    "word2vec-chat",
    tokenizer=tokenizer,
    epochs=2000,
    window=3,
    min_count=2,
)

training_set = [
    nlp.Data(post.text, post.get("class"))
    for post in posts
]

model = nlp.Classifier(
    training_set,
    "chat-classifier",
    w2v,
    validate=True,
    variant="svm",
)

test_messages = [
    "Do you have time to meet at 5 pm ?",
    "Come with me !",
    "Nope",
    "What do you think ?",
]

for item in test_messages:
    label, confidence = model.prob_classify(item)
    print(item, label, confidence)
```

!!! note

    `validate=True` splits the corpus into training and test subsets and prints accuracy metrics. This helps tune the tokenizer, Word2Vec parameters, and classifier variant. Once the metrics are satisfactory, retrain with `validate=False` so the final model can use all available samples.

The saved classifier can later be loaded in filters or user scripts:

```python
from core import nlp

model = nlp.Classifier.load("chat-classifier")
label, confidence = model.prob_classify("What do you think ?")
```

## Conclusion

The NLP objects are designed to be saved and reused later in filters. The general workflow is:

1. Gather labelled text, either from crawled pages, emails, contacts, comments, or another protocol.
2. Tokenize the text with [core.nlp.Tokenizer][].
3. Train a language model with [core.nlp.Word2Vec][].
4. Train a classifier with [core.nlp.Classifier][] from `Data(text, label)` samples.
5. Load the classifier in a processing filter and act only when `prob_classify()` returns a confidence above a threshold.
