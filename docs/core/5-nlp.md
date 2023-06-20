# NLP

::: core.nlp

## Examples


### Training a language model

Assuming you followed the example of the [crawler](4-crawler.md#examples) module, write another user script with:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import utils
from core import nlp

embedding_set = []

# Open an existing dataset
for post in utils.open_data("ansel"):
    # Use only the content field of a `crawler.web_page` object
    embedding_set.append(post["content"])

# Build the word2vec language model
w2v = nlp.Word2Vec(embedding_set, "word2vec", epochs=200, window=15, min_count=32, sample=0.0005)

# Test word2vec: get the closest words from "free"
print(w2v.wv.most_similar("free"))
```

This will save a `word2vec` file into `VirtualSecretary/models`. To retrieve it later, use:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import nlp

w2v = nlp.Word2Vec.load_model("word2vec-public")
```

### Training an AI-based search engine indexer

Assuming you built the `word2vec` model above, create another user script with:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import utils
from core import nlp

# Create an index with pre-computed word embedding for each page
indexer = nlp.Indexer(utils.open_data("ansel"),
                      "search_engine",
                      nlp.Word2Vec.load_model("word2vec"))

# Do a test search
text_request = "install on linux"
tokenized_request = indexer.tokenize_query(text_request)
vectorized_request = indexer.vectorize_query(tokenized_request)
results = indexer.rank(vectorized_request, nlp.search_methods.AI)

# Display only the 25 best results
for url, similarity in results[0:25]:
    page = indexer.get_page(url)
    print(page["title"], page["excerpt"], page["url"], page["date"], similarity)

```

The [Indexer][core.nlp.Indexer] object is automatically saved to `VirtualSecretary/models` as a compressed [joblib][] object containing its own [Word2Vec][core.nlp.Word2Vec] language model, so the indexer is standalone. To retrieve it later, use:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import nlp

indexer = nlp.Indexer.load("search_engine")
```

### Training an AI classifier

We will use the [NPS Chat](http://faculty.nps.edu/cmartell/npschat.htm) text corpus. It's a text corpus of chat messages, labelled by category (like "Yes-No question", "Wh- question", "Greeting", "Statement", "No answer", "Yes answer", etc.).  The purpose of the classifier will be to automatically find the label of a new message, by learning the properties of each label into the training corpus.

Create a new user script with:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
import nltk
from core import nlp

# Download the training set
nltk.download('nps_chat')

# Build the word2vec language model
training_set = [post.text
                for post in nltk.corpus.nps_chat.xml_posts()]
w2v = nlp.Word2Vec(embedding_set, "word2vec_chat", epochs=2000, window=3)

# Test word2vec
print("free:\n",
      w2v.wv.most_similar(w2v.tokenizer.normalize_token("free", "en")))

# Build the classifier model
training_set = [nlp.Data(post.text, post.get('class')) # (content, label)
                for post in nltk.corpus.nps_chat.xml_posts()]
model = nlp.Classifier(training_set, "chat", w2v, validate=True)

# Classify test messages
test_messages = ["Do you have time to meet at 5 pm ?",
                 "Come with me !",
                 "Nope",
                 "What do you think ?"]

for item in test_messages:
    print(item, model.prob_classify(item))
```

Output:
```
free:
[('xbox', 0.3570968210697174),
 ('gam', 0.3551534414291382),
 ('wz', 0.3535629212856293),
 ('howdi', 0.3532298803329468),
 ('anybodi', 0.340751051902771),
 ('against', 0.33561158180236816),
 ('hb', 0.32573479413986206),
 ('yawn', 0.3226745128631592),
 ('tx', 0.32188209891319275),
 ('hiya', 0.31899407505989075)]

accuracy against test set: 0.803030303030303
accuracy against train set: 0.9188166152007172

Do you have time to meet at 5 pm ? ('whQuestion', 0.39620921476180504)
Come with me ! ('Emphasis', 0.46625803160949525)
Nope ('nAnswer', 0.48401087375968443)
What do you think ? ('whQuestion', 0.9756292257900939)
```

!!! note

    The classifier above was trained with `validate=True` which splits the training corpus in 2 sets: an actual training set used to extract the properties of labels, and a test set, discarded from the training and only used at the end to test if the model prediction matches the actual label. This helps tuning the hyper-parameters of the model.

    The accuracies of the model against each set are shown in the terminal output. Values close to 1.0 mean the model gets it right everytime. It is expected that the accuracy against training set would be higher than against the test set. However, if there is a large difference between both (like 0.65/0.95), it means your model is over-fitting and will lack generality.

    When satisfying accuracies have been found, you can retrain the model with `validate=False` to use all available data for maximum accuracy before using it in production.

The `chat` model will again be saved automatically in `VirtualSecretary/models` folder. Similarly to the previous objects, once saved, it can be later retrieved with :

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import nlp

model = nlp.Classifier.load("chat")
```

## Conclusion

The NLP objects are designed to be easily saved and later retrieved to be used in filters. The general workflow is as follow:

1. gather data, either from the `Secretary` (using the `learn` filter mode to aggregate email bodies, contacts, comments, etc.) or from the `Crawler`, to scrape web pages and local documents,
2. train (offline) a language model (`Word2Vec`) with the data, with user scripts,
3. train (offline) a search engine `Indexer` model or a content `Classifier` model, depending on your needs, with user scripts,
4. from your processing filters, retrieve the trained models and process (online) the email contents to decide what actions should be taken. For the `Classifier`, the probability (confidence) of the label is returned as well and can be used in filters to act only when the confidence is above a threshold.
