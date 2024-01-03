"""
Test the natural language model generation with the "Chat" dataset.

This corpus has 10567 chat messages, manually tagged with a category like "statement", "yes/no question", "wh* question", "greet", etc.

"""

import os
import sys

import nltk

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import nlp

# Build the training set of Data
#nltk.download('nps_chat')
embedding_set = [post.text for post in nltk.corpus.nps_chat.xml_posts()]

# Build the word2vec language model
w2v = nlp.Word2Vec(embedding_set, "word2vec_chat", epochs=2000, window=3)

# Test word2vec
print(w2v.wv.most_similar(w2v.tokenizer.normalize_token("free", "en")))

# Load existing model
w2v = nlp.Word2Vec.load_model("word2vec_chat")

# Test word2vec again
print(w2v.wv.most_similar(w2v.tokenizer.normalize_token("free", "en")))

# Build the model
training_set = [nlp.Data(post.text, post.get('class')) for post in nltk.corpus.nps_chat.xml_posts()]
model = nlp.Classifier(training_set, "chat", w2v, validate=True)

# Classify test stuff
l = ["Do you have time to meet at 5 pm ?", "Come with me !", "Nope", "Well, fuck", "What do you think ?"]

for item in l:
  print(item, model.classify(item))
  print(item, model.prob_classify(item))

# Load the existing model and retry
model = nlp.Classifier.load("chat")
for item in l:
  print(item, model.classify(item))
  print(item, model.prob_classify(item))
