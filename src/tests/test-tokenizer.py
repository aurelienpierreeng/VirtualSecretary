"""
Test the sentences and word tokenization.

"""

import os
import sys

import tests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import nlp
from core import utils

# Without stopwords
tokenizer = nlp.Tokenizer()

@utils.timeit(runs=100)
def tokenize(tokenizer, text):
  return tokenizer.tokenize_per_sentence(text)

sentences = tokenize(tokenizer, tests.text)

#for sentence in sentences:
#  print(sentence)

# With stopwords
stopwords = list(utils.get_stopwords_file("stopwords-chantal").keys())
tokenizer = nlp.Tokenizer(stopwords=stopwords)

sentences = tokenize(tokenizer, tests.text)

for sentence in sentences:
  print(sentence)
