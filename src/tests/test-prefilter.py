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

detyped = utils.typography_undo(str(tests.text).lower())
print(detyped)

tokenizer = nlp.Tokenizer()
filtered = tokenizer.prefilter(detyped)
print(filtered)
