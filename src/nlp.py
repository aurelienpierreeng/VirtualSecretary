"""
Natural language processing module.

© 2023 - Aurélien Pierre
"""

import re
import nltk
nltk.download('punkt') # Import tokenizer
nltk.download('word2vec_sample') # Import word2vec

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

import gensim
from nltk.data import find

import random
import pandas as pd

import joblib
import os

import numpy as np

class Data():
  def __init__(self, text: str, label: str):
    """Represent an item of training data

    Arguments:
      text (str): the content to label, which will be vectorized
      label (str): the category of the content, which will be predicted by the model
    """
    self.text = text
    self.label = label


class Classifier:
  def __init__(self, feature_list: list, embedding_list: list, path: str, force_recompute = False, validate = True):
    """
    feature_list : list of Data elements
    path : path to save the trained model for reuse, as a Python joblib
    force_recompute : recompute even if an existing joblib has been found
    """

    self.path = path

    if feature_list:
      self.set = feature_list

    if embedding_list:
      self.train_word2vec(embedding_list)
    else:
      if os.path.exists(path + ".embed"):
        self.word2vec = gensim.models.Word2Vec.load(path + ".embed")
      else:
        # If no embedding list, use Google News corpora
        word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
        self.word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

    if os.path.exists(path):
      # Load the existing model if any
      print("found a pre-trained model")
      self.model = joblib.load(path)

    if not os.path.exists(path) or force_recompute:
      # Recompute the model
      self.train_dataset_type(validate=validate)


  def update_dict(self, dt:dict, vector: list):
    # Increment the number of occurences for each feature
    for key, value in zip(dt.keys(), vector):
      dt[key] += value

    # Note : it's not clear if this is better than just having a TRUE/FALSE state discarding number of occurrences

  def normalize_token(self, word: str, language: str):
    # Return normalized tokens
    dates = {
      "english" : [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "aug", "sept", "oct", "nov", "dec",
        "jan.", "feb.", "mar.", "apr.", "aug.", "sept.", "oct.", "nov.", "dec.",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "mon.", "tue.", "wed.", "thu.", "fri.", "sat.", "sun.",
        ],
      "french" : [
        "janvier", "février", "mars", "avril", "mai", "juin",
        "juillet", "août", "septembre", "octobre", "novembre", "décembre",
        "jan", "fév", "mars", "avr", "jui", "juil", "aou", "sept", "oct", "nov", "déc",
        "jan.", "fév.", "mars.", "avr.", "jui.", "juil.", "aou.", "sept.", "oct.", "nov.", "déc.",
        "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche",
        "lun", "mar", "mer", "jeu", "ven", "sam", "dim",
        "lun.", "mar.", "mer.", "jeu.", "ven.", "sam.", "dim.",
      ]
    }

    if re.match(r".*user\d+", word):
      # Discard the names of chat users
      return ''

    elif re.match(r"\d{2,4}(-|\/)\d{2}(-|\/)\d{2,4}", word) or \
      re.match(r"\d{1,2}(th|nd|st)", word):
      # Record dates format - we don't need to know which date
      return 'ec7161663d7823b791ac47093cf1c8f5'

    elif re.match(r"\d{1,2}:\d{2}", word) or re.match(r"\d{1,2}(am|pm)", word):
      # Record time/hour format - we don't need to know what time
      return '91bdc7a74c60f8958ade537cab4ae040'

    elif re.match(r"(CAD|USD|EUR|€|\$|£)", word):
      # Record price - we don't need to know which one
      # Note that the word tokenizer will split numbers and monetary units, so no need to regex digits + unit
      return 'd35ec937471d312fd81fbb2eccc41fbb'

    elif re.match(r"^\d+$", word):
      # Discard numbers, just record that we have a number
      return 'd0452d7d9010e16cc725cd9531c965c4'

    elif re.match(r"\d", word):
      # We have a mix of digits and something else. Weird stuff, count it as digit.
      return '6e6bddf7ffaead377667afd427f5901b'

    elif re.match(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)", word):
      # Contains url - we don't need to know which site
      return '66b3b72d44d757f5312a9cc2461dbdd9'

    elif re.match(r"\.{2,}", word):
      # Teenagers need to calm the fuck down, ellipses need no more than three dots
      return '...'

    elif re.match(r"-{1,}", word):
      # Same with dashes
      return '-'

    elif re.match(r"\?{1,}", word):
      # Same with question marks
      return '?'

    elif word in dates[language]:
      # Record textual dates - we don't need to know which date
      return 'ec7161663d7823b791ac47093cf1c8f5'

    else:
      return word


  def wordvec(self, word):
    try:
      return self.word2vec.wv[word]
    except:
      return np.zeros(self.word2vec.vector_size)


  def dialogue_act_features(self, post, external_data:list = [], language='english'):
    """
      Extract features from the text.

      We use meta-features like date, prices, time, number that discard the actual value but retain the property.
      That is, we don't care about the actual date, price or time, we only care that there is a date, price or time.
      Meta-features are tagged with random hashes as to not be mistaken with text.

      For everything else, we use the actual words.

      external_data is an optional, arbitrary set of (meta)data that can be added to the feature set

    """

    # The dict is not the best suited data representation for us here
    # we create bullshit keys from integer indices of the word2vec vector size
    features = dict.fromkeys(range(self.word2vec.vector_size), 0.)

    i = 0
    for word in nltk.word_tokenize(post.lower(), language=language):
      self.update_dict(features, self.wordvec(self.normalize_token(word, language)))
      i += 1

    # Normalize (average)
    if i > 0:
      for key in features.keys():
        features[key] /= i

    if external_data:
      for elem in external_data:
        features["{}".format(elem)] = 1

    return features


  def train_word2vec(self, embedding, language="english"):
    sentences = [[self.normalize_token(token, language) for token in nltk.word_tokenize(sentence.lower())] for sentence in embedding]
    # window = 20 is the average sentence length in French administrative correspondance
    # that gives each word in its sentence context.
    self.word2vec = gensim.models.Word2Vec(sentences, vector_size=150, window=20, min_count=5, workers=8,
                                           epochs=50, ns_exponent=-0.5)
    self.word2vec.save(self.path + ".embed")


  def train_dataset_type(self, validate=False):
      # Validate : split data between training and testing sub-sets to validate accuracy
      # Don't validate : use the whole dataset for training
      print("samples:", len(self.set))

      # Get all features in featureset into a flat list
      features = [(self.dialogue_act_features(post.text), post.label) for post in self.set]

      # If validation is on, split the set into a training and a test subsets
      new_featureset = features
      if validate:
        size = int(len(new_featureset) * 0.05)
      else:
        size = 0

      random.shuffle(new_featureset) # shuffle in-place
      train_set, test_set = new_featureset[size:], new_featureset[:size]

      # C is regularization, decrease below 1 if noisy training input.
      # Here, noise got filtered already in word2vec, so no need and 12 is empiric optimum.
      self.model = SklearnClassifier(SVC(kernel="rbf", probability=True, C=15, gamma='scale'), sparse=False).train(train_set)

      if validate:
        print("accuracy against train set:", nltk.classify.accuracy(self.model, train_set))
        print("accuracy against test set:", nltk.classify.accuracy(self.model, test_set))

      # Save the model to a reusable object
      joblib.dump(self.model, self.path)


  def classify(self, post: str, external_data:list = [], language='english'):
    # Apply a label on a post based on the trained model
    item = self.dialogue_act_features(post, external_data, language)
    return self.model.classify(item)

  def classify_many(self, posts: list, external_data:list = [], language='english'):
    # Apply a label on posts based on the trained model
    items = [self.dialogue_act_features(post, external_data, language) for post in posts]
    return self.model.classify_many(items)

  def prob_classify_many(self, posts: list, external_data:list = [], language='english'):
    # Apply a label on posts based on the trained model and output the probability too
    items = [self.dialogue_act_features(post, external_data, language) for post in posts]

    # This returns a weird distribution of probabilities for each label that is not quite a dict
    proba_distros = self.model.prob_classify_many(items)

    # Build the list of dictionnaries like `label: probability`
    output = [dict([(i, distro.prob(i)) for i in distro.samples()]) for distro in proba_distros]

    # Finally, return label and probability only for the max proba of each element
    return [(max(item, key=item.get), max(item.values())) for item in output]
