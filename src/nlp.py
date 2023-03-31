"""
Natural language processing module.

© 2023 - Aurélien Pierre
"""

import re
import nltk
nltk.download('punkt')

from nltk.classify import SklearnClassifier
from sklearn.svm import SVC

import random
import pandas as pd

import joblib
import os


class Data():
  def __init__(self, text: str, label: str):
    """
    text : the content to label, which will be vectorized
    label : the category of the content, which will be predicted by the model
    """
    self.text = text
    self.label = label


class Classifier:
  def __init__(self, feature_list: list, path: str, force_recompute = False, validate = True):
    """
    feature_list : list of Data elements
    path : path to save the trained model for reuse, as a Python joblib
    force_recompute : recompute even if an existing joblib has been found
    """

    self.path = path

    if feature_list:
      self.set = [(self.dialogue_act_features(post.text), post.label) for post in feature_list]

    if os.path.exists(path):
      # Load the existing model if any
      print("found a pre-trained model")
      self.model = joblib.load(path)

    if not os.path.exists(path) or force_recompute:
      # Recompute the model
      self.train_dataset_type(validate=validate)


  def update_dict(self, dt:dict, key:str):
    # Increment the number of occurences for each feature
    if key in dt:
      dt[key] += 1
    else:
      dt[key] = 1

    # Note : it's not clear if this is better than just having a TRUE/FALSE state discarding number of occurrences


  def dialogue_act_features(self, post, external_data:list = [], language='english'):
    """
      Extract features from the text.

      We use meta-features like date, prices, time, number that discard the actual value but retain the property.
      That is, we don't care about the actual date, price or time, we only care that there is a date, price or time.
      For everything else, we use the actual word.

      external_data is an optional, arbitrary set of (meta)data that can be added to the feature set

    """

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

    features = {}
    for word in nltk.word_tokenize(post, language=language):

      word = word.lower()

      if re.match(r".*user\d+", word):
        # Discard the names of chat users
        continue

      elif re.match(r"\d{2,4}(-|\/)\d{2}(-|\/)\d{2,4}", word) or \
        re.match(r"\d{1,2}(th|nd|st)", word):
        # Record dates format - we don't need to know which date
        self.update_dict(features, 'DATE')

      elif re.match(r"\d{1,2}:\d{2}", word) or re.match(r"\d{1,2}(am|pm)", word):
        # Record time/hour format - we don't need to know what time
        self.update_dict(features, 'TIME')

      elif re.match(r"(CAD|USD|EUR|€|\$|£)", word):
        # Record price - we don't need to know which one
        # Note that the word tokenizer will split numbers and monetary units, so no need to regex digits + unit
        self.update_dict(features, 'MONEY')

      elif re.match(r"^\d+$", word):
        # Discard numbers, just record that we have a number
        self.update_dict(features, 'NUMBER')

      elif re.match(r"\d", word):
        # We have a mix of digits and something else. Weird stuff, count it as digit.
        self.update_dict(features, 'DIGIT')

      elif re.match(r"https?\:\/\/([^:\/?#\s\\]*)(?:\:[0-9])?([\/]{0,1}[^?#\s\"\,\;\:>]*)", word):
        # Contains url - we don't need to know which site
        self.update_dict(features, 'URL')

      elif re.match(r"\.{2,}", word):
        # Teenagers need to calm the fuck down, ellipses need no more than three dots
        self.update_dict(features, '...')

      elif re.match(r"-{1,}", word):
        # Same with dashes
        self.update_dict(features, '-')

      elif re.match(r"\?{1,}", word):
        # Same with question marks
        self.update_dict(features, '?')

      elif word in dates[language]:
        # Record textual dates - we don't need to know which date
        self.update_dict(features, 'DATE')

      else:
        self.update_dict(features, '{}'.format(word))

    if external_data:
      for elem in external_data:
        features["{}".format(elem)] = 1

    return features


  def train_dataset_type(self, validate=False):
      # Validate : split data between training and testing sub-sets to validate accuracy
      # Don't validate : use the whole dataset for training
      print("samples:", len(self.set))

      # Get all features in featureset into a flat list
      words = [list(features[0].keys()) for features in self.set]
      words_flat = [[item, 1] for sublist in words for item in sublist]

      # Put features into a dataframe, count their occurrences and
      # * remove the ones we find less than once in 1000 samples because they are outliers
      # * remove the ones we find more than 800 times in 1000 samples because they are too common to help classification
      df = pd.DataFrame(words_flat)
      df = df.groupby(0).count().sort_values(1)
      print("possible features: ", len(df.index))
      threshold = max(0.0005 * len(self.set), 1)
      df = df[df[1] > threshold]
      df = df[df[1] < 0.80 * len(self.set)]
      top_tokens = df.index.tolist()
      print("retained features: ", len(top_tokens))

      # Rebuild the featuresets with only the most common features
      new_featureset = [(dict([(key, entry[0][key]) for key in entry[0] if key in top_tokens]), entry[1]) for entry in self.set]

      if validate:
        size = int(len(new_featureset) * 0.05)
      else:
        size = 0

      random.shuffle(new_featureset) # shuffle in-place
      train_set, test_set = new_featureset[size:], new_featureset[:size]
      self.model = SklearnClassifier(SVC(kernel="rbf", probability=True), sparse=False).train(train_set)

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
