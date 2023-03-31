from nlp import *

# Build the training set of Data
nltk.download('nps_chat')
training_set = [Data(post.text, post.get('class')) for post in nltk.corpus.nps_chat.xml_posts()]
#own_set = [(self.dialogue_act_features(post.text), post.label) for post in english_training.training_set]

# Build the model
model = Classifier(training_set, 'sentences_classifier.joblib', True)

# Classify test stuff
print(model.classify("Do you have time to meet at 5pm ?"))
print(model.classify_many(["Come with me !", "Nope", "Well, fuck", "What do you think ?"]))
print(model.prob_classify_many(["Come with me !", "Nope", "Well, fuck", "What do you thing ?"]))
