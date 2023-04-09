from core import nlp

# Build the training set of Data
nltk.download('nps_chat')
training_set = [Data(post.text, post.get('class')) for post in nltk.corpus.nps_chat.xml_posts()]
embedding_set = [post.text for post in nltk.corpus.nps_chat.xml_posts()]

# Build the model
model = Classifier(training_set, embedding_set, 'sentences_classifier.joblib', False)

# Test word2vec
print(model.word2vec.wv.most_similar("free"))

# Classify test stuff
l = ["Do you have time to meet at 5 pm ?", "Come with me !", "Nope", "Well, fuck", "What do you think ?"]

for item in l:
  print(model.classify(item))

print(model.classify_many(l))
print(model.prob_classify_many(l))
