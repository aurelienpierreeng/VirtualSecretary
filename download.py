"""Download NLTK corpora"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
