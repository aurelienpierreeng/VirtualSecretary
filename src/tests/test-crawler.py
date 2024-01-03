import pickle
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import crawler


output = crawler.Crawler().get_website_from_crawling("https://github.com/aurelienpierreeng/ansel/wiki", "en", markup=["h1", "article", ("div", { 'class': 'markdown-body' })], recurse=True)

print(output)
