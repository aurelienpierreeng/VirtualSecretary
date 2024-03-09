import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import crawler

def basic_fetch(url):
  response = crawler.get_page_content(url)
  print(response)
  page = crawler.parse_page(response[0], url, "en", "body")[0]
  print(url, page["date"])

urls = ["https://en.wikipedia.org/wiki/Purple_fringing",
        "https://aurelienpierre.com",
        "https://ansel.photos/en"]

for url in urls:
  basic_fetch(url)
