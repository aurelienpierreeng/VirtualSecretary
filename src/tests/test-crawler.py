import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import crawler
from core import utils
from core import deduplicator
from core import nlp
from core import batching

# Ansel

output = []
with crawler.Crawler(delay=1.) as cr:
  cr.no_follow += [
    "darktable.org",
    "darktable.fr",
    "pixls.us",
    "aurelienpierre.com",
    "persons-profile",
    "/view-album/",
    "/cmts-view/",
    "/blob/",
    "github.com/aurelienpierreeng/ansel",
  ]

  output += cr.get_website_from_sitemap("https://ansel.photos",
                                        "en",
                                        sitemap="/en/sitemap.xml",
                                        markup=("div", {"id": "content-body"}),
                                        category="reference",
                                        internal_links="external",
                                        mine_pdf=True)

# Dedup needs normalization ahead
output = batching.batch_parse_web_page(output, nlp.Tokenizer())
dedup = deduplicator.Deduplicator()
dedup.urls_to_ignore += [
  "darktable.org",
  "darktable.fr",
  "pixls.us",
  "aurelienpierre.com",
]
output = dedup(output)
utils.save_data(output, "test-crawler")
