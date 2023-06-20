# Crawler

::: core.crawler

<style>
/* Hide core.crawler.web_page attributes in TOC */
li a[href^='#core.crawler.web_page.'] { display: none; }
</style>

## Examples

The best way to use the crawler is by adding a script in `src/user_scripts`, since it is meant to be used as an offline training step (and not in filters).

To crawl a website where some content has a sitemap and the rest does not:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import crawler
from core import utils

# Init a crawler object
cr = crawler.Crawler()

# Scrape one site using the sitemap
output = cr.get_website_from_sitemap("https://ansel.photos",
                                      "en",
                                      markup="article")

# Scrape another site recursively
output += cr.get_website_from_crawling("https://community.ansel.photos",
                                       "en",
                                       child="/discussions-home",
                                       markup=[("div", {"class": "bx-content-description"}),
                                               ("div", {"class": "cmt-body"})],
                                       contains_str="view-discussion")

# ... can keep appending as many websites as you want to `output` list

utils.save_data(output, "ansel")
```

!!! Warning

    In the above example, we reuse the `cr` object between the "sitemap" and the "recurse" calls. It means that the second call will inherit the [Crawler.crawled_URL][core.crawler.Crawler.crawled_URL] list from the previous, which contains all the URLs already processed. All URLs from this list will be ignored in the next calls. This can be good to avoid duplicates, but can be bad for some use cases. For those cases, instantiate a new `Crawler` object instead of reusing the previous one.


The [core.utils.save_data][] method will directly save the list of [core.crawler.web_page][] objects as a [pickle][] file compressed in a `.tar.gz` archive, into the `VirtualSecretary/data` folder. To re-open, decompress and decode it later, use [core.utils.open_data][]:

```python
# Boilerplate stuff to access src/core from src/user_scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Here starts the real code
from core import crawler
from core import utils

pages = utils.open_data("ansel")

for page in pages:
  # do stuff...
  print(page["title"], page["url"])
```
