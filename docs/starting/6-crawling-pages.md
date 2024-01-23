# Crawling pages

Whether you want to build a language model to be used in an email classifier or you want to build an indexer to be used as a search engine for information retrieval, you will need to aggregate a corpus of text files. Virtual Secretary has out-of-the-box methods to crawl HTML and PDF documents frome websites, and then remove duplicate documents to maintain clean indexes.

## Getting pages content

### Websites with sitemaps

The easiest case is websites which have a [sitemap](https://en.wikipedia.org/wiki/Sitemaps). The sitemap is an XML file listing all the pages that the webmaster wants search engine to index, so it is clean, simple, and yields little noise. Most [CMS](https://en.wikipedia.org/wiki/Content_management_system) have core options or plugins allowing to declare a sitemap, though many institutional websites using (old) custom-made CMS don't. The usual place for sitemaps is at the root of the website, like `https://your-domain.com/sitemap.xml`. Sitemaps can be recursive, so the main sitemap is actually a sitemap of sitemaps, or not. Both cases are supported by [core.crawler.Crawler.get_website_from_sitemap][].

In your `VirtualSecretary/src/user-scripts/` directory, create a new script called for example `scrape-ansel.py`, in which you can put:

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Actual code
from core import crawler
from core import utils

# Scrape Ansel
cr = crawler.Crawler()
output = cr.get_website_from_sitemap("https://ansel.photos",
                                      sitemap="/sitemap.xml",
                                      default_lang="en",
                                      langs=("en", "fr"),
                                      markup="article",
                                      category="reference")
utils.save_data(output, "ansel")
```

This will produce a list of [core.crawler.web_page][] dictionnaries, which describe a webpage (or any document accessible from an URI/URL) in an uniform way that is understood by the rest of the application.

[core.utils.save_data][] will handle the saving to `VirtualSecretary/data` folder and the serialization into a Python [pickle][] object, itself compressed into a `tar.gz` archive to save space. To reuse the dataset later, the convenience function [core.utils.open_data][] can be used with the same name and will restore the list of [core.crawler.web_page][], taking care of decompressing and de-serializing the file. You can just use `utils.save_data("name")` and `utils.open_data("name")` and forget about files I/O altogether.

The `default_lang` argument is purely declarative and is used in case the HTML page doesn't declare any language. The `langs` argument is the tuple of language codes for which you want to follow the alternative versions (translated) of the page too, as declared in HTML headers like `<link rel="alternate" hreflang="fr">`. This will produce a multilingual dataset of pages, where the language of each page is stored in the `lang` key of the [core.crawler.web_page][] dictionnary, should you need it at implementation time.

The sitemap crawler follows every link (`<a>` HTML markup) found in the page and indexes it too. This is useful to aggregate PDF files and reference pages (Wikipedia, other websites) linked from the page body and therefore possibly semantically linked to the topic of the page (which will help generalizing the AI language model if that's your goal).


### Websites without sitemaps

A lot of institutional websites, as well as forum CMS, don't use sitemaps, which creates a much harder task to follow relevant informations while avoiding noise. As far as natural language goes, many types of webpages are irrelevant:

- posts, categories and tags archives pages (typical in blogs),
- login, signup, subscribe pages,
- cart and shop pages,
- user profiles,
- any technical page.

These will be a challenge to keep out of our index if we need to crawl websites recursively. Here is a full example of user script crawling a website recursively:

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Actual code
from core import crawler
from core import utils

cr = crawler.Crawler()
output = cr.get_website_from_crawling("https://community.ansel.photos",
                                      "en",
                                      child="/discussions-home/",
                                      markup=[("div", {"class": "bx-content-description"}),
                                              ("div", {"class": "cmt-body"})],
                                      contains_str="/view-discussion/",
                                      category="forum")

utils.save_data(output, "ansel")
```

The `child` argument defines what our entry page is on the website, that is what we use as an index. It is usually good to use some sort of archive page. If not defined, it defaults to `/`, the root of the website.

Then, you may want to restrict the indexed content to a section of the website, using the `contains_str` argument: only URLs containing `/view-discussion/` will be indexed here, though all pages will be crawled for new links. This argument accepts also lists of `str` if several sections need indexing.


### Pages served from asynchronous Rest API calls

Some websites are not serving (x)HTML content anymore, but only a blank HTML canvas. The actual content is fetched from a [Rest API](https://fr.wikipedia.org/wiki/Representational_state_transfer) and the markup rendering is done client-side from a JSON toolkit (like [React](https://react.dev/)). This makes sense for website serving highly-personnalized content (React coming from Facebook is telling), but gets in the way when it's implemented on more informational websites because CLI XML parsers will not be able to render the asynchronously-loaded and server-side-rendered content, so they will only see the blank HTML canvas.

The silver lining though is we can tap directly into the Rest API and the JSON responses it produces, which is much cleaner than parsing (x)HTML when it comes to isolating actual content from navigation menues, metadata and so on. The ugly part is we need to write one script per API, so one per website.

#### Youtube example

Here is a full example of user script to extract title, description and date from YouTube videos through the YouTube data API v3.

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Actual code
import requests
import json

from core import crawler
from core import utils

API_KEY = ... # Get yours : https://developers.google.com/youtube/v3/getting-started
CHANNEL_ID = "UCmsSn3fujI81EKEr4NLxrcg"

videos = requests.get(f"https://youtube.googleapis.com/youtube/v3/search?maxResults=1000&part=snippet&channelId={CHANNEL_ID}&type=video&key={API_KEY}")
videos = json.loads(videos.content)["items"]

videos_list = []

for item in videos:
    id = item["id"]["videoId"]
    content = requests.get(f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet&id={id}&key={API_KEY}")
    content = json.loads(content.content)["items"][0]["snippet"]
    result = crawler.web_page(title=content["title"],
                              url="https://www.youtube.com/watch?v=" + id,
                              excerpt=item["snippet"]["description"],
                              content=content["description"],
                              date=content["publishedAt"],
                              h1={},
                              h2={},
                              lang=content["defaultLanguage"] if "defaultLanguage" in content else "en",
                              category="video")
    videos_list.append(result)

utils.save_data(videos_list, "youtube")
```

#### Github example

Here is a full user script to extract Github issues and pull requests from a list of repositories:

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Actual code
import time
import re
import requests
import json
import markdown

from core import crawler
from core import utils

API_KEY = ... # Get yours: https://docs.github.com/en/rest/authentication/authenticating-to-the-rest-api?apiVersion=2022-11-28
headers={"Authorization": "Bearer %s" % API_KEY,
         "Accept": "application/vnd.github+json"}

# Instanciate a crawler to avoid following links more than once
cr = crawler.Crawler()
POSTS_PER_PAGE = 100 # Github max


def append_str(string:str):
    """Github uses Markdown, our parser needs (x)HTML : Convert."""

    if string is not None:
        return markdown.markdown(string) + "\n\n"
    else:
        return ""


def get_github_item(c: crawler.Crawler, user: str, repo: str, feature: str, page: int, title_prepend: str, category: str):

    # Github throttles API access to 5000 requests/hour.
    # We deal with that by using a timeout of 0.72s between requests.
    time.sleep(0.72)

    url = f"https://api.github.com/repos/{user}/{repo}/{feature}?per_page={POSTS_PER_PAGE}&page={page}"
    output = []
    response = requests.get(url, headers=headers, timeout=30)

    for item in json.loads(response.content):
        content = ""
        title = ""
        date = None
        current_url = ""

        if "html_url" in item:
            current_url = item["html_url"]
        else:
            continue

        # Store current URL in memory list to be sure we don't recrawl it from internal links later
        cr.crawled_URL.append(current_url)

        # Get generic item body and title
        if "body" in item:
            content += append_str(item["body"])

        if "title" in item:
            title = item["title"]

        if "created_at" in item:
            date = item["created_at"]

        # Build a fake HTML page to connect with generic parsing API
        content = "<title>" + title_prepend + title + "</title><body>" + content + "</body>"

        # Follow internal links and index PDF/HTML
        entry = crawler.get_page_content(None, content)
        output += cr.get_immediate_links(entry, "github.com", current_url, "en", ["en", "fr"], "unknown", "")

        # Parse content
        output += crawler.parse_page(entry, current_url, "en", "body", date, category)

    return output


def items_count(user, repo, feature):
    """Count the current number of objects (issues, commits, pull requests)"""

    url = f'https://api.github.com/repos/{user}/{repo}/{feature}?per_page=1'
    response = requests.get(url, headers=headers, timeout=30)
    links = response.links['last']['url']
    return int(re.search(r'\d+$', links).group())


# could parse commits, discussions, project kanban bourds, etc.
# but their JSON needs special handling
features = ["issues", "pulls"]

# tuples of (user, repository)
projects = [("aurelienpierreeng", "ansel"),
            ("darktable-org", "rawspeed")]

# Aggregate pages
results = []
for project in projects:
    for feature in features:
      num_items = items_count(project[0], project[1], feature)
      num_pages = num_items // POSTS_PER_PAGE + 1
      for page in range(num_pages):
          results += get_github_item(cr,
                                     project[0],
                                     project[1],
                                     feature,
                                     page,
                                     feature.capitalize() + ": ",
                                     feature)

utils.save_data(results, "github")
```

## Details on crawling

Wether you use the recursive or the sitemap-based crawling, there are several things you need to be aware of.

### robots.txt

None of the crawling methods use [robots.txt](https://en.wikipedia.org/wiki/Robots.txt) files, which webmasters may use to forbid access to certain pages or to throttle certain user agents (like crawling bots like ourselves).

Some servers may block bots user-agents that try to access the forbidden pages, or simply serve a captcha blocking access to content (especially Amazon, Google and YouTube). Given that the goal here is to collect natural language (that is, descriptive and long-enough samples of meaningful text), this is a non-issue for us.

### Cleaning up non-language

Websites typically come with navigation menues, breadcrumbs links, sidebars containing recent posts and recent comments (or advertising), metadata section containing categories, tags, date, time, author, etc. Those are not "natural language" per-se, and they are often not unique to a certain page, so they should be discarded from any natural language processing.

For this purpose, the HTML parser removes the following HTML markup before indexing the page content:

- `<code>`,
- `<pre>`,
- `<math>`,
- `<style>`,
- `<script>`,
- `<svg>`,
- `<img>`,
- `<picture>`,
- `<audio>`,
- `<video>`,
- `<iframe>`,
- `<embed>`,
- `<blockquote>`,
- `<quote>`,
- `<aside>`,
- `<nav>`
- `<header>`,
- `<footer>`,
- `<button>`,
- `<form>`,
- `<input>`,
- `<dialog>`,
- `<textarea>`,
- `<select>`,
- `<option>`,
- `<fieldset>`,
- `<summary>`.

This might still not be enough, so both crawling techniques have a `markup` argument that can be used to restrict the parsing to one or several HTML tags, with or without CSS selectors.

By default, `Crawler.get_website_from_sitemap()` and `Crawler.get_website_from_crawling()` capture the whole content of the `<body>` tag. Here are some examples about how to restrict the parsing to predefined markup:

- parse only the content of the `<article>` tag: 
```python
get_website_from_xxx(markup="article")
```
- parse only the content of the `div.post-content` CSS selector:
```python
get_website_from_xxx(markup=("div", {"class": "post-content"})
```
- parse the content of the `article.entry` and `div#content` CSS selectors:
```python
get_website_from_xxx(markup=[("div",     {"id": "content"}),
                             ("article", {"class": "entry"})])
```

### Indexing PDF

PDF files are originally meant to exchange printable documents between applications and printer drivers. Because they are read-only and largely supported across applications and OSes, their use has spread to any read-only publication, from standards and specifications, to invoices, through scientific papers, reports, thesis, etc. Problem is: they were never intended to contain indexable content, beyond the basic keyword search in Acrobat Reader.

PDF come in many shapes and many flavours, which is a problem for us. Virtual Secretary supports only two of them:

- continuous text: single layer and single column or double-column of single-flow text, that is mostly PDF prepared with LaTeX and Microsoft® Word® without using layers,
- scanned papers: images of single-column and double-column text.

Both cases are treated automatically if they are found within a web page crawled from one of the 2 crawling methods above. Other cases involving layers of text boxes (stacked in the depth axis) cannot reliably be automatically processed because the order of the text in the document content flow does not necessarily reflect the layout styling. Tables of figures (invoices, spreadsheets) are not reliable as well, as text and figures could come out in row-major order or in column-major order, or in any random mix of both.

For the first case, we simply extract the text. If an outline is found (producing a table of contents in most PDF readers), we extract it too and index the PDF content section by section: each section gets its own [core.crawler.web_page][] element in the output list, and its own URL where we append the anchor `#page=n`, where `n` is the page of the beginning of the section. Such URLs (like `https://your-domain.com/file.pdf#page=3`) are understood by Google Chrome and Adobe Acrobat Reader, which will open the PDF file directly at the defined page. This feature is designed with books and long reports in mind.

For the second case, we perform a pass of image processing, filling "voids in ink" (that happens a lot when cheap-printing text on office paper), denoising and sharpening, then perform OCR using [Tesseract](https://github.com/tesseract-ocr/tesseract) configured for French, English and equations. This works fairly well for most documents but produces bad results for small characters on badly digitized files.

If those options don't fit your use case, you can re-implement your own PDF handling with minor adjustments. [Inherit](https://docs.python.org/3/tutorial/classes.html#inheritance) the [core.crawler.Crawler][] class and re-implement the `core.crawler.Crawler._parse_pdf_content()` method. By default, it uses [core.crawler.get_pdf_content][] which you can use as a model. For example:

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Actual code
from core import crawler
from core import utils

import pytesseract
import cv2
import pdf2image
import requests

from pypdf import PdfReader

def your_custom_pdf_handler(link: str, default_lang: str, category: str ="") -> list[crawler.web_page]:
    try:
        document : BytesIO
        if "http" in link:
            # if link is a network URL
            page = requests.get(link, timeout=30, allow_redirects=True)
            print(f"{url}: {page.status_code}")

            if page.status_code != 200:
                print("couldn't download %s" % url)
                return []

            document = BytesIO(page.content)
        else:
            # if link is a local path
            document = open(file_path, "rb")

        blob = document.read() # need to backup PDF content here because PdfReader kills it next

        # decode PDF content
        reader = PdfReader(document)

        # do your own stuff here.
        # For exemple, dummy page-by-page text reading:
        content = "\n".join([page.extract_text() for page in reader.pages]).strip("\n ")

        if not content:
            # If no text found, we probably have an image. Try OCR on blob
            for image in pdf2image.convert_from_bytes(blob):
                content += pytesseract.image_to_string(image)

        if content:
            result = utils.web_page(title=...,
                                    url=...,
                                    date=...,
                                    content=content,
                                    excerpt=...,
                                    h1={},
                                    h2={},
                                    lang=default_lang,
                                    category=category)
            return [result]

    except Exception as e:
        print(e)

    return []


class MyCrawler(crawler.Crawler):
    # Define your own child class

    # Boiler plate init stuff, nothing magical: pass everything through
    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

    # You only want to re-implement this:
    def _parse_pdf_content(self, link, default_lang, category="") -> list[crawler.web_page]:
        return your_custom_pdf_handler(link, default_lang, category=category)


# From there, it's just like using the native crawler.Crawler() class
cr = MyCrawler()
output = cr.get_website_from_crawling(...)
utils.save_data(output, "website")
```

You can also parse PDF files from your local filesystem, here is a full example for an user script:

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import crawler
from core import utils

output = crawler.get_pdf_content(
    "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118653128",
    "en",
    file_path="/home/user/fairchild_color_appearance_models_2013.pdf",
    category="reference")

utils.save_data(output, "fairchild")
```

The URL passed as argument will be used as a publicly-retrievable URI for your search-engine application. It is purely informational here and can also be a local network address or a filesystem path. The `file_path` argument needs to point to the real local PDF file that will be actually parsed. If `file_path` is not provided, a GET HTTP request is done against the URL (first argument of the function), which will need to be freely accessible.

### List of no-follow URLs

Both methods of crawling (recursive and sitemap-based) have a default [core.crawler.Crawler.no_follow][] list containing social sharing URLs and typical login/signup/profile/member handles. While crawling websites, any URL containing any of the strings from the `Crawler.no_follow` list will be entirely discarded, which means it will not be accessed at all.

This is different from the `contains_str` argument, first of all because `contains_str` defines what to __exclusively__ include, but also because the pages that don't match the criteria of the `contains_str` will still be parsed to find new links (but their content will not be added to the index), meaning they will generate network traffic. URLs matching anything from the `no_follow` list generate no traffic.

You can extend or overwrite the `no_follow` list after instanciating the object and before starting the crawling:

```python
cr = crawler.Crawler()
cr.no_follow += [
  "google.com",
  "amazon.com",
  "persons-profile-",
  "/view-album/",
]
output = cr.get_website_from_sitemap(...)
```

### Caveats

The crawler tries its best to ignore JSON, CSS and Javascript files (relying on MIME type and file extension), as well as inline JS and CSS code (removing `<style>` and `<script>` tags). That still doesn't work 100%, especially on Github, because the content MIME types declared in HTML headers is not always accurate and because they are sometimes weirdly embedded in pages.

For JSON files, you may want to try a `json.loads()` on the content of the [core.crawler.web_app][] elements and use them only if it raises a type exception (which means it's not a valid JSON file).

## Removing duplicates

The [core.crawler.Crawler][] instances have a list of already-crawled URLs, in [core.crawler.Crawler.crawled_URL][]. Those store URL as they are found in the `<a href="...">` tags of the page (including anchors and URL parameters), and you can append manually to the list (as seen above in the Github example). Any new URL found in a page that matches exactly an URL from that list will not be crawled again.

Because URLs are stored with anchors at crawling stage, some pages may be indexed several times under slightly different URLs. Before training a language model or building a search engine index, it might be worth it to ensure uniqueness of the corpus elements, for performance reason and to avoid biaising the AI model with some content. This is done with the [core.crawler.Deduplicator][] class.

```python
# Boilerplate stuff
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# Actual code
from core import crawler
from core import utils

cr = crawler.Crawler()
cr.no_follow += [
  "persons-profile-",
  "/view-album/",
]
output = cr.get_website_from_sitemap("https://ansel.photos",
                                     "en",
                                     markup="article",
                                     category="reference")

dedup = crawler.Deduplicator()
dedup.urls_to_ignore += [
  "aurelienpierre.com",
]
output = dedup.process(output)

utils.save_data(output, "ansel")
```

The deduplication ensure unicity of canonical URLs (without parameters and anchors, except for `#page=n`, used to index PDF sections, and `?lang=` used for ugly translated pages), and of content. By default, it also performs a near-duplicate detection, using the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), and contents having a distance ratio above 0.9 is factorized. This near-duplicate detection is quite expensive and can be bypassed on large dataset by calling `crawler.Deduplicator(threshold=1.0)`.

When (near-)duplicates are found, the most recent duplicate is always kept, otherwise (if dates are equal or if no duplicate has a date), the longest content is kept.
