# Crawling Pages

Whether you want to build a language model for an email classifier or a search engine for information retrieval, you will need to aggregate a corpus of text documents. Virtual Secretary has built-in methods to crawl HTML and PDF documents from websites, assemble them into a SQLite database, and remove duplicate documents to maintain a clean index.

## Getting Page Content

### Websites with Sitemaps

The easiest case is websites that publish a [sitemap](https://en.wikipedia.org/wiki/Sitemaps) — an XML file listing all pages the webmaster wants search engines to index. Most [CMS](https://en.wikipedia.org/wiki/Content_management_system) platforms include sitemap support out of the box. The usual location is `https://your-domain.com/sitemap.xml`. Sitemaps can be nested (a sitemap-of-sitemaps); both flat and nested structures are handled transparently by `Crawler.get_website_from_sitemap()`.

Crawl output should always go into a temporary database first (no primary-key constraint on `url`), be cleaned up and deduplicated, then promoted to the permanent store. See [Deduplication](#deduplication) for the full pattern.

In your `VirtualSecretary/src/user-scripts/` directory, create a new script called for example `scrape-ansel.py`, in which you can put:

```python
# Boilerplate stuff to call core package from user scripts
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import crawler, database, deduplicator, nlp, batching

tmp_db = database.create_temp_db()

with crawler.Crawler(delay=1.0) as cr:
    pages = cr.get_website_from_sitemap(
        website      = "https://ansel.photos",
        default_lang = "en",
        markup       = "article",
        category     = "reference",
    )

database.populate_db(tmp_db, pages)
```

The `default_lang` argument is used as a fallback when the HTML page does not declare a language. Language is later confirmed or overridden by machine-learning detection during text normalisation (`batch_parse_web_page`).

### The `markup` Parameter

Both crawling methods accept a `markup` argument that restricts content extraction to specific HTML elements, letting you capture article bodies while discarding sidebars, headers, and navigation menus:

```python
# Plain tag name
markup = "article"

# Tag + CSS attribute dict (BeautifulSoup find_all syntax)
markup = ("div", {"class": "post-content"})

# Tag + id
markup = ("div", {"id": "main-content"})

# Multiple selectors — content from all matches is concatenated in order
markup = [("div", {"id": "content"}), ("article", {"class": "entry"})]

# "body" / None — grab the whole body (suitable for documentation and reference pages)
markup = "body"
```

### Filtering with `contains_str`

The `contains_str` argument restricts which URLs get *added to the index*. Pages not matching the filter are still visited to discover new links, but their content is discarded:

```python
# Only index thread pages, skip user profiles and category archives
pages = cr.get_website_from_sitemap(
    website      = "https://discuss.pixls.us",
    default_lang = "en",
    markup       = ("div", {"class": "topic-body"}),
    contains_str = "/t/",        # also accepts a list: ["/t/", "/articles/"]
    category     = "forum",
)
```

### Extending Coverage with `internal_links`

By default the sitemap crawler only indexes pages listed in `sitemap.xml`. Set `internal_links="external"` to also follow and index every `<a>` link found in those pages content — useful when to also index referenced pages and PDFs outside of the current website:

```python
pages = cr.get_website_from_sitemap(
    website        = "https://aurelienpierre.com",
    default_lang   = "fr",
    markup         = ("div", {"class": "post-content"}),
    contains_str   = "/photography/",
    internal_links = "external",
    category       = "blog",
)
```

### Websites without Sitemaps

Many forum platforms and institutional websites do not publish a sitemap. Use `get_website_from_crawling()` to recursively follow links from an entry point instead:

```python
with crawler.Crawler(delay=1.5) as cr:
    pages = cr.get_website_from_crawling(
        website      = "https://community.ansel.photos",
        default_lang = "en",
        child        = "/discussions-home/",   # entry page within the domain
        markup       = [("div", {"class": "bx-content-description"}),
                        ("div", {"class": "cmt-body"})],
        contains_str = "/view-discussion/",
        category     = "forum",
    )
```

The `child` argument defines the starting page within the domain (defaults to `/`). Pages not matching `contains_str` are still visited to find new links, but their content is not indexed.

To prevent accidentally crawling beyond a specific section — essential on large sites — use `restrict_section=True` combined with `child`. Link-following will then stay within `website + child/*`:

```python
pages = cr.get_website_from_crawling(
    website           = "https://discuss.pixls.us",
    default_lang      = "en",
    child             = "/c/software/darktable",
    contains_str      = "/t/",
    markup            = ("div", {"class": "topic-body"}),
    max_recurse_level = -1,       # -1 = exhaustive, no depth limit
    restrict_section  = True,     # stay within /c/software/darktable/*
    category          = "forum",
)
```

### Combining Crawling Methods

A single `Crawler` instance tracks already-visited URLs across calls, so mixing sitemap and recursive crawling within one `with` block will never fetch the same URL twice, which makes it much more efficient for domains that use different CMS (for example, forum/community + blog) that often link against each other:

```python
tmp_db = database.create_temp_db()

with crawler.Crawler(delay=1.0) as cr:
    pages  = cr.get_website_from_sitemap("https://docs.darktable.org", "en",
                                          markup="body", category="docs")
    pages += cr.get_website_from_crawling("https://darktable.fr", "fr",
                                           child="/blog/", markup="article",
                                           category="blog")

database.populate_db(tmp_db, pages)
```

## Incremental crawling

All crawling methods — HTML, PDF, YouTube, and GitHub — share the same two-attribute incremental-update mechanism:

| Attribute | How to set | Effect |
|---|---|---|
| `cr.known_urls` | `cr.load_known_urls(db)` | Maps URL → last crawled `datetime` from an existing database |
| `cr.since` | `cr.since = datetime(…)` | Global cut-off: any URL in `known_urls` crawled on or after this datetime is skipped |

For sitemap crawling the page's own `<lastmod>` field takes precedence over `since`, so a page that was modified after the last crawl is re-fetched even if it was crawled recently. `since` is used as a fallback for sitemap entries that carry no `<lastmod>`.

```python
import datetime
from core import crawler, database

# Append new URLs to a temporary DB that supports
# duplicated URLs. Deduplication will be handled after (see below)
tmp_db = database.create_temp_db()

with crawler.Crawler(delay=1.0) as cr:
    # Populate the map once — applies to all crawling calls below:
    # Get all known URLs from a permament DB storage
    # where URLs are primary keys and therefore unique
    # or ensure to deduplicate a temporary database.
    db = database.open_db("my-engine.db")
    cr.load_known_urls(db)
    db.close()

    cr.since = datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc)

    # Skips sitemap entries whose <lastmod> <= stored crawl date
    pages  = cr.get_website_from_sitemap("https://ansel.photos", "en", markup="article")

    # Skips any URL crawled after cr.since
    pages += cr.get_website_from_crawling("https://community.ansel.photos", "en",
                                           child="/discussions-home/", contains_str="/view-discussion/")

    # Skips videos crawled after cr.since (no API-side filter needed)
    pages += cr.get_youtube_channels(
        channel_ids  = ["UCmsSn3fujI81EKEr4NLxrcg"],
        api_key      = "AIza…",
        since        = cr.since,
    )

    # Passes since to the GitHub API's ?since= param for issues/pulls/commits
    pages += cr.get_github_repositories(
        repositories = [("aurelienpierreeng", "ansel")],
        api_key      = "ghp_…",
        since        = cr.since,
    )

database.populate_db(tmp_db, pages)
tmp_db.close()
```

Recursively crawling websites that don't have a sitemap is expensive and can take ages. So you may want to re-crawl them only once every `n` months, which can be achieved automatically in your script with:

```python
import utils

# Recrawl everything older than 3 months
cr.since = utils.get_past_n_months(3)
```

Then you don't need to worry about updating dates manually.

## Mining PDF Documents

### Embedded in a Crawl

Pass `mine_pdf=True` to either crawling method and every `.pdf` link found on the pages will be automatically downloaded and extracted — no separate handling needed:

```python
pages = cr.get_website_from_sitemap(
    website      = "https://www.cie.co.at/publications",
    default_lang = "en",
    markup       = "body",
    mine_pdf     = True,
)
```

### Direct `get_pdf_content()` Call

For PDFs that are not reachable through a crawl — large reference books, local files, or document archives — use `pdf.get_pdf_content()` directly. It handles both remote URLs and local file paths, extracts the table of contents to split long documents by chapter, and falls back to Tesseract OCR for scanned images:

```python
from core.pdf import get_pdf_content
from core.network import DelayedClass

class SimpleDelay(DelayedClass):
    """Minimal rate-limiter required by the get_pdf_content API."""
    def __init__(self):
        self.delay = 0.5
        self.last_requests = {}
        self.domain_thresholds = {}
        self.main_domain = None
        self.robots_txt = None

delay = SimpleDelay()

# Remote PDF — split by table of contents into individual chapters
sections = get_pdf_content(
    url             = "https://colour-science.org/papers/luo_2001.pdf",
    lang            = "en",
    delay           = delay,
    process_outline = True,   # each ToC entry becomes its own web_page
    category        = "paper",
    ocr             = 1,      # use OCR only when no embedded text is found (default)
)

# Local scanned PDF — force full OCR
sections += get_pdf_content(
    url       = "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118653128",
    lang      = "en",
    delay     = delay,
    file_path = "/home/user/fairchild_color_appearance_models.pdf",
    ocr       = 2,       # force OCR even when embedded text is present
    max_size  = 50,      # skip files larger than 50 MiB
    max_pages = 800,
    repair    = 2,       # image pre-processing strength (0–3)
    upscale   = 3,       # upscaling factor before OCR
    contrast  = 1.4,
)

database.populate_db(tmp_db, sections)
```

The URL passed as the first argument is used as the canonical address stored in the index. When `process_outline=True`, each section's URL receives a `#page=n` fragment so deep-linking from search results works directly in Chrome and Acrobat Reader.

The `ocr` parameter controls when OCR is attempted:

| Value | Behaviour |
|---|---|
| `0` | Never — text extraction only |
| `1` | Only when no embedded text is found (default) |
| `2` | Always, even when embedded text is present |

### Custom PDF Handling

If you need different extraction logic — tables, multi-column layouts, layered content — subclass `Crawler` and override `_parse_pdf_content()`:

```python
from core import crawler, database
from core.types import web_page, sanitize_web_page
from io import BytesIO
import pytesseract, pdf2image, requests
from pypdf import PdfReader

def my_pdf_handler(url: str, lang: str, category: str = "") -> list[web_page]:
    try:
        if url.startswith("http"):
            resp = requests.get(url, timeout=30, allow_redirects=True)
            if resp.status_code != 200:
                return []
            document = BytesIO(resp.content)
        else:
            document = open(url, "rb")

        blob = document.read()
        reader = PdfReader(BytesIO(blob))

        # Page-by-page text extraction — replace with your own logic
        content = "\n".join(p.extract_text() or "" for p in reader.pages).strip()

        if not content:
            # No embedded text: fall back to OCR
            for image in pdf2image.convert_from_bytes(blob):
                content += pytesseract.image_to_string(image)

        if content:
            return [sanitize_web_page(web_page(
                title    = url.split("/")[-1],
                url      = url,
                content  = content,
                excerpt  = content[:300],
                lang     = lang,
                category = category,
                h1=[], h2=[], date="",
            ))]
    except Exception as e:
        print(e)
    return []


class MyCrawler(crawler.Crawler):
    def _parse_pdf_content(self, link, default_lang, category="") -> list[web_page]:
        return my_pdf_handler(link, default_lang, category=category)


with MyCrawler(delay=1.0) as cr:
    pages = cr.get_website_from_crawling("https://example.com", "en")

tmp_db = database.create_temp_db()
database.populate_db(tmp_db, pages)
```

---

## Custom Data Sources (REST APIs)

Some sites serve content entirely through client-side rendering: their HTML is an empty canvas populated by JavaScript at runtime, so a standard HTML parser sees nothing useful. The workaround is to query the underlying REST API directly and construct `web_page` objects manually.

### YouTube Example

```python
import requests, json
from core import database
from core.types import web_page, sanitize_web_page

API_KEY    = "..."   # https://developers.google.com/youtube/v3/getting-started
CHANNEL_ID = "UCmsSn3fujI81EKEr4NLxrcg"

response = requests.get(
    "https://youtube.googleapis.com/youtube/v3/search"
    f"?maxResults=1000&part=snippet&channelId={CHANNEL_ID}&type=video&key={API_KEY}"
)
items = json.loads(response.content)["items"]

pages = []
for item in items:
    vid_id  = item["id"]["videoId"]
    detail  = requests.get(
        f"https://youtube.googleapis.com/youtube/v3/videos?part=snippet&id={vid_id}&key={API_KEY}"
    )
    snippet = json.loads(detail.content)["items"][0]["snippet"]

    pages.append(sanitize_web_page(web_page(
        title    = snippet["title"],
        url      = f"https://www.youtube.com/watch?v={vid_id}",
        excerpt  = item["snippet"]["description"],
        content  = snippet["description"],
        date     = snippet["publishedAt"],
        lang     = snippet.get("defaultLanguage", "en"),
        category = "video",
        h1=[], h2=[],
    )))

tmp_db = database.create_temp_db()
database.populate_db(tmp_db, pages)
```

### GitHub Issues and Pull Requests

```python
import time, re, json, requests, markdown
from core import database
from core.types import web_page, sanitize_web_page

API_KEY = "..."  # https://docs.github.com/en/rest/authentication
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "application/vnd.github+json",
}
POSTS_PER_PAGE = 100   # GitHub maximum


def items_count(user: str, repo: str, feature: str) -> int:
    r = requests.get(
        f"https://api.github.com/repos/{user}/{repo}/{feature}?per_page=1",
        headers=HEADERS, timeout=30,
    )
    return int(re.search(r"\d+$", r.links["last"]["url"]).group())


def get_github_items(user: str, repo: str, feature: str, category: str) -> list[web_page]:
    total  = items_count(user, repo, feature)
    pages_ = total // POSTS_PER_PAGE + 1
    result = []
    for page in range(pages_):
        time.sleep(0.72)   # stay within GitHub's 5 000 req/hr limit
        r = requests.get(
            f"https://api.github.com/repos/{user}/{repo}/{feature}"
            f"?per_page={POSTS_PER_PAGE}&page={page}",
            headers=HEADERS, timeout=30,
        )
        for item in json.loads(r.content):
            url = item.get("html_url", "")
            if not url:
                continue
            body = markdown.markdown(item.get("body") or "")
            result.append(sanitize_web_page(web_page(
                title    = f"{feature.capitalize()}: {item.get('title', '')}",
                url      = url,
                content  = body,
                excerpt  = body[:300],
                date     = item.get("created_at", ""),
                lang     = "en",
                category = category,
                h1=[], h2=[],
            )))
    return result


pages = []
repos = [("aurelienpierreeng", "ansel"), ("darktable-org", "rawspeed")]
for user, repo in repos:
    for feature in ["issues", "pulls"]:
        pages += get_github_items(user, repo, feature, "Github")

tmp_db = database.create_temp_db()
database.populate_db(tmp_db, pages)
```

---

## Crawling Details

### robots.txt

The crawler respects [robots.txt](https://en.wikipedia.org/wiki/Robots.txt) automatically. For every new domain it visits it fetches `/robots.txt`, honours `Disallow` directives for its user-agent string, and reads `Crawl-delay` / `Request-rate` entries to set its per-domain inter-request throttle. If a `robots.txt` file cannot be fetched (domain unreachable, 404), crawling proceeds without restrictions.

Pages listed in `sitemap.xml` are assumed pre-authorised and skip the per-URL `robots.txt` check, reducing request overhead on large sitemaps.

Some servers implement Cloudflare-style bot blocking that goes beyond `robots.txt`. The crawler handles this by trying multiple user-agent / header combinations and falling back to [web.archive.org](https://web.archive.org) in last resort for pages that return persistent 403 or 404 errors.

### Cleaning Up Non-Language Content

Websites come with navigation menus, breadcrumbs, sidebars, and metadata sections that are not natural-language content and are not unique to any particular page. The HTML parser removes the following elements **before** text extraction:

`<style>`, `<script>`, `<svg>`, `<img>`, `<picture>`, `<audio>`, `<video>`, `<iframe>`, `<embed>`, `<aside>`, `<nav>`, `<input>`, `<header>`, `<button>`, `<form>`, `<fieldset>`, `<footer>`, `<summary>`, `<dialog>`, `<textarea>`, `<select>`, `<option>`

Inline `style` and `data` attributes are also stripped from all remaining elements.

If this is still not enough, use the `markup` argument to whitelist specific content containers — that is usually more reliable and faster than trying to remove noise after the fact.

!!! note "`<blockquote>` and `<code>` are intentionally kept"
    Unlike some older documentation may suggest, `<blockquote>`, `<code>`, and `<pre>` blocks are **not** stripped. Quoted replies in forum pages are a known source of near-duplicate content, but that is handled by the deduplicator rather than at parse time.

### The `no_follow` List

Both crawling methods share a default blocklist of URLs that are never fetched — social-media share links, login and signup pages, cart pages, and user-profile paths. You can extend it at instantiation time or by appending to the attribute:

```python
# Via the constructor (preferred — applies before any crawling starts)
with crawler.Crawler(
    delay     = 1.0,
    no_follow = [
        "google.com",
        "amazon.com",
        "/tag/",           # tag archive pages anywhere
        "?replytocom=",    # WordPress comment reply links
        ".pdf",            # skip PDFs during an HTML-only crawl
    ],
) as cr:
    ...

    # Appending after instantiation (same effect)
    cr.no_follow += ["/view-album/", "persons-profile-"]
```

`no_follow` entries are substring-matched against the full URL. Any match causes the URL to be **silently discarded — it generates no network request at all**. This is more aggressive than `contains_str`, which still visits non-matching URLs to discover new links.

### Caveats

The crawler tries to ignore JSON, CSS, and JavaScript files based on MIME type and file extension. This does not work 100%, particularly on GitHub where declared MIME types can be inaccurate and code is sometimes embedded in HTML in unusual ways. Use the SQL filtering step in [Deduplication](#content-filtering) to clean up any machine-parseable content that slips through.

---

## Deduplication

### The Two-Database Pattern

Virtual Secretary uses two complementary database types throughout the pipeline:

| Function | Primary key on `url`? | Use case | Target folder |
|---|---|---|---|
| `database.create_temp_db()` | No | Raw crawl buffer; tolerates duplicate URLs; lives in a temp directory | `~/.virtual-secretary` |
| `database.create_db(name)` | **Yes** | Permanent store; each URL is unique; used for all NLP and serving | `VirtualSecretary/models` |

Crawl output **always** goes into a temp database first. After content normalisation, filtering, and deduplication, the cleaned data is promoted to the permanent store with `database.import_pages()`. Writing directly from a multi-source crawl into `create_db` produces undefined behaviour when two sources contain the same URL, because the `ON CONFLICT` handler cannot determine which copy's content fields should win.

[`database.create_db`][core.database.create_db] is meant to prepare reusable models, for example to feed to [`nlp.Word2Vec` training][core.nlp.Word2Vec] or [`search.Indexer` web index][core.search.Indexer]. [`database.create_temp_db`][core.database.create_temp_db] is meant to be saved to less storage-hungry datasets using [`utils.save_data`][core.utils.save_data].

### Content Filtering

Before deduplication runs, remove pages that should never appear in the index. This is most efficiently done at the SQL level directly on the temp database, after `batch_parse_web_page` has run (which populates `lang`):

```python
from core import batching, nlp

# Detect languages and write the `parsed` column — required before any filtering on `lang`
batching.batch_parse_web_page(tmp_db, nlp.Tokenizer())

cur = tmp_db.cursor()

# Remove pages in languages your NLP pipeline doesn't support
cur.execute("DELETE FROM pages WHERE lang NOT IN ('fr', 'en') OR lang IS NULL")

# Remove GitHub blob viewer pages (raw code, no natural-language content)
cur.execute("DELETE FROM pages WHERE url LIKE ?", ("%/blob/%",))

# Remove Discourse boilerplate session-alert pages
cur.execute("DELETE FROM pages WHERE content LIKE ?",
            ("%you signed in with another tab or window%",))

# Remove a domain superseded without redirects
cur.execute("DELETE FROM pages WHERE url LIKE ?", ("%old-domain.example.com%",))

tmp_db.commit()
```

Alternatively, filter at import time using `import_pages(where_clause=...)` to avoid even inserting unwanted rows:

```python
database.import_pages(
    source_db      = pixls_db,
    destination_db = tmp_db,
    where_clause   = "title NOT LIKE ? AND content NOT LIKE ?",
    params         = ("%playraw%", "%sigmoid%"),
)
```

### Running Deduplication

`Deduplicator` compares the `parsed` (normalised) column of every pair of pages and retains the best copy — shortest URL, most recent date, or longest content when dates are equal. `batch_parse_web_page` must have run first, since that is what writes the `parsed` column.

```python
from core import deduplicator

dedup = deduplicator.Deduplicator(
    threshold      = 1.0,    # 1.0 = exact duplicates only
                             # < 1.0 enables Levenshtein near-duplicate detection
    distance       = 50,     # minimum tokens needed to compare two pages
    discard_params = False,  # keep URL query parameters in the canonical URL
    fix_urls       = False,
)

# Domains always discarded
dedup.urls_to_ignore += [
    "translate.goog",    # machine translations — keep canonical instead
    "flickr.com",
    "facebook.com",
]

dedup(tmp_db)
```

Near-duplicate detection (`threshold < 1.0`) is more expensive. For a corpus of hundreds of thousands of pages, start with `threshold=1.0` and tighten it only if exact matching leaves obvious near-duplicates in the index.

## Saving the dataset for later reuse

Datasets are saved into the `VirtualSecretary/data` folder.

```python
from core import utils, database

tmp_db = database.create_temp_db()

...

# Run SQLite VACCUUM
database.compress_db(tmp_db)

# Save as gzipped SQLite dump
utils.save_data(tmp_db, "my-dataset")

# Clean-up the temporary databases
# WARNING: this removes all existing databases,
# ensure another thread is not running that needs
# one of those.
database.close_db(tmp_db)
database.cleanup_temp_db()
```

To re-open the database as it was saved, use `tmp_db = utils.open_data("my-dataset")` and you will get the SQLite database opened in memory. To copy it into another opened database, you may use:

```python
from core import utils, database

# Open saved dataset
tmp_db = utils.open_data("my-dataset")

# Create a new permanent database
final_db = database.create_db("my-engine.db")

# Dump the memory-hosted dataset DB
tmp_db.backup(final_db)

# Close memory DB
tmp_db.close()

# Optimize the permament DB and close it
database.close_db(final_db)
```

### Full Pipeline Example

Putting all steps together for a multi-source crawl:

```python
from core import crawler, database, deduplicator, nlp, batching, utils

filename = "photo-websites"

tmp_db = database.create_temp_db()

sources = [
    ("https://ansel.photos",     "en", "article",  "docs"),
    ("https://darktable.fr",     "fr", "article",  "blog"),
    ("https://discuss.pixls.us", "en", ("div", {"class": "topic-body"}), "forum"),
]

# 1. Crawl all sources into the temp database
for site, lang, markup, category in sources:
    with crawler.Crawler(delay=1.0) as cr:
        pages = cr.get_website_from_sitemap(site, lang, markup=markup, category=category)
    database.populate_db(tmp_db, pages)

# 2. Normalise text and detect language
batching.batch_parse_web_page(tmp_db, nlp.Tokenizer())

# 3. Drop unsupported languages and known noise
tmp_db.execute("DELETE FROM pages WHERE lang NOT IN ('fr', 'en') OR lang IS NULL")
tmp_db.execute("DELETE FROM pages WHERE url LIKE ?", ("%/blob/%",))
tmp_db.commit()

# 4. Deduplicate while the temp DB has no primary-key constraint on url
dedup = deduplicator.Deduplicator(threshold=1.0, discard_params=False, fix_urls=False)
dedup.urls_to_ignore += ["translate.goog", "facebook.com", "flickr.com"]
dedup(tmp_db)

# 5. Compress and save. 2 options:
save_as = "data"

if save_as == "data":

    # Save to the data folder, 
    # as VirtualSecretary/data/photo-websites.sql.tar.gz
    # This takes less storage space but loads slower
    # on utils.open_data()
    database.compress_db(tmp_db)
    utils.save_data(tmp_db, filename)

elif save_as == "model":

    # Save to the models folder,
    # as VirtualSecretary/models/my-engine.db
    # This takes (a lot) more storage space,
    # but is a directly-usable SQLite database (loads very fast)
    final_db = database.create_db("my-engine.db")
    database.import_pages(source_db=tmp_db, destination_db=final_db)
    database.close_db(final_db) 
    # this optimizes final_db before calling final_db.close()
    # calling final_db.close() directly is ok too.

tmp_db.close()
database.cleanup_temp_db()

```