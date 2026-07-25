# Crawling Pages

Whether you want to build a language model for an email classifier or a search engine for information retrieval, you will need to aggregate a corpus of text documents. Virtual Secretary provides methods to crawl HTML and PDF documents from websites, assemble them into a single SQLite database, and remove duplicate documents as it goes to keep a clean index.

## The DB-native crawl model

The crawler is **DB-native**: instead of returning a list of pages that you then save, it writes each page straight into **one corpus database** and deduplicates as it writes. A crawl has three moves — attach a database, fetch, finalise — expressed with [`begin_dataset`][core.crawler.Crawler.begin_dataset] / `get_*` / [`commit_dataset`][core.crawler.Crawler.commit_dataset]:

```python
# Boilerplate so a user script can import the core package
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from core import crawler, database

# One corpus database. `url_primary_key=False`: the crawler writes into it directly,
# the same URL may be captured several ways, and duplication is resolved by content
# rather than enforced by the schema.
db = database.create_db("corpus.db", url_primary_key=False)

with crawler.Crawler(delay=1.0) as cr:
    cr.begin_dataset(db, "ansel")          # attach the DB + compute the crawl-since watermark
    cr.get_website_from_sitemap(
        website      = "https://ansel.photos",
        default_lang = "en",
        markup       = "article",
        category     = "reference",
    )
    cr.commit_dataset("ansel")             # flush writes, normalise, deduplicate the delta
```

- The `"ansel"` string is a **dataset tag** written on every page this call produces (its provenance). A page shared by several sources keeps all their tags.
- `begin_dataset` reads the existing rows for that tag to figure out what to skip — crawls are **incremental by default** (see below).
- `commit_dataset` flushes the buffered writes, runs the crawl-time NLP prep, and deduplicates just the rows this run touched.

The `default_lang` argument is a fallback used when a page does not declare a language; the real language is confirmed by machine-learning detection during normalisation.

!!! tip "Live demo"
    The photography-domain search engine at **[chantal.aurelienpierre.com](https://chantal.aurelienpierre.com)** is built on these APIs. Its own crawl jobs, orchestration scripts, and app code are a private reference implementation; this guide reproduces the same patterns with the public library.

---

## Getting page content

### Websites with sitemaps

The easiest case is a website that publishes a [sitemap](https://en.wikipedia.org/wiki/Sitemaps). The usual location is `https://your-domain.com/sitemap.xml`; nested sitemaps (a sitemap-of-sitemaps) are handled transparently.

### The `markup` parameter

Both crawling methods accept a `markup` argument that restricts content extraction to specific HTML elements, so you capture the article body and discard sidebars, headers, and navigation:

```python
markup = "article"                                   # plain tag name
markup = ("div", {"class": "post-content"})          # tag + CSS attribute dict
markup = ("div", {"id": "main-content"})             # tag + id
markup = [("div", {"id": "content"}),                # several selectors,
          ("article", {"class": "entry"})]           #   concatenated in order
markup = "body"                                      # whole <body> (docs / reference pages)
```

### Filtering with `contains_str`

`contains_str` restricts which URLs are *indexed*. Non-matching pages are still visited to discover new links, but their content is discarded:

```python
cr.get_website_from_sitemap(
    website      = "https://discuss.pixls.us",
    default_lang = "en",
    markup       = ("div", {"class": "topic-body"}),
    contains_str = "/t/",          # also accepts a list: ["/t/", "/articles/"]
    category     = "forum",
)
```

### Extending coverage with `internal_links`

By default the sitemap crawler only indexes pages listed in `sitemap.xml`. Set `internal_links="external"` to also follow and index every `<a>` link found in the pages' content — useful to pull in referenced pages and PDFs outside the current site:

```python
cr.get_website_from_sitemap(
    website        = "https://aurelienpierre.com",
    default_lang   = "fr",
    markup         = ("div", {"class": "post-content"}),
    contains_str   = "/photography/",
    internal_links = "external",
    category       = "blog",
)
```

### Websites without a sitemap

Use [`get_website_from_crawling`][core.crawler.Crawler.get_website_from_crawling] to recursively follow links from an entry point:

```python
cr.get_website_from_crawling(
    website      = "https://community.ansel.photos",
    default_lang = "en",
    child        = "/discussions-home/",     # entry page within the domain (default "/")
    markup       = [("div", {"class": "bx-content-description"}),
                    ("div", {"class": "cmt-body"})],
    contains_str = "/view-discussion/",
    category     = "forum",
)
```

To keep a recursive crawl inside one section of a large site, combine `child` with `restrict_section=True` — link-following then stays within `website + child/*`:

```python
cr.get_website_from_crawling(
    website           = "https://discuss.pixls.us",
    default_lang      = "en",
    child             = "/c/software/darktable",
    contains_str      = "/t/",
    markup            = ("div", {"class": "topic-body"}),
    max_recurse_level = -1,      # -1 = exhaustive, no depth limit
    restrict_section  = True,    # stay within /c/software/darktable/*
    category          = "forum",
)
```

### Combining methods in one crawl

A single `Crawler` tracks already-visited URLs across calls, so mixing sitemap and recursive crawling between one `begin_dataset` and `commit_dataset` never fetches the same URL twice — handy for a domain that runs a forum and a blog on different CMS that link to each other:

```python
with crawler.Crawler(delay=1.0) as cr:
    cr.begin_dataset(db, "darktable")
    cr.get_website_from_sitemap("https://docs.darktable.org", "en",
                                markup="body", category="docs")
    cr.get_website_from_crawling("https://darktable.fr", "fr",
                                 child="/blog/", markup="article", category="blog")
    cr.commit_dataset("darktable")
```

---

## Incremental crawling

Incremental updates are **automatic**. Every write stamps the page with `crawled = now` and its `dataset` tag, so `begin_dataset` computes a per-dataset watermark — `min(latest item date, latest crawl time)` for that tag's own rows — and only re-fetches what changed since.

The crawler exposes two attributes it fills for you, which you can also set by hand for a one-off:

| Attribute | Effect |
|---|---|
| `cr.since` | Global cut-off: URLs already crawled at/after this datetime are skipped |
| `cr.known_urls` | Map URL → last-crawled datetime, preloaded for this dataset |

For sitemap crawling the page's own `<lastmod>` takes precedence over `since`, so a page modified after the last crawl is re-fetched even if it was crawled recently; `since` is the fallback for entries with no `<lastmod>`. The REST-API sources use `since` too — GitHub passes it to the API's `?since=`, Stack Exchange to `fromdate`, YouTube filters uploads client-side.

To force a re-crawl window by hand (e.g. re-visit an expensive recursive site only every few months):

```python
from core import utils

cr.since = utils.get_past_n_months(3)     # re-crawl anything older than 3 months
```

---

## Mining PDF documents

### Embedded in a crawl

Pass `mine_pdf=True` and every `.pdf` link found on the crawled pages is downloaded and its text extracted — no separate handling:

```python
cr.get_website_from_sitemap("https://www.cie.co.at/publications", "en",
                            markup="body", mine_pdf=True)
```

### A single PDF (remote or local)

For a PDF that is not reachable through a crawl — a large reference book, a local file — call [`get_pdf`][core.crawler.Crawler.get_pdf]. It handles both remote URLs and local file paths, splits long documents by their table of contents (one page per chapter, with a `#page=n` anchor so deep links open at the right page), and falls back to Tesseract OCR when a page has no embedded text:

```python
with crawler.Crawler(delay=1.0) as cr:
    cr.begin_dataset(db, "fairchild")
    cr.get_pdf(
        "https://onlinelibrary.wiley.com/doi/book/10.1002/9781118653128",  # canonical URL stored
        "en",
        file_path = "/home/user/Fairchild_Color_Appearance_Models.pdf",     # read from disk
        category  = "reference",
    )
    cr.commit_dataset("fairchild")
```

The first argument is always the canonical address stored in the index, even when the bytes come from `file_path`. The `ocr` parameter controls OCR: `0` never, `1` only when no embedded text is found (default), `2` always.

---

## Crawling from a REST API

Some sites serve content through client-side rendering or behind APIs a plain HTML crawler cannot reach. Three dedicated methods build pages from those APIs; all integrate with the incremental machinery.

### YouTube channels

Walks each channel's uploads playlist via the YouTube Data API v3, one page per video (description = indexable content).

```python
cr.get_youtube_channels(
    channel_ids  = ["UCmsSn3fujI81EKEr4NLxrcg"],
    api_key      = "YOUR_GOOGLE_CLOUD_API_KEY",
    default_lang = "en",
    category     = "video",
    since        = cr.since,
)
```

### GitHub repositories

Indexes issues, pull requests, commits, and discussions. Comments are concatenated with the parent body; external links in Markdown are followed one level; linked PDFs are mined when `mine_pdf=True`.

```python
cr.get_github_repositories(
    repositories = [("aurelienpierreeng", "ansel"), ("darktable-org", "rawspeed")],
    api_key  = "ghp_YOUR_PERSONAL_ACCESS_TOKEN",
    features = ["issues", "pulls", "commits", "discussions"],
    category = "Github",
    since    = cr.since,
    mine_pdf = True,
)
```

For `issues`/`pulls`/`commits` the `since` value is sent as the API's `?since=` filter, so the server returns only items updated after it — the fast incremental path. A read-only fine-grained token is enough.

### Stack Exchange forums

Indexes questions + answers + comments from any Stack Exchange community via the public API v2.3, with a sliding date-window pattern to work around the 25-page cap and a two-layer incremental filter (`fromdate=since` server-side, plus a per-post `last_edit_date` check).

```python
cr.get_stackexchange_posts(
    site     = "photo",             # site name as used by the SE API
    api_key  = "YOUR_SE_APP_KEY",   # optional but raises the quota to 10 000/day
    category = "forum",
    since    = cr.since,
)
```

---

## When the live site is unreachable — archives & scholarly APIs

Some sources go behind bot-protection, turn into JavaScript single-page apps with no crawlable HTML, or simply die. Rather than bypass protections, two adapters fetch the same content from **legitimate archives / open APIs**.

### Internet Archive (Wayback Machine)

[`get_wayback_pages`][core.crawler.Crawler.get_wayback_pages] enumerates a site's archived HTML captures via the public CDX API and fetches each *raw* snapshot (no Wayback toolbar), storing it under its original URL. It skips captures already indexed, so it is resumable, and takes an optional `clean_url(original) -> str | None` to canonicalise / de-duplicate noisy archived URLs (return `None` to skip one).

```python
import regex as re

def clean_url(original):
    m = re.search(r"munsell\.com(?::80)?(/color-blog/[^?#\s]*)", original, re.I)
    return "https://munsell.com" + m.group(1).rstrip("/") + "/" if m else None

cr.get_wayback_pages("munsell.com/color-blog*",
                     default_lang="en",
                     markup=["article", ("div", {"class": "entry-content"})],
                     category="reference",
                     clean_url=clean_url)
```

### OpenAlex (open scholarly metadata)

[`get_openalex_works`][core.crawler.Crawler.get_openalex_works] pulls titles and reconstructed abstracts for a journal or source from the free [OpenAlex](https://openalex.org) API — a legitimate route to open-access scholarly content whose publisher site blocks crawlers. Pass your email for the polite pool:

```python
cr.get_openalex_works(source_ids=["S2736465063"], category="reference", mailto="you@example.com")
```

---

## Crawling many sources concurrently

[`CrawlManager`][core.crawler.CrawlManager] runs several sources at once: each gets its **own** `Crawler` (own connection, own per-domain rate limiting) so the time one site spends waiting on a rate-limit or timeout is spent fetching another, while **writes are serialised** — only one thread writes the single-writer SQLite database at a time. `begin_dataset` / `commit_dataset` are handled internally, and deduplication runs **once for the whole run** at the end.

```python
from core import crawler

def crawl_ansel(cr):
    cr.get_website_from_sitemap("https://ansel.photos", "en", markup="article", category="reference")

def crawl_github(cr):
    cr.get_github_repositories([("aurelienpierreeng", "ansel")], api_key="ghp_…", category="Github")

mgr = crawler.CrawlManager("corpus.db", max_workers=6, dedup_once=True)
mgr.add("ansel",  crawl_ansel,  delay=1.0)
mgr.add("github", crawl_github, delay=1.0)
mgr.run()
```

### Organising sources as jobs (a pattern)

Once you have more than a handful of sources, it helps to put each in its own small module that declares *what* it fetches and *how often*, and drive them all from a thin runner. A convenient convention is a module exposing a dataset name, a cadence, and a `crawl(cr)` function:

```python
# sources/ansel.py
DATASET  = "ansel"
SCHEDULE = "daily"                       # your own cadence label
CRAWLER_KWARGS = {"delay": 1.0}          # forwarded to the Crawler constructor

def crawl(cr):
    cr.no_follow += ["/tag/", "?replytocom="]
    cr.get_website_from_sitemap("https://ansel.photos", "en",
                                markup="article", category="reference")
```

```python
# run.py — discover the modules for a cadence and hand them to CrawlManager
import importlib, pkgutil, sys
from core import crawler
import sources

cadence = sys.argv[1]                    # e.g. "daily"
mgr = crawler.CrawlManager("corpus.db", dedup_once=True)
for m in pkgutil.iter_modules(sources.__path__, "sources."):
    mod = importlib.import_module(m.name)
    if getattr(mod, "SCHEDULE", None) == cadence:
        mgr.add(mod.DATASET, mod.crawl, **getattr(mod, "CRAWLER_KWARGS", {}))
mgr.run()
```

You can then crawl fast-moving sources daily and slow references monthly with a cron per cadence. The Chantal reference implementation uses exactly this shape (plus a cross-process lock so two runs never write the database at once); adapt it to your needs.

---

## Crawl-time NLP prep

To make good use of the time the crawler spends blocked on the network, each HTML page is **normalised and tokenised at crawl time**, using the same [`Tokenizer`][core.nlp.Tokenizer] the batch stages use. When a page is written the crawler already computes and stores `parsed` (normalised text), `content_hash` (`SHA-1(parsed)`, used to skip identical content seen under another URL or in a previous crawl), and `tokenized` (the non-destructive token lists). Pass your own configured tokenizer if you have one:

```python
from core import nlp
cr = crawler.Crawler(delay=1.0, tokenizer=nlp.Tokenizer(...))
```

Because these values are identical to what the batch stages would compute, the later `batch_parse` / `batch_tokenize` passes are near-no-ops on freshly crawled rows. Stemming and vectorising — which need the *trained* language model — stay in the later batch stages (see [Build your own search engine](7-build-your-own-search-engine.md)).

---

## Crawling details

### robots.txt

The crawler respects [robots.txt](https://en.wikipedia.org/wiki/Robots.txt) automatically: for every new domain it fetches `/robots.txt`, honours `Disallow` for its user-agent, and reads `Crawl-delay` / `Request-rate` to set the per-domain throttle. A site whose `robots.txt` is `Disallow: /` (e.g. reddit) is therefore not crawled at all — that is intentional and correct; use the site's official API instead. Pages listed in `sitemap.xml` are treated as pre-authorised and skip the per-URL check.

### Cleaning up non-language content

Navigation menus, sidebars, and metadata are stripped **before** text extraction: the parser removes `<style>`, `<script>`, `<svg>`, `<img>`, `<picture>`, `<audio>`, `<video>`, `<iframe>`, `<embed>`, `<aside>`, `<nav>`, `<input>`, `<header>`, `<button>`, `<footer>`, `<summary>`, `<dialog>`, `<textarea>`, `<select>`, `<option>`, and form controls, and strips inline `style`/`data` attributes. When that is not enough, whitelist the real content container with `markup`. `<blockquote>`, `<code>` and `<pre>` are kept — quoted forum replies are handled by deduplication rather than at parse time.

### The `no_follow` list

Both crawling methods share a default blocklist of URLs that are never fetched (share links, login/signup, cart, profile paths). Extend it at construction or inside a crawl:

```python
cr = crawler.Crawler(delay=1.0, no_follow=["google.com", "/tag/", "?replytocom=", ".pdf"])
cr.no_follow += ["/view-album/", "persons-profile-"]
```

`no_follow` entries are substring-matched against the full URL; a match discards it with **no network request at all** — more aggressive than `contains_str`, which still visits non-matching URLs to find links.

---

## Deduplication and the corpus database

There is **one** database — the corpus — which has **no URL primary key**: the same URL may legitimately be captured several ways (via a content tag, as an external whole-body capture, under several parameter URLs), and duplication is resolved by content, not enforced by the schema.

Deduplication happens **as part of the crawl**, not as a separate script:

- at write time, `content_hash` lets the crawler skip a page whose exact content was already seen;
- on finalisation, one **incremental** pass ([`Deduplicator.run_incremental`][core.deduplicator.Deduplicator.run_incremental]) runs over just the URLs touched — targeted `DELETE`s that elect the best copy per URL (newest `crawled`, then longest content), never a full-table rewrite.

`commit_dataset` does this for a single-source crawl; `CrawlManager` defers it to one run-level pass over the union of touched URLs.

---

## Deferring the crawl to another machine

Because the corpus is the single source of truth, the crawl can run on cheap always-on hardware while heavier processing runs elsewhere. The two machines reconcile their copies with a timestamp delta using [`database.export_delta`][core.database.export_delta] / [`database.apply_delta`][core.database.apply_delta] / [`database.latest_crawled`][core.database.latest_crawled]:

```python
from core import database

# on the target: its watermark
watermark = database.latest_crawled("corpus.db")

# on the source: export everything newer into a small file, transfer it (FTP/rsync/…)
database.export_delta("corpus.db", "delta.db", since=watermark)

# on the target: apply it
database.apply_delta("delta.db", "corpus.db")
```

Only rows crawled after the target's watermark are transferred, so a routine sync moves kilobytes, not gigabytes.
