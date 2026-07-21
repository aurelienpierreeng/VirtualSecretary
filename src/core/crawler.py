"""Module containing utilities to crawl websites for HTML, XML and PDF pages for their text content. PDF can be read from their text content if any, or through optical characters recognition for scans. Websites can be crawled from a `sitemap.xml` file or by following internal links recursively from and index page. Each page is aggregated on a list of [core.types.web_page][] objects, meant to be used as input to train natural language AI models and to index and rank for search engines.

© 2023-2024 - Aurélien Pierre
"""

import os
import time
import datetime
import random
import json
import copy
import hashlib
import threading
from contextlib import nullcontext

from urllib.parse import urljoin
from email.utils import formatdate
from charset_normalizer import from_bytes
from dateutil.relativedelta import relativedelta

import requests
import regex as re
import numpy as np
import markdown
import sqlite3


from . import patterns, utils
from .pdf import get_pdf_content
from .types import web_page, sanitize_web_page, compute_content_hash
from .network import try_url, get_url, DelayedClass, HEADER
from .parser import ParsedHTML


def get_content_type(url: str, delay: DelayedClass, bypass_robots_txt=False) -> tuple[str, bool, str, dict | None, int]:
    """Probe an URL for HTTP headers only to see what type of content it returns.
    Try to sanitize partly-invalid URLs, like when protocols are not handled/redirected
    (`http` vs `https`), or invalid trailing characters, URL parameters and anchors are passed.

    Args:
        url: fully-formed link to try (including protocol)
        delay: time to wait before re-trying different sanitized URLs

    Returns:
        type (str): the type of content, like `plain/html`, `application/pdf`, etc.
        status (bool): the state flag:

            - `True` if the URL exists, but it might unreachable or forbidden,
            - `False` if the URL raises an HTTP 404 error (not found).

        response_url (str): the actual target URL, sanitized and possibly redirected.
        states (int): HTTP return code
    """
    try:
        response, header, new_url = try_url(url, delay, timeout=20, bypass_robots_txt=bypass_robots_txt)
        if response and response.headers and 'content-type' in response.headers:
            return (response.headers['content-type'], 
                    (response.status_code > 0 and response.status_code < 400), 
                    new_url, 
                    header, 
                    response.status_code)
        else:
            return ("", False, new_url, header, -1)
    except Exception as e:
        print("Header error:", url, e)
        return ("", False, url, None, -1)


def relative_to_absolute(URL: str, domain: str, current_page: str) -> str:
    """Convert a relative URL to absolute by prepending the domain.

    Arguments:
        URL: the URL string to normalize to absolute,
        domain: the domain name of the website, without protocol (`http://`) nor trailing slash.
            It will be appended to the relative links starting by `/`.
        current_page: the URL of the page from which we analyze links.
            It will be appended to the relative links starting by `./`.

    Returns:
        The normalized, absolute URL on this website.

    Examples:

        >>> relative_to_absolute("folder/page", "me.com")
        "://me.com/folder/page"
    """
    # relative URL from current page
    if current_page is None:
        raise TypeError("`current_page` should be defined")

    # relative path declared from current page. Hard
    test_url = ""
    
    try:
        test_url = urljoin(current_page, URL)
    except:
        if "://" in URL:
            # already a fully-formed URL
            test_url = URL
        elif URL.startswith("#"):
            # internal anchor. Nothing to do with that.
            test_url = "://" + domain.strip("/") + "/"
        elif URL.startswith("/"):
            # relative path declared from root. Easy
            test_url = "://" + domain.strip("/") + "/" + URL.lstrip("/")
        elif current_page.endswith(URL):
            # a wrong link is trying to recursively call itself.
            test_url = "://" + domain.strip("/") + "/"

    return test_url


def radical_url(URL: str) -> str:
    """Trim an URL to the page (radical) part, removing anchors if any (internal links)

    Examples:
        >>> radical_url("http://me.com/page#section-1")
        "http://me.com/page"
    """
    anchor = re.match(r"(.+?)\#(.+)", URL)
    return anchor.groups()[0] if anchor else URL


@utils.exit_after(120)
def get_content(url, custom_header, delay: DelayedClass) -> tuple[str, str, int]:
    content, url, status, encoding, apparent_encoding = get_url(url, delay, timeout=60, custom_header=custom_header)
    if content is None:
        raise(Exception("No page found"))

    if isinstance(content, bytes):
        result = from_bytes(content).best()
        # Of course some institutionnal websites don't use UTF-8, so let's guess
        try:
            # Try UTF-8. Note that the Selenium backend doesn't return encoding.
            content = str(result) if result else content.decode()
        except (UnicodeDecodeError, AttributeError):
            try:
                content = content.decode(apparent_encoding)
            except (UnicodeDecodeError, AttributeError):
                content = content.decode(encoding)
    elif not isinstance(content, str):
        raise(Exception("Content is neither bytes nor string"))

    return content, url, status



def parse_content(content: str) -> ParsedHTML:
    # Minified HTML doesn't have line breaks after block-level tags.
    # This is going to make sentence tokenization a nightmare because ParsedHTML doesn't add them in get_text()
    # Re-introduce here 2 carriage-returns after those tags to create paragraphs.
    unminified = re.sub(r"(\<\/(?:div|section|main|section|aside|header|footer|nav|time|article|h[1-6]|p|ol|ul|li|details|pre|dl|dt|dd|table|tr|th|td|blockquote|style|img|audio|video|iframe|embed|figure|canvas|fieldset|hr|caption|figcaption|address|form|noscript|select)\>)",
                        r"\1\n\n\n\n", content, timeout=60)
    # Same with inline-level tags, but only insert space, except for superscript and subscript
    unminified = re.sub(r"(\<\/(?:a|span|time|sup|abbr|b|i|em|strong|code|dfn|big|kbd|label|textarea|input|option|var|q|tt)\>)",
                        r"\1 ", unminified, timeout=60)

    return ParsedHTML.from_html(unminified)


def get_page_content(url: str | None, 
                     delay: DelayedClass,
                     content: str | None = None, 
                     custom_header={}) -> tuple[ParsedHTML | None, str | None, int]:
    """Request an (x)HTML page through the network with HTTP GET and feed its response to a ParsedHTML handler. This needs a functionnal network connection.

    The DOM is pre-filtered as follow to keep only natural language and avoid duplicate strings:

    - media tags are removed (`<iframe>`, `<embed>`, `<img>`, `<svg>`, `<audio>`, `<video>`, etc.),
    - code and machine language tags are removed (`<script>`, `<style>`, `<math>`),
    - menus and sidebars are removed (`<nav>`, `<aside>`),
    - forms, fields and buttons are removed(`<select>`, `<input>`, `<button>`, `<textarea>`, etc.)

    The HTML is un-minified to help end-of-sentences detections in cases where sentences don't end with punctuation (e.g. in titles).

    Arguments:
        url: a valid URL that can be fetched with an HTTP GET request.
        content: a string buffer used as HTML source. If this argument is passed, we don't fetch `url` from network and directly use this input.

    Returns:
        a tuple with:
            1. [core.parser.ParsedHTML][] object initialized with the page DOM for further text mining. `None` if the HTML response was empty or the URL could not be reached. The list of URLs found in page before removing meaningless markup is stored as a list of strings in the `object.links` member. `object.h1` and `object.h2` contain a set of headers 1 and 2 found in the page before removing any markup. `object.date` contains the best-guess for the date.
            2. the final URL of the retrieved page, which might be different from the input URL if HTTP redirections happened,
    """

    try:
        status = 200

        if content is None and url is not None:
            content, url, status = get_content(url, custom_header, delay)

        if not content:
            return None, url, -1
        else:
            return parse_content(content), url, status
    
    except Exception as e:
        print("Page content error", e)
        return None, url, -1


def parse_page(page: ParsedHTML,
               url: str,
               lang: str | None, markup: str|tuple|list[str]|list[tuple]|None,
               date: str | None = None,
               category: str | None = None,
               tokenizer=None) -> list[web_page]:
    """Get the requested markup from the requested page URL.

    This chains in a single call:

    - [core.parser.ParsedHTML.get_page_markup][]
    - [core.parser.ParsedHTML.get_date][]
    - [core.parser.ParsedHTML.get_excerpt][]

    Arguments:
        page: a [core.parser.ParsedHTML][] handler with pre-filtered DOM,
        url: the valid URL accessible by HTTP GET request of the page
        lang: the provided or guessed language of the page,
        markup: the markup to search for. See [core.parser.ParsedHTML.get_page_markup][] for details.
        date: if the page was retrieved from a sitemap, usually the date is available in ISO format (`yyyy-mm-ddTHH:MM:SS`) and can be passed directly here. Otherwise, several attempts will be made to extract it from the page content (see [core.parser.ParsedHTML.get_date][]).
        category: arbitrary category or label defined by user

    Returns:
        The content of the page, including metadata, in a [core.types.web_page][] singleton.
    """
    if ".wikipedia.org" in url and markup == "body":
        # Wikipedia is massive enough to appear in at least one external link.
        # For external links, the default behaviour is to get the whole <body>.
        # For Wikipedia, make an special case here by restricting markup to meaningfull stuff.
        # TODO: is this portable to all pages generated by MediaWiki ?
        markup = ("div", {"id": "mw-content-text"})

    # Parse everything in one go from the content markup
    page.parse(markup)

    if page.content and page.title:
        # When a Tokenizer is available, do the reusable NLP prep at crawl time:
        # normalize the text once (via the SAME Tokenizer the batch stages use, so a custom
        # subclass is honored), STORE the normalized `parsed` so it is reused downstream
        # without recomputation, and fingerprint it. The `content_hash` is identical to what
        # batch_parse_web_page would compute, so identical content under a different URL — or
        # from a previous crawl — can be detected and skipped early.
        parsed = None
        content_hash = None
        tokenized = None
        page_lang = page.lang or lang
        if tokenizer is not None:
            parsed = tokenizer.compute_parsed(page.title, page.content)
            content_hash = compute_content_hash(parsed)
            # Crawl-time tokenization: fill the `tokenized` column now, while the crawl is mostly
            # blocked on network I/O and rate-limit sleeps, so this CPU work overlaps other sources'
            # idle time (in CrawlManager) instead of waiting for the batch stage. Uses the SAME
            # Tokenizer instance as the batch pass, so `batch_tokenize(only_none=True)` later is a
            # no-op on these rows. `page_lang` is upgraded to the resolved ISO-639-1 code, matching
            # what batch_tokenize would have written.
            page_lang, tokenized = tokenizer.tokenize_parsed(parsed, page_lang)

        result = sanitize_web_page(web_page(
            title=page.title,
            url=url,
            date=page.date or date,
            content=page.content,
            excerpt=page.excerpt,
            h1=page.h1,
            h2=page.h2,
            lang=page_lang,
            category=category,
            crawled=datetime.datetime.now(datetime.timezone.utc),
            parsed=parsed,
            tokenized=tokenized,
            content_hash=content_hash,
        ))
        print(f"Parsed: {result['url']} [{result['lang']}] hash={result['content_hash'][:12]}")
        return [result]
    else:
        return []


def check_contains(contains_str: list[str] | str, url: str):
    if contains_str == "":
        return True
    elif isinstance(contains_str, str):
        return contains_str in url
    elif isinstance(contains_str, list):
        for elem in contains_str:
            if elem in url:
                return True
        return False

    raise TypeError("contains_str has a wrong type")


def hash_with_category(data: str, category: str) -> str:
    """Produce a unique identifier mixing data and category."""
    hasher = hashlib.sha256()

    hasher.update(category.encode("utf-8"))
    hasher.update(b"\0")  # separator
    hasher.update(data.encode("utf-8"))

    return hasher.hexdigest()


def _parse_iso_date(date_str: str) -> datetime.datetime:
    """Parse an ISO 8601 / RFC 3339 string to a timezone-aware datetime.

    Handles both the ``Z`` suffix (accepted by Python 3.11+) and the
    ``+00:00`` form accepted by all supported Python versions.
    """
    cleaned = date_str.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    dt = datetime.datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _normalize_tz(dt: datetime.datetime) -> datetime.datetime:
    """Return *dt* with timezone info, assuming UTC if the datetime is naïve."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.timezone.utc)
    return dt


class Crawler(DelayedClass):
    no_follow: list[str] = [
        # Social sharing links: no interest for us
        "api.whatsapp.com/share",
        "api.whatsapp.com/send",
        "pinterest.fr/pin/create",
        "pinterest.com/pin/create",
        "facebook.com/sharer",
        "twitter.com/intent/tweet",
        "twitter.com/share",
        "x.com/share",
        "reddit.com/submit",
        "t.me/share", # Telegram share
        "linkedin.com/share",
        "vk.com/share.php",
        "bufferapp.com/add",
        "getpocket.com/edit",
        "tumblr.com/share",
        "www.addtoany.com/add_to",
        "share.flipboard.com/bookmarklet/",
        "?share=",
        "?replytocom=",
        # Machine-translated pages
        "translate.google.com/translate", 
        # Social media platforms: they all block us globally by robots.txt
        # so no point wasting requests on them. Anyway, there is no relevant content
        # for us there.
        "flickr.com",
        "//flic.kr/",
        "instagram.com",
        "threads.com",
        "facebook.com",
        "linkedin.com",
        "twitter.com",
        "//t.co/"
        "tiktok.com",
        "pinterest.com",
        "//x.com/",
        "reddit.com",
        "sciprofiles.com",
        "www.citeulike.org",
        "linktr.ee",
        # Non-HTTP links: nothing we can do with them
        "mailto:",
        # Endpoints to private website areas: no info to crawl for us there
        "/profile/",
        "/login/",
        "/login.php",
        "/wp-login.php"
        "/signup/",
        "/signup.php",
        "/login?",
        "/signup?"
        "/user/",
        "/member/",
        "/register?",
        "login.microsoftonline.com",
        # Files extensions we don't handle: no point sending requests either
        ".css",
        ".js",
        ".json",
        ".jpg",
        ".png",
        ".jpeg",
        ".gif",
        ".webp",
        ".heif",
        ".tif",
        ]
    """List of URLs sub-strings that will disable crawling if they are found in URLs. Mostly social networks sharing links."""

    def __init__(self, delay: float = 1., no_follow: list[str] = [],
                 known_urls: dict[str, datetime.datetime] | None = None,
                 since: datetime.datetime | None = None,
                 tokenizer=None):
        """Crawl a website from its sitemap or by following internal links recusively from an index page.
        This class needs therefore to be used within a `with` statement that will take care of resources
        allocations and releases in background.

        Parameters:
            delay: 
                time in seconds to wait before 2 HTTP requests.
                The right delay will prevent the crawler from being throttled by anti-DoS rules while making it as fast as possible.
                Set to `0.0` if you are crawling your own servers and they have no DoS protection.
            no_follow: list of URL parts to completely ignore, that is not index them but not even crawl them for internal links.
            known_urls:
                mapping of ``url → last crawled datetime`` for pages already in the index.
                When provided, crawling methods use it to skip pages that have not changed since
                they were last indexed.  Populate it conveniently with [load_known_urls][core.crawler.Crawler.load_known_urls].
            since:
                global freshness cut-off for recursive crawling.  Any URL present in *known_urls*
                and last crawled **on or after** this datetime will be skipped entirely.
                Has no effect when *known_urls* is empty or when a URL is not yet known.
                For sitemap crawling, the sitemap's own ``<lastmod>`` field takes precedence;
                *since* is only used as a fallback for entries that have no ``<lastmod>``.

        Example:
            ```python
            db = database.open_db("my-engine.db")

            with crawler.Crawler(delay=1.0) as cr:
                cr.load_known_urls(db)      # populate incremental-update map
                cr.since = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)

                # Only re-fetches pages whose <lastmod> is newer than the stored crawl date.
                pages = cr.get_website_from_sitemap("https://domain.com", "en")

                # Only re-fetches pages not yet in the index, or crawled before since.
                pages += cr.get_website_from_crawling("https://forum.domain.com", "en")
            ```

        """
        self.crawled_URL: list[str] = []
        """List of { URL + category } hashes already visited.
        Websites crawled from sitemap and also following internal links recursively will tag
        recursively-crawled pages with an `external` category, which will later be considered
        by [core.deduplicator.Deduplicator][] with a lower priority than any other category.
        Sitemap crawling may restrict content to selected HTML tags and produce better-quality data,
        with less noise. So we need to keep crawling everything from sitemap, whether or not it was
        already crawled from internal links earlier, and dedup will sort it out."""
        
        self.crawled_content: list[str] = []
        """List of hashes of content already known"""

        self.known_content_hashes: set[str] = set()
        """Canonical `content_hash` values (SHA-1 of normalized content) already indexed.
        Seeded from an existing index via [load_known_content][core.crawler.Crawler.load_known_content]
        and extended with hashes seen during this crawl, so the same content reached under a
        different URL — within this crawl or from a previous one — is not indexed twice."""

        self.known_urls: dict[str, datetime.datetime] = dict(known_urls) if known_urls else {}
        """Mapping of URL → last-crawled datetime for incremental updates.
        Populated at construction time or via [load_known_urls][core.crawler.Crawler.load_known_urls].

        Note:
            We strip leading and trailing `/` for generality, in URL keys.
        """

        self.db: sqlite3.Connection | None = None
        """Live connection to the canonical DB for the DB-native crawl flow. When set (by
        pre_process_crawling_canonical), the crawler answers "is this URL known / this content
        already indexed?" with on-demand indexed lookups against the DB, instead of the
        preloaded `known_urls` / `known_content_hashes` sets — no large static structures in RAM.
        Falls back to the preloaded sets when None (the legacy tarball flow)."""

        self._session_content: set[str] = set()
        """Canonical content_hash values produced in THIS crawl session (not yet persisted to the
        DB when reads and writes are separate). Combined with the DB lookup so content reached
        twice within one crawl is still deduplicated."""

        self.write_lock: threading.Lock | None = None
        """Optional shared lock serializing DB writes across concurrently-running crawlers (set by
        CrawlManager). When several sources crawl at once to hide per-site latency, fetching runs in
        parallel but every write (`_flush_writes`, `commit_dataset`) holds this lock, so only ONE
        thread writes the shared canonical at a time. None = single-crawler mode (no locking)."""

        self.dataset_name: str | None = None
        """Origin dataset tag applied to pages written in the DB-native flow (set by
        pre_process_crawling_canonical)."""

        self._write_buffer: list = []
        """Small buffer of pages awaiting a batched INSERT into `self.db` (DB-native flow).
        Bounds memory and fsync churn instead of holding every crawled page in one big list."""

        self._written_urls: set[str] = set()
        """URLs written directly to `self.db` this crawl — the delta post_process deduplicates."""

        self.since: datetime.datetime | None = since
        """Global freshness cut-off for recursive and API-based crawling.
        Pages in *known_urls* last crawled on or after this datetime are skipped."""

        self._tokenizer = tokenizer
        """Tokenizer used to normalize page text at crawl time. Lazily defaults to a bare
        `nlp.Tokenizer()` (see the `tokenizer` property) — the same configuration
        post_process_crawling / chantal-03 use for batch parsing, so the `parsed` text and
        `content_hash` computed here match downstream and can be reused. Inject a custom
        (e.g. subclassed) Tokenizer to override normalization for a non-FR/EN corpus."""

        self.no_follow += no_follow
        self.delay = delay
        self.last_request = datetime.datetime.now().timestamp()

        self.errors = []
        """URLs that couldn't be accessed due to blocking or throttling"""

        self.notfound = []
        """URLs returning error 404 - not found"""


    @property
    def tokenizer(self):
        """The Tokenizer used for crawl-time normalization. Defaults to a bare
        `nlp.Tokenizer()`, imported lazily so `crawler` never imports `nlp` at module
        load time (that would be a circular import: `nlp` imports `crawler`)."""
        if self._tokenizer is None:
            from .nlp import Tokenizer  # deferred: breaks the crawler↔nlp import cycle
            self._tokenizer = Tokenizer()
        return self._tokenizer


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Safety net: flush any buffered writes and close the connection if post_process didn't
        # (e.g. crawl error, or a scraper that forgot the post step). Normal flow closes it in
        # post_process, which sets self.db = None, so this branch is a no-op then.
        if self.db is not None:
            try:
                self._flush_writes()
            except Exception:
                pass
            try:
                self.db.close()
            except Exception:
                pass
            self.db = None

        print("PROCESSED URLS:", len(set(self.crawled_URL)))
        print("404 ERRORS:", len(set(self.notfound)))
        for error in set(self.notfound):
            print(error)

        print("OTHER ERRORS:", len(set(self.errors)))
        for error in set(self.errors):
            print(error)


    def load_known_urls(self, db: sqlite3.Connection) -> int:
        """Populate the incremental-update map from an existing index database.

        After calling this, all crawling methods will skip pages whose stored crawl
        timestamp indicates they are still fresh (see ``self.since`` and the
        ``<lastmod>`` logic in [get_website_from_sitemap][core.crawler.Crawler.get_website_from_sitemap]).

        Arguments:
            db: an open SQLite connection to a Virtual Secretary database
                (as returned by [core.database.open_db][] or [core.database.create_db][]).

        Returns:
            Number of URL entries loaded.

        Example:
            ```python
            db  = database.open_db("my-engine.db")
            with crawler.Crawler(delay=1.0) as cr:
                cr.load_known_urls(db)
                cr.since = datetime.datetime(2025, 6, 1, tzinfo=datetime.timezone.utc)
                pages = cr.get_website_from_sitemap("https://domain.com", "en")
            db.close()
            ```
        """
        cursor = db.execute(
            "SELECT url, crawled, wayback FROM pages WHERE crawled IS NOT NULL AND url IS NOT NULL"
        )
        count = 0
        for url, crawled, wayback in cursor.fetchall():
            if not url or not crawled:
                continue
            if isinstance(crawled, str):
                try:
                    crawled = datetime.datetime.fromisoformat(crawled)
                except (ValueError, AttributeError):
                    continue
            if isinstance(crawled, datetime.datetime):
                self.known_urls[url.strip("/")] = crawled
                count += 1

            if wayback:
                self.known_urls[wayback.strip("/")] = crawled
                count += 1


        print(f"Loaded {count} known URLs for incremental crawling")
        return count


    def load_known_content(self, db: sqlite3.Connection) -> int:
        """Seed [known_content_hashes][core.crawler.Crawler.known_content_hashes] from an
        existing index so the crawler can skip re-indexing content that is already stored
        under any URL (see the content-dedup step in
        [get_website_from_crawling][core.crawler.Crawler.get_website_from_crawling]).

        Safe on legacy databases that predate the ``content_hash`` column (returns 0).

        Arguments:
            db: an open SQLite connection to a Virtual Secretary database.

        Returns:
            Number of content hashes loaded.
        """
        try:
            cursor = db.execute(
                "SELECT content_hash FROM pages WHERE content_hash IS NOT NULL"
            )
        except sqlite3.OperationalError:
            # No content_hash column yet (old bundle) — nothing to seed.
            return 0

        count = 0
        for (content_hash,) in cursor:
            if content_hash:
                self.known_content_hashes.add(content_hash)
                count += 1

        print(f"Loaded {count} known content hashes for content dedup")
        return count


    # ── DB-native known-state lookups ──────────────────────────────────────────────────
    # These answer "have we seen this URL / this content before?" without preloading the whole
    # index into RAM: when `self.db` is set they query it on demand (url and content_hash are
    # indexed, so each lookup is O(log n)); otherwise they use the legacy preloaded sets.

    def _known_last_crawled(self, url: str) -> datetime.datetime | None:
        """Most recent `crawled` datetime recorded for *url* (None if never crawled)."""
        if self.db is not None:
            row = self.db.execute(
                "SELECT MAX(crawled) FROM pages WHERE url = ? OR wayback = ?",
                (url, url),
            ).fetchone()
            val = row[0] if row else None
            if not val:
                return None
            return _normalize_tz(datetime.datetime.fromisoformat(val) if isinstance(val, str) else val)
        # Legacy: preloaded map (keyed on stripped url).
        val = self.known_urls.get(url.strip("/"))
        return _normalize_tz(val) if val else None

    def _site_last_crawled(self, website: str) -> datetime.datetime | None:
        """Most recent time we crawled ANY page of *website*'s domain (None if never / no DB).
        Used as the `If-Modified-Since` reference for the site-level sitemap gate."""
        addr = patterns.split_url(website)
        if not addr or self.db is None:
            return self.since
        domain = addr[1]
        row = self.db.execute(
            "SELECT MAX(crawled) FROM pages WHERE url LIKE ? OR url LIKE ?",
            (f"https://{domain}/%", f"http://{domain}/%"),
        ).fetchone()
        val = row[0] if row else None
        if val:
            return _normalize_tz(datetime.datetime.fromisoformat(val) if isinstance(val, str) else val)
        return self.since

    def _sitemap_unmodified(self, sitemap_url: str, since: datetime.datetime | None) -> bool:
        """Site-level change gate for full-redeploy static sites (doxygen, most static-site
        generators): every file — sitemap.xml included — receives one uniform deploy mtime, so a
        single conditional request on the sitemap reveals whether ANYTHING was redeployed since we
        last crawled. There is no reliable per-page date on such sites (no `<lastmod>`, and the HTTP
        `Last-Modified`/`ETag` is the deploy timestamp, identical across all pages), so this whole-
        site check is the finest-grained cheap signal available; per-page `content_hash` dedup then
        filters which pages were *actually* changed once we do crawl.

        Returns True (skip the whole site) only on an explicit `304 Not Modified`. Any error,
        missing `Last-Modified`, or first-ever crawl (`since is None`) falls through to False so we
        crawl normally — the gate can only ever *save* requests, never hide a change."""
        if since is None:
            return False
        addr = patterns.split_url(sitemap_url)
        domain = addr[1] if addr else ""
        try:
            self.sleep(domain, self.delay)  # respect the crawl rate for the probe
            resp = requests.head(
                sitemap_url,
                headers={**HEADER, "If-Modified-Since": formatdate(since.timestamp(), usegmt=True)},
                timeout=20, allow_redirects=True,
            )
            if resp.status_code == 304:
                return True
        except Exception as e:
            print(f"[sitemap-gate] {sitemap_url}: conditional check failed ({e}); crawling normally")
        return False

    def _content_known(self, content_hash: str | None) -> bool:
        """Whether *content_hash* was already produced this session or is persisted in the DB."""
        if not content_hash:
            return False
        if content_hash in self._session_content:
            return True
        if self.db is not None:
            return self.db.execute(
                "SELECT 1 FROM pages WHERE content_hash = ? LIMIT 1", (content_hash,)
            ).fetchone() is not None
        return content_hash in self.known_content_hashes

    def _mark_content_seen(self, content_hash: str | None) -> None:
        """Record a content_hash produced this session so later pages dedup against it even before
        it is persisted."""
        if content_hash:
            self._session_content.add(content_hash)


    def _emit(self, pages: list) -> int:
        """Central sink for freshly-produced pages, and the ONLY way pages leave the crawler. The
        crawler is DB-native: it never returns a `list[web_page]`. Each produced page is deduped
        against known content (this session or persisted) and, if new, written straight into
        `self.db` (buffered) — so the caller keeps nothing in RAM.

        Returns the number of pages actually written (survivors after content dedup), purely so
        callers can log progress; the pages themselves are already in the DB."""
        if self.db is None:
            raise RuntimeError(
                "Crawler is DB-native: set cr.db before crawling (scraping.pre_process_crawling / "
                "pre_process_crawling_canonical). _emit has no in-RAM fallback.")
        written = 0
        for page in pages:
            content_hash = page.get("content_hash")
            if self._content_known(content_hash):
                print(f"Skip (duplicate content): {page.get('url')}")
                continue
            self._mark_content_seen(content_hash)
            if self.dataset_name:
                page["dataset"] = f",{self.dataset_name},"
            self._write_buffer.append(page)
            if page.get("url"):
                self._written_urls.add(page["url"])
            if len(self._write_buffer) >= 100:
                self._flush_writes()
            written += 1
        return written


    def _flush_writes(self) -> None:
        """Flush the buffered pages into `self.db` with a batched INSERT (no URL primary key on the
        canonical, so old versions of a re-crawled URL coexist and post_process's URL-election keeps
        the newest). Deferred import: `database` imports `crawler`, so importing it here avoids the
        module-load cycle."""
        if self._write_buffer and self.db is not None:
            from . import database
            with (self.write_lock or nullcontext()):  # serialize writes across concurrent crawlers
                database.populate_db(self.db, self._write_buffer)
            self._write_buffer.clear()


    # ── DB-native crawl lifecycle ──────────────────────────────────────────────────────
    # A DB-native crawl is bracketed by begin_dataset() / commit_dataset(): the first attaches
    # the corpus DB the crawler reads known state from and writes pages into, the second finalizes
    # and closes it. Generic (no project-specific DB name); a caller supplies the DB.

    def begin_dataset(self, db, dataset_name: str) -> None:
        """Attach a corpus DB for a DB-native crawl of *dataset_name* and compute the incremental
        crawl-since threshold from it.

        *db* is either an open ``sqlite3.Connection`` or a DB name (created no-URL-PK if absent, so
        the same URL may be captured several ways and dedup elects the best). The crawler then
        answers known-URL / known-content lookups from this DB and writes each produced page
        straight into it; :meth:`commit_dataset` finalizes and closes it.
        """
        from . import database
        conn = database.create_db(db, url_primary_key=False) if isinstance(db, str) else db
        database.ensure_incremental_autovacuum(conn)
        self.db = conn
        self.dataset_name = dataset_name
        self.since = self.get_crawling_threshold(conn, dataset=dataset_name)
        print(f"Crawling {dataset_name!r} since {self.since} (DB-native lookups)")

    def commit_dataset(self, dataset_name: str) -> None:
        """Finalize the DB-native crawl opened by :meth:`begin_dataset`: flush buffered writes, then
        parse/merge-provenance/dedup THIS crawl's delta in place, compress, and close the DB. Use
        this for a single-source run; under CrawlManager's dedup-once mode use :meth:`flush_dataset`
        instead so the whole run is deduplicated once at the end.
        """
        from . import database
        db = self.db
        if db is None:
            raise RuntimeError("commit_dataset requires a prior begin_dataset (cr.db unset)")
        database.ensure_incremental_autovacuum(db)
        self._flush_writes()

        before = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        urls = self._written_urls
        print(f"[{dataset_name}] {len(urls)} URLs touched (direct-written); DB before {before} rows")

        # Serialize the write-heavy finalization (concurrent crawlers must not parse/merge/dedup the
        # shared canonical at once). _flush_writes took/released the lock itself, so this isn't
        # re-entrant. finalize_canonical no-ops the dedup when there's nothing new.
        with (self.write_lock or nullcontext()):
            finalize_canonical(db, self.tokenizer, urls, label=dataset_name)
        after = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        print(f"[{dataset_name}] DB after: {after} rows total ({after - before:+d})")
        database.close_db(db)
        self.db = None

    def flush_dataset(self, dataset_name: str) -> set:
        """Flush buffered writes and close the DB WITHOUT deduplicating — for CrawlManager's
        dedup-once mode, where a single finalize pass dedups the whole run's delta at the end.
        Returns the set of URLs this crawl touched (so the manager can dedup their union)."""
        if self.db is None:
            raise RuntimeError("flush_dataset requires a prior begin_dataset (cr.db unset)")
        self._flush_writes()  # takes/releases the write lock itself — don't re-wrap (non-reentrant)
        urls = set(self._written_urls)
        print(f"[{dataset_name}] {len(urls)} URLs touched (flushed; dedup deferred to run end)")
        from . import database
        database.close_db(self.db)
        self.db = None
        return urls


    def get_most_recent_page(self, db:sqlite3.Connection, dataset: str | None = None) -> datetime.datetime | None:
        """Get the datetime of the most recent `web_page` indexed in the `db` database.

        When *dataset* is given, only rows belonging to that source are considered — essential
        when many sources share one canonical DB, so each keeps its own incremental threshold.
        """
        cursor = db.execute(*self._dataset_scoped_sql("MAX(datetime)", db, dataset))
        date = cursor.fetchone()[0]

        if date is not None:
            return _normalize_tz(datetime.datetime.fromisoformat(date))
        else:
            return None


    @staticmethod
    def _dataset_scoped_sql(agg: str, db: sqlite3.Connection, dataset: str | None) -> tuple:
        """Build (sql, params) for `SELECT agg FROM pages`, optionally scoped to a dataset via the
        ',a,b,' membership substring match (`dataset` is a set, not a single value). This is a
        single aggregate query per crawl and the `dataset` column is small, so the substring scan
        is cheap (~0.1s at 700k rows)."""
        if dataset is None:
            return (f"SELECT {agg} FROM pages",)
        return (f"SELECT {agg} FROM pages WHERE dataset LIKE ?", (f"%,{dataset},%",))


    def get_most_recent_crawl(self, db:sqlite3.Connection, dataset: str | None = None) -> datetime.datetime | None:
        """Get the datetime of the most recently crawled `web_page` indexed in the `db` database.

        When *dataset* is given, restricts to that source (per-source threshold in a shared DB).
        """
        cursor = db.execute(*self._dataset_scoped_sql("MAX(crawled)", db, dataset))
        date = cursor.fetchone()[0]

        if date is not None:
            return _normalize_tz(datetime.datetime.fromisoformat(date))
        else:
            return None


    def get_crawling_threshold(self, db:sqlite3.Connection, dataset: str | None = None) -> datetime.datetime | None:
        """Get the safe date from which we should restart incremental crawling of a website.
        We use the oldest among the last crawling date and the most recent page, to account
        for possibly badly-formed page dates set in the future at the time of crawling.

        *dataset* scopes the threshold to a single source when several share one canonical DB.
        """

        recent_page = self.get_most_recent_page(db, dataset)
        recent_crawl = self.get_most_recent_crawl(db, dataset)

        return min((d for d in (recent_page, recent_crawl) if d is not None), default=None)


    def discard_link(self, url):
        """Returns True if the url is found in the `self.no_follow` list"""
        for elem in self.no_follow:
            if elem in url:
                return True

        return False


    def get_immediate_links(self, links: list[str], domain, default_lang, langs, category, contains_str, internal_links: str = "any", mine_pdf = False) -> int:
        """Follow internal and external links contained in a webpage only to one recursivity level,
        including PDF files and HTML pages. This is useful to index references docs linked from a page.

        Args:
            internal_links: defines what to do with links found inside the HTML page body/content:
                - `any`: follow and include all links found in page content, no matter what domain they point to,
                - `internal`: follow and include links found in page content only if they point to the same domain as the current page,
                - `external`: follow and include links found in page content only if they point to a different domain than the current page,
                - `ignore`: don't follow links found in page content.

        Returns:
            number of pages written to the DB from the followed links.
        """
        written = 0
        if internal_links == "ignore":
            return written

        for nextURL in links:
            if hash_with_category(nextURL, category) in self.crawled_URL:
                continue

            current_address = patterns.split_url(nextURL)
            if not current_address:
                continue

            current_protocol, current_domain, current_page, current_params, current_anchor = current_address

            include = False
            if internal_links == "external":
                # Follow internal links only if they point outside the current domain
                include = domain not in current_domain
            elif internal_links == "any":
                # Follow all internal links
                include = True
            elif internal_links == "internal":
                # Follow internal links only if they point inside the current domain
                include = domain in current_domain
            else:
                raise ValueError("Internal link following mode %s is unknown" % internal_links)

            if include:
                if domain not in current_domain:
                    # If the current URL doesn't belong to the same domain as the parent,
                    # we don't pass on the category of the parent page
                    # because we have no idea what the external URL is.
                    category = "external"

                written += self.get_website_from_crawling(current_protocol + "://" + current_domain + current_page + current_params, default_lang, "", langs, max_recurse_level=1, category=category, contains_str=contains_str, mine_pdf=mine_pdf, _recursion_level=0, _mainthread=False)

        return written


    def update_link(self, old_link: str, new_link: str | None, category: str, status_code: int) -> str:
        """Update target link with possible HTTP redirections

        Arguments:
            old_link: 
                original URL followed, found in HTML

            new_link: 
                destination URL retrieved, possibly after HTTP redirections.

            category:
                tagged category of the page

            status_code:
                HTML returned status code
        """

        self.crawled_URL.append(hash_with_category(old_link, category))

        if new_link and new_link != old_link:
            self.crawled_URL.append(hash_with_category(new_link, category))

        if status_code == 404:
            self.notfound += list({new_link, old_link})
        elif status_code == -1 or status_code >= 400:
            self.errors += list({new_link, old_link, status_code})

        return new_link or old_link



    def get_website_from_crawling(self,
                                  website: str,
                                  default_lang: str = "en",
                                  child: str = "/",
                                  langs: tuple = ("en", "fr"),
                                  markup: str | list[str] | tuple | list[tuple] = "body",
                                  contains_str: str | list[str] = "",
                                  max_recurse_level: int = -1,
                                  category: str = "",
                                  restrict_section: bool = False,
                                  mine_pdf: bool = False,
                                  _recursion_level: int = 0,
                                  _mainthread: bool = True) -> int:
        """Recursively crawl all pages of a website from internal links found starting from the `child` page. This applies to all HTML pages hosted on the domain of `website` and to PDF documents either from the current domain or from external domains but referenced on HTML pages of the current domain.

        Arguments:
            website: 
                root of the website, including `https://` or `http://` without trailing slash.

            default_lang: 
                provided or guessed main language of the website content. Not used internally.

            child: 
                page of the website to use as index to start crawling for internal links.

            langs: 
                ISO-something 2-letters code of the languages for which we attempt to fetch the translation 
                if available, looking for the HTML `<link rel="alternate" hreflang="...">` tag.
            
            contains_str: 
                a string or a list of strings that should be contained in a page URL for the page to be indexed.
                On a forum, you could for example restrict pages to URLs containing `"discussion"` to get only
                the threads and avoid user profiles or archive pages.
            
            markup: 
                see [core.parser.ParsedHTML.get_page_markup][]

            max_recurse_level: 
                this method will call itself recursively on each internal link found in the current page, 
                starting from the `website/child` page. The `max_recursion_level` defines how many times 
                it calls itself until it is stopped, if it is stopped. When set to `-1`, it stops when all 
                the internal links have been crawled.
            
            category: 
                arbitrary category or label set by user for classification. Will be automatically set to `external`
                for URLs followed outside of the main domain.

            restrict_section: 
                set to `True` to limit crawling to the website section defined by `://website/child/*`. 
                This is useful when indexing parts of very large websites when you are only interested in a small subset.
            
            mine_pdf: 
                set to `True` to aggressively try to crawl PDF linked on external HTML pages. 
                This may increase RAM consumption dramatically.
            
            _recursion_level: 
                __DON'T USE IT__. Everytime this method calls itself recursively, 
                it increments this variable internally, and recursion stops when the level is equal to the `max_recurse_level`.

        Returns:
            a list of all valid pages found. Invalid pages (wrong markup, empty HTML response, 404 errors) will be silently ignored.

        Examples:
            >>> from core import crawler
            >>> cr = crawler.Crawler()
            >>> pages = cr.get_website_from_crawling("https://aurelienpierre.com", default_lang="fr", markup=("div", { "class": "post-content" }))
        """
        written = 0
        index_url = radical_url(website + child)
        #print("trying", index_url)

        if self.discard_link(index_url):
            #print("no follow")
            return written

        # Abort now if the page was already crawled or recursion level reached
        if hash_with_category(index_url, category) in self.crawled_URL:
            #print("already crawled")
            return written

        if max_recurse_level > -1 and _recursion_level >= max_recurse_level:
            #print("max recursivity level reached")
            return written

        # Incremental update: skip pages that are still fresh according to self.since,
        # except of course for the recursion entry point which acts as our index.

        # is_internal_link means we are called from sitemap crawling following internal links
        is_internal_link = (_mainthread == False) and (_recursion_level == 0)

        # is_apex means we are at the first stage of recursion from true recursive crawling
        is_apex = (_mainthread == True) and (_recursion_level == 0)

        last_crawled = self._known_last_crawled(index_url) if (not is_apex and self.since is not None) else None
        if last_crawled is not None:
            if last_crawled >= (_normalize_tz(self.since) - relativedelta(months=3)):
                print(f"Skip (recent): {index_url}")
                self.update_link(index_url, index_url.strip("/"), category, 200)
                return written
            else:
                print(f"{index_url} has no crawling date or was crawled too long ago, will be recrawled")
        elif _recursion_level > 0:
            print(f"{index_url} is unknown")

        # Extract the domain name, to prepend it if we find relative URL while parsing
        address = patterns.split_url(website)
        if not address:
            print("%s can't be parsed as URL" % website)
            return written

        protocol, domain, page, params, anchor = address
        include = check_contains(contains_str, index_url)
        #print("processing", index_url, "include", include)

        # Init global robots.txt only at the root of the recursion
        if _recursion_level == 0 and _mainthread:
            super().__init__(protocol, domain, self.delay, 30)

        # Fetch and parse current (top-most) page
        content_type, status, new_url, custom_header, status_code = get_content_type(index_url, self)
        index_url = self.update_link(index_url, new_url, category, status_code)

        if self.discard_link(index_url) or not status:
            #print("no follow")
            return written

        # FIXME: we nest 7 levels of if here. It's ugly but I don't see how else
        # to cover so many cases.
        if "pdf" in content_type:
            #print("got pdf")
            written += self._parse_pdf_content(index_url, default_lang, category=category, delay=self)
            # No link to follow from PDF docmuents
            
        elif "text" in content_type \
            and "javascript" not in content_type \
            and "css" not in content_type \
            and "json" not in content_type:

            index, new_url, status_code = get_page_content(index_url, self, custom_header=custom_header)
            index_url = self.update_link(index_url, new_url, category, status_code)
            
            # Valid HTML response
            if index and index.body:
                # Detecting already-known pages based solely on URL is not always reliable,
                # since some URL params may change while still pointing to the same content,
                # though some URL params may query a different content.
                # Can't get any more clever with URLs, so we check content hash here.
                content_hash = hash_with_category(index.body.text, category)
                if content_hash in self.crawled_content:
                    return written
                else:
                    self.crawled_content.append(content_hash)

                #print("MARKUP:", markup, "URL:", index_url, "CATEGORY:", category, "RESTRICT:", restrict_section, "DOMAIN:", domain, "RECURSE:", _recursion_level)
                    
                # Parse current page content. _parse_original / _parse_translations funnel through
                # _emit, which applies content dedup-at-crawl (drop pages whose canonical content is
                # already indexed, under any URL, this crawl or a previous one) and either writes each
                # survivor directly to the DB (returning []) or returns it for the legacy list. Link-
                # following below is intentionally left untouched so the incremental crawl still
                # discovers new URLs from unchanged index pages.
                if include or _recursion_level == 0:
                    written += self._parse_original(index, index_url, default_lang, markup, None, category)
                    written += self._parse_translations(index, domain, index_url, markup, None, langs, category)
                    
                # Follow internal links whether or not this page was mined, if we didn't reach the final recursion level
                if _recursion_level + 1 != max_recurse_level:
                    internal_urls = self.get_unique_internal_url(index, domain, index_url)
                    #print(internal_urls)
                    for currentURL in internal_urls:
                        if hash_with_category(currentURL, category) in self.crawled_URL:
                            continue

                        current_address = patterns.split_url(currentURL)
                        if not current_address:
                            continue

                        current_protocol, current_domain, current_page, current_params, current_anchor = current_address

                        #print(current_page, current_page, current_params)
                        if not restrict_section and domain == current_domain:
                            # Recurse only through local pages, aka :
                            # 1. domains match
                            #print("recursing", currentURL)
                            written += self.get_website_from_crawling(
                                website, default_lang, child=current_page + current_params, langs=langs, markup=markup, contains_str=contains_str, mine_pdf=mine_pdf,
                                _recursion_level=_recursion_level + 1, max_recurse_level=max_recurse_level, restrict_section=restrict_section, category=category,
                                _mainthread=False)
                        elif restrict_section and domain == current_domain and child in current_page:
                            # Recurse only through local subsections, aka :
                            # 1. domains match
                            # 2. current page is in a subsection of current child
                            #print("recursing bis", currentURL, "domain:", domain, "-", current_domain, "child:", child, "-", current_page)
                            written += self.get_website_from_crawling(
                                website, default_lang, child=current_page + current_params, langs=langs, markup=markup, contains_str=contains_str, mine_pdf=mine_pdf,
                                _recursion_level=_recursion_level + 1, max_recurse_level=max_recurse_level, restrict_section=restrict_section, category=category,
                                _mainthread=False)
                        elif include and domain == current_domain and not restrict_section:
                            # Follow internal links on only one recursivity level
                            # Aka HTML reference pages (Wikipedia) and attached PDF (docs, manuals, spec sheets)
                            #print("following local", currentURL, "domain:", domain, "-", current_domain, "child:", child, "-", current_page)
                            written += self.get_website_from_crawling(
                                current_protocol + "://" + current_domain + current_page + current_params, default_lang, "", langs, contains_str="", max_recurse_level=1,
                                mine_pdf=mine_pdf, restrict_section=restrict_section, category=category, _recursion_level=0, _mainthread=False)
                        elif include and domain != current_domain:
                            # Follow external links on only one recursivity level.
                            # Aka HTML reference pages (Wikipedia) and attached PDF (docs, manuals, spec sheets)
                            #print("following distant", currentURL)
                            written += self.get_website_from_crawling(
                                current_protocol + "://" + current_domain + current_page + current_params, default_lang, "", langs, contains_str="", max_recurse_level=1,
                                mine_pdf=mine_pdf, restrict_section=restrict_section, category="external", _recursion_level=0, _mainthread=False)
                        else:
                            #print("discarding")
                            pass

                if mine_pdf:
                    written += self._parse_internal_pdfs(index, domain, index_url, default_lang, category)
                    
            else:
                # No index, aka no ParsedHTML HTML content.
                # Some websites display PDF in web applets on pages
                # advertising content-type=text/html but UTF8 codecs
                # fail to decode because it's actually not HTML but PDF.
                # If we end up here, it's most likely what we have.
                written += self._parse_pdf_content(index_url, default_lang, category=category, delay=self)
                #print("no page object")
        else:
            # Got an image, video, compressed file, binary, etc.
            #print("nothing done")
            pass

        # Process internal links found in pages
        if _mainthread:
            print("FINAL NUMBER of POSTS:", written)

        return written


    def get_website_from_sitemap(self,
                                 website: str,
                                 default_lang: str,
                                 sitemap: str = "/sitemap.xml",
                                 langs: tuple[str] = ("en", "fr"),
                                 markup: str | tuple | list[str] | list[tuple] = "body",
                                 category: str = "",
                                 contains_str: str | list[str] = "",
                                 internal_links: str = "any",
                                 mine_pdf: bool = False,
                                 _recursion_level: int = 0) -> int:
        """Recursively crawl all pages of a website from links found in a sitemap. 
        This applies to all HTML pages hosted on the domain of `website` and to PDF documents either from 
        the current domain or from external domains but referenced on HTML pages of the current domain. 
        Sitemaps of sitemaps are followed recursively.

        Arguments:
            website: root of the website, including `https://` or `http://` without trailing slash.
            default_lang: provided or guessed main language of the website content. Not used internally.
            sitemap: relative path of the XML sitemap.
            langs: 
                ISO-something 2-letters code of the languages for which we attempt to fetch the translation if available, 
                looking for the HTML `<link rel="alternate" hreflang="...">` tag.
            markup: see [core.parser.ParsedHTML.get_page_markup][]
            category: arbitrary category or label
            contains_str: 
                limit recursive crawling from sitemap-defined pages to pages containing this string or list of strings. 
                Will get passed as-is to [core.crawler.Crawler.get_website_from_crawling][].
            internal_links: defines what to do with links found inside the HTML page body/content.
                - `any`: follow and include all links found in page, no matter what domain they point to,
                - `internal`: follow and include links found in page only if they point to the same domain as the current page,
                - `external`: follow and include links found in page only if they point to a different domain than the current page,
                - `ignore`: don't follow internal links

        Returns:
            a list of all valid pages found. Invalid pages (wrong markup, empty HTML response, 404 errors) will be silently ignored.

        Examples:
            >>> from core import crawler
            >>> cr = crawler.Crawler()
            >>> pages = cr.get_website_from_sitemap("https://aurelienpierre.com", default_lang="fr", markup=("div", { "class": "post-content" }))
        """
        written = 0

        index_url = website + sitemap

        # Site-level change gate: a full-redeploy static site stamps sitemap.xml with the same deploy
        # mtime as its pages, so one conditional request tells us whether ANYTHING changed since our
        # last crawl. On 304 we skip the entire site with a single request instead of probing every
        # URL; on any other outcome we crawl normally and let per-page content-hash dedup filter.
        if self._sitemap_unmodified(index_url, self._site_last_crawled(website)):
            print(f"[sitemap-gate] {index_url} unmodified since last crawl — skipping {website} entirely")
            return written

        content_type, status, new_url, custom_header, status_code = get_content_type(index_url, self, bypass_robots_txt=True)
        index_url = self.update_link(index_url, new_url, category, status_code)

        if not status:
            return written

        index_page, new_url, status_code = get_page_content(index_url, self, custom_header=custom_header)
        if index_page is None:
            self.update_link(index_url, new_url, category, status_code)
            return written

        address = patterns.split_url(website)
        if not address:
            print("%s can't be parsed as URL" % website)
            return written

        protocol, domain, page, params, anchor = address
        super().__init__(protocol, domain, self.delay, 30)

        # Sitemaps of sitemaps enclose elements in `<sitemap> </sitemap>`
        # While sitemaps of pages enclose them in `<url> </url>`.
        # In both cases, we find URL in `<loc>` and dates in `<lastmod>`
        print("%i sitemaps found in sitemap" % len(index_page.find_all('sitemap')))
        print("%i URLs found in sitemap" % len(index_page.find_all('url')))

        # We got a sitemap of sitemaps, recurse over the sub-sitemaps
        for link in index_page.find_all('sitemap'):
            url = link.find("loc").get_text()
            print(url)
            _sitemap = re.sub(r"(http)?s?(\:\/\/)?%s" % domain, "", url)
            written += self.get_website_from_sitemap(website, default_lang, sitemap=_sitemap, langs=langs, markup=markup, category=category, internal_links=internal_links, _recursion_level=_recursion_level+1)

        # Process pages
        for link in index_page.find_all('url'):
            written += self._sitemap_process(domain, website, sitemap, link, default_lang, langs, markup, category, internal_links, contains_str, mine_pdf, _recursion_level)

        return written


    def emit(self, pages: list) -> int:
        """Public entry point for pages built OUTSIDE the crawl loop (e.g. from local files/exports):
        applies content dedup-at-crawl and writes each survivor into the DB. Returns the count
        written. Thin wrapper over the internal `_emit` sink."""
        return self._emit(pages)


    def get_pdf(self, url: str, default_lang: str = "en", category: str = "",
                file_path: str | None = None, custom_header: dict = {}, **kwargs) -> int:
        """Crawl a single PDF document (from the network, or from *file_path* if given) straight
        into the DB. DB-native: the parsed page goes through `_emit` (content dedup-at-crawl +
        buffered write); returns the number of pages written. Public entry point for PDF-only
        scrapers that previously did `crawler.get_pdf_content(...)` + `save_data`."""
        return self._emit(get_pdf_content(url, default_lang, delay=self, file_path=file_path,
                                          category=category, custom_header=custom_header, **kwargs))


    def get_unique_internal_url(self, page: ParsedHTML, domain: str, currentURL:str) -> list[str]:
        """Grab the internal links found in page, except PDF, and return only the ones we don't already know"""
        # Get a set of unique absolute URLs
        links = {relative_to_absolute(url, domain, currentURL) for url in page.links}

        # Get URLs that are neither already crawled or already on the list nor PDF
        return list({url 
                     for url in links
                     if not self.discard_link(url) and ".pdf" not in url.lower()})


    def _sitemap_process(self, domain, website, sitemap, link, default_lang, langs, markup, category, internal_links, contains_str, mine_pdf, _recursion_level) -> int:
        written = 0
        url = link.find("loc")
        date = link.find("lastmod")

        if not url:
            print("No URL found in ", link)
            # Nothing to process, ignore this item
            return written

        date = date.get_text() if date else None

        currentURL = relative_to_absolute(url.get_text(), domain, website + sitemap)
        include = check_contains(contains_str, currentURL)

        if not include or self.discard_link(currentURL):
            return written

        # Incremental update: skip pages that haven't changed since last crawl.
        # Priority: sitemap's own <lastmod> field > self.since fallback.
        stripped_url = currentURL.strip("/")
        last_crawled = self._known_last_crawled(currentURL)
        if last_crawled is not None:
            last_crawled = _normalize_tz(last_crawled)
            if date:
                try:
                    lastmod = _parse_iso_date(date)
                    if last_crawled >= lastmod:
                        print(f"Skip (unchanged): {currentURL}")
                        self.update_link(currentURL, stripped_url, category, 200)
                        return written
                except (ValueError, TypeError):
                    print(f"{currentURL} has unparseable date, will be recrawled")
                    pass  # unparseable date — crawl anyway
            # No usable <lastmod> in this sitemap entry: there is no reliable per-page date to skip
            # on (see _sitemap_unmodified). We deliberately do NOT skip by last-crawled here — the
            # cheap whole-site skip is the sitemap gate in get_website_from_sitemap, and if we
            # reached this page the site was (re)deployed since our last crawl, so we must re-fetch
            # and let content-hash dedup (_emit) decide whether the content actually changed.
        else:
            print(f"{currentURL} is unknown")

        self.crawled_URL.append(hash_with_category(currentURL, category))
        content_type, status, new_url, custom_header, status_code = get_content_type(currentURL, self, bypass_robots_txt=True)
        currentURL = self.update_link(currentURL, new_url, category, status_code)

        if not status:
            return written

        page, new_url, status_code = get_page_content(currentURL, self, custom_header=custom_header)
        currentURL = self.update_link(currentURL, new_url, category, status_code)

        if page is not None:
            # We got a proper web page, parse it
            written += self._parse_original(page, currentURL, default_lang, markup, date, category)
            written += self._parse_translations(page, domain, currentURL, markup, date, langs, category)
            written += self._parse_internal_pdfs(page, domain, currentURL, default_lang, category)

            # Follow internal and external links found in body
            # Since this is recursion from whithin page links, we have to flag the category as "external"
            # to distinguish from pages crawled from the sitemap in case we get both flavours.
            # The rationale is pages crawled from sitemap may target selective (clean) HTML tags, and produce better quality
            # data/content than external recursively-crawled pages, that fetch the whole <body>
            # (including non-data/formatting, like sidebars, nav menus, etc.).
            written += self.get_immediate_links(self.get_unique_internal_url(page, domain, currentURL), domain, default_lang, langs, "external", contains_str, internal_links=internal_links, mine_pdf=mine_pdf)

        return written


    def get_youtube_channels(self,
                              channel_ids: list[str],
                              api_key: str,
                              default_lang: str = "en",
                              category: str = "video",
                              since: datetime.datetime | None = None) -> int:
        """Index YouTube channels via the Data API v3 (no OAuth required).

        Retrieves the full upload list for each channel by walking the channel's
        uploads playlist, then fetches the complete snippet for each video.  The
        result mirrors what [get_website_from_sitemap][core.crawler.Crawler.get_website_from_sitemap] 
        produces for a normal
        website: one [core.types.web_page][] per video, with ``title``,
        ``content`` (video description), ``date``, ``lang``, and ``category``
        populated.

        Incremental update logic:

        - If *since* is provided, any video URL already present in ``self.known_urls``
          and last crawled **on or after** *since* is skipped.
        - If *since* is ``None`` but ``self.since`` is set, ``self.since`` is used
          as the cut-off.
        - Videos not yet in ``self.known_urls`` are always fetched.

        Rate limiting respects ``self.delay`` and the ``www.googleapis.com`` domain
        bucket, consistent with the rest of the crawler.

        Arguments:
            channel_ids:
                list of YouTube channel IDs — the ``UC…`` string visible in any
                channel URL (``youtube.com/channel/UC…``).
            api_key:
                Google Cloud API key with *YouTube Data API v3* enabled.
                See https://developers.google.com/youtube/v3/getting-started.
            default_lang:
                fallback language code when the video metadata does not declare one.
            category:
                label applied to every indexed video, reused by search filters.
            since:
                skip videos whose URL is already known and was crawled on or after
                this datetime.  Pass ``None`` (default) to (re-)index everything.

        Returns:
            list of [core.types.web_page][] objects, one per video.

        Example:
            ```python
            db = database.open_db("my-engine.db")
            with crawler.Crawler(delay=0.5) as cr:
                cr.load_known_urls(db)
                pages = cr.get_youtube_channels(
                    channel_ids = ["UCmsSn3fujI81EKEr4NLxrcg",
                                   "UCkqe4BYsllmcxo2dsF-rFQw"],
                    api_key     = "YOUR_KEY",
                    default_lang = "en",
                    category    = "video",
                    since       = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc),
                )
            database.populate_db(db, pages)
            ```
        """
        written = 0
        effective_since = since or self.since
        cutoff = _normalize_tz(effective_since) if effective_since else None

        for channel_id in channel_ids:
            print(f"[YouTube] Channel {channel_id}")
            self.sleep("www.googleapis.com")
            try:
                channel_resp = requests.get(
                    "https://youtube.googleapis.com/youtube/v3/channels"
                    f"?part=contentDetails&id={channel_id}&key={api_key}",
                    timeout=30,
                )
                channel_data = json.loads(channel_resp.content)
            except Exception as e:
                print(f"[YouTube] Failed to fetch channel {channel_id}: {e}")
                continue

            if not channel_data.get("items"):
                print(f"[YouTube] Channel {channel_id}: no items returned (invalid ID or key?)")
                continue

            playlist_id = channel_data["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
            page_token = ""

            while True:
                self.sleep("www.googleapis.com")
                try:
                    playlist_resp = requests.get(
                        "https://www.googleapis.com/youtube/v3/playlistItems"
                        f"?part=snippet,contentDetails&maxResults=50"
                        f"&playlistId={playlist_id}&key={api_key}"
                        + (f"&pageToken={page_token}" if page_token else ""),
                        timeout=30,
                    )
                    playlist_data = json.loads(playlist_resp.content)
                except Exception as e:
                    print(f"[YouTube] Failed to fetch playlist page: {e}")
                    break

                for item in playlist_data.get("items", []):
                    snippet    = item["snippet"]
                    video_id   = snippet["resourceId"]["videoId"]
                    video_url  = f"https://www.youtube.com/watch?v={video_id}"
                    published  = snippet.get("publishedAt", "")

                    # Incremental skip
                    if cutoff is not None:
                        vlast = self._known_last_crawled(video_url)
                        if vlast is not None and vlast >= cutoff:
                            continue

                    if hash_with_category(video_url, category) in self.crawled_URL:
                        continue
                    self.crawled_URL.append(hash_with_category(video_url, category))

                    # Fetch the full video snippet (richer than the playlist item snippet)
                    self.sleep("www.googleapis.com")
                    try:
                        detail_resp = requests.get(
                            "https://youtube.googleapis.com/youtube/v3/videos"
                            f"?part=snippet&id={video_id}&key={api_key}",
                            timeout=30,
                        )
                        detail_data = json.loads(detail_resp.content)
                        if detail_data.get("items"):
                            full = detail_data["items"][0]["snippet"]
                            description = full.get("description") or snippet.get("description", "")
                            lang        = full.get("defaultLanguage", default_lang)
                        else:
                            description = snippet.get("description", "")
                            lang        = default_lang
                    except Exception as e:
                        print(f"[YouTube] Failed to fetch video details for {video_id}: {e}")
                        description = snippet.get("description", "")
                        lang        = default_lang

                    title = snippet.get("title", "")
                    if not title or not description:
                        continue

                    page = sanitize_web_page(web_page(
                        title    = title,
                        url      = video_url,
                        excerpt  = description[:800],
                        content  = description,
                        date     = published,
                        lang     = lang,
                        category = category,
                        h1       = [title],
                        h2       = [],
                        crawled  = datetime.datetime.now(datetime.timezone.utc),
                    ))
                    written += self._emit([page])
                    print(f"[YouTube] {page['url']}: {page['title']}")

                page_token = playlist_data.get("nextPageToken", "")
                if not page_token:
                    break

            print(f"[YouTube] Channel {channel_id}: {written} videos indexed so far")

        return written


    def get_github_repositories(self,
                                 repositories: list[tuple[str, str]],
                                 api_key: str,
                                 features: list[str] | None = None,
                                 langs: tuple[str, ...] = ("en", "fr"),
                                 category: str = "Github",
                                 since: datetime.datetime | None = None,
                                 mine_pdf: bool = True) -> int:
        """Index GitHub repository content via the REST API.

        Supported *features*: ``"issues"``, ``"pulls"``, ``"commits"``,
        ``"discussions"``.  Issue and pull-request comments are concatenated with
        the parent body.  External links found in Markdown bodies are followed at
        one recursion level (same behaviour as 
        [get_website_from_crawling][core.crawler.Crawler.get_website_from_crawling] with
        ``max_recurse_level=1``), and PDF files linked from those pages are mined
        when *mine_pdf* is ``True``.

        Incremental update:

        - For ``issues``, ``pulls``, and ``commits``, the GitHub API's native
          ``?since=`` query parameter is used when *since* is provided, so only
          items updated after that date are fetched — minimising API quota usage.
        - For ``discussions``, the REST API has no ``since`` filter; client-side
          filtering by ``created_at`` is applied instead.
        - 429 / 403 rate-limit responses are handled automatically: the crawler
          reads the ``Retry-After`` header and waits accordingly.

        Arguments:
            repositories:
                list of ``(owner, repo)`` tuples,
                e.g. ``[("aurelienpierreeng", "ansel"), ("darktable-org", "darktable")]``.
            api_key:
                GitHub personal access token (classic or fine-grained, read-only
                ``repo`` scope is sufficient).
                See https://docs.github.com/en/rest/authentication.
            features:
                subset of ``["issues", "pulls", "commits", "discussions"]`` to index.
                Defaults to all four when ``None``.
            langs:
                language codes passed through to 
                [get_immediate_links][core.crawler.Crawler.get_immediate_links] when
                following external links from item bodies.
            category:
                label applied to every indexed item.
            since:
                only fetch items created or updated after this datetime.
                Overrides ``self.since`` when provided.
            mine_pdf:
                whether to follow and extract PDF files linked from item bodies.

        Returns:
            list of [core.types.web_page][] objects.

        Example:
            ```python
            db = database.open_db("my-engine.db")
            with crawler.Crawler(delay=0.72) as cr:
                cr.load_known_urls(db)
                pages = cr.get_github_repositories(
                    repositories = [("aurelienpierreeng", "ansel"),
                                    ("darktable-org", "rawspeed")],
                    api_key      = "ghp_…",
                    features     = ["issues", "pulls", "commits"],
                    since        = datetime.datetime(2025, 1, 1,
                                                     tzinfo=datetime.timezone.utc),
                    mine_pdf     = True,
                )
            database.populate_db(db, pages)
            ```
        """
        if features is None:
            features = ["issues", "pulls", "commits", "discussions"]

        written = 0
        posts_per_page = 100  # GitHub API maximum

        effective_since = since or self.since
        since_str = (
            effective_since.strftime("%Y-%m-%dT%H:%M:%SZ")
            if effective_since else None
        )

        gh_headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # ── internal helpers ──────────────────────────────────────────────────

        def _fetch_page(url: str) -> list:
            """Single paginated request; handles rate-limit back-off."""
            self.sleep("api.github.com")
            try:
                resp = requests.get(url, headers=gh_headers, timeout=30)
                if resp.status_code in (403, 429):
                    wait = int(resp.headers.get("Retry-After", 60))
                    print(f"[GitHub] Rate limited — waiting {wait} s…")
                    time.sleep(wait)
                    resp = requests.get(url, headers=gh_headers, timeout=30)
                if resp.status_code != 200:
                    print(f"[GitHub] HTTP {resp.status_code} for {url}")
                    return []
                data = json.loads(resp.content)
                # Some endpoints return a dict (e.g. error) instead of a list
                return data if isinstance(data, list) else []
            except Exception as e:
                print(f"[GitHub] Request failed: {url} — {e}")
                return []

        def _fetch_all_pages(base_url: str) -> list:
            """Walk all pages of a paginated GitHub endpoint."""
            results, page = [], 1
            sep = "&" if "?" in base_url else "?"
            while True:
                items = _fetch_page(f"{base_url}{sep}per_page={posts_per_page}&page={page}")
                if not items:
                    break
                results.extend(items)
                if len(items) < posts_per_page:
                    break
                page += 1
            return results

        def _item_to_pages(item_url: str, title: str, body: str,
                            date: str | None) -> int:
            """Parse a Markdown body into web_page objects and follow external links."""
            if hash_with_category(item_url, category) in self.crawled_URL:
                return 0
            self.crawled_URL.append(hash_with_category(item_url, category))

            html = (
                f"<title>{title}</title><body>\n\n"
                + markdown.markdown(body or "")
                + "\n\n</body>"
            )
            entry, _, _ = get_page_content(None, self, content=html)
            if entry is None:
                return 0

            pages = self._emit(parse_page(entry, item_url, "en", "body", date, category, tokenizer=self.tokenizer))

            # Extract bare URLs from raw Markdown (regex match covers URLs that are
            # not wrapped in Markdown link syntax and thus absent from the rendered HTML)
            entry.links += [
                patterns.remove_unmatched_parentheses(m.group(0))
                for m in patterns.URL_PATTERN.finditer(body or "", concurrent=True)
            ]

            # Follow external links at one recursion level
            unique_links = self.get_unique_internal_url(entry, "github.com", item_url)
            pages += self.get_immediate_links(
                unique_links, "github.com", "en", langs,
                category, "", internal_links="any", mine_pdf=mine_pdf,
            )

            return pages

        # ── main loop ────────────────────────────────────────────────────────

        for owner, repo in repositories:
            print(f"[GitHub] Crawling {owner}/{repo}")

            for feature in features:
                print(f"[GitHub] Fetching {feature}…")

                # Build the API URL.
                # Only issues and commits support a server-side ?since= filter.
                # The pulls endpoint silently ignores ?since=, so PRs are filtered
                # client-side below (same path as discussions).
                since_param = f"&since={since_str}" if since_str else ""
                base_url = f"https://api.github.com/repos/{owner}/{repo}/{feature}?state=all{since_param}&sort=updated"
                items = _fetch_all_pages(base_url)

                print(f"[GitHub] {len(items)} {feature} fetched for {owner}/{repo}")

                for item in items:
                    item_url = item.get("html_url", "")
                    if not item_url:
                        continue
 
                    # Client-side date filter for pulls and discussions.
                    # Pulls: the REST API ignores ?since=, so we filter by updated_at here.
                    # Discussions: the REST API has no since filter at all.
                    if feature in ("pulls", "discussions") and effective_since:
                        updated = item.get("updated_at") or item.get("created_at", "")
                        try:
                            if _normalize_tz(_parse_iso_date(updated)) < _normalize_tz(effective_since):
                                continue
                        except (ValueError, TypeError):
                            pass

                    # Assemble title, body, date
                    body_parts: list[str] = []

                    if feature == "commits":
                        commit = item.get("commit", {})
                        msg    = commit.get("message", "")
                        title  = msg.split("\n\n")[0] or item_url.split("/")[-1]
                        body_parts.append(msg)
                        date   = commit.get("committer", {}).get("date")
                    else:
                        title = item.get("title", item_url.split("/")[-1])
                        body_parts.append(item.get("body") or "")
                        date  = item.get("created_at")

                    full_title = f"{feature.capitalize()}: {title}"

                    # Fetch comments
                    comments_url = None
                    if feature in ("issues", "pulls", "commits"):
                        comments_url = item.get("comments_url", "")
                    elif feature == "discussions":
                        comments_url = f"https://api.github.com/repos/{owner}/{repo}/{feature}/{item.get("number", "")}/comments"

                    if comments_url:
                        self.crawled_URL.append(hash_with_category(comments_url, category))
                        for comment in _fetch_page(comments_url):
                            if comment.get("body"):
                                body_parts.append(comment["body"])

                    full_body = "\n\n---\n\n".join(filter(None, body_parts))
                    written += _item_to_pages(item_url, full_title, full_body, date)

            print(f"[GitHub] {owner}/{repo}: {written} total pages indexed so far")

        return written


    def get_stackexchange_posts(self,
                                 site: str,
                                 api_key: str | None = None,
                                 category: str = "forum",
                                 langs: tuple[str, ...] = ("en",),
                                 since: datetime.datetime | None = None,
                                 window_days: int = 90,
                                 earliest_date: datetime.datetime | None = None,
                                 se_filter: str = "!14e92L7CSAvro*ufn5-s.s23LqfumIAci09lv0z)*cLWPr") -> int:
        """Index a Stack Exchange community via the public API v2.3.

        Retrieves all posts (questions, answers) together with their embedded
        comments from the `posts` endpoint.  Each post's body and its comments
        are concatenated into a single [core.types.web_page][] and external
        links found in the Markdown bodies are followed at one recursion level
        (PDFs included).

        **Pagination and rate limits.**  Without an API key the SE API allows
        300 requests/day and a maximum of 25 pages per date window.  With a key,
        the daily quota rises to 10 000 requests.  The method handles both
        cases: it pages through 25-page windows, each covering *window_days*
        days of posts, sliding backward in time until *earliest_date* is
        reached.  When *since* is provided the window collapses to a single
        forward pass from *since* to now, which is the efficient path for
        incremental updates.  The API's ``backoff`` field is always respected.

        **Incremental update.**  Two complementary mechanisms combine:

        - *since* (or ``self.since``) is passed as ``fromdate`` to the API, so
          the server only returns posts created or edited after that point.
        - ``self.known_urls`` provides per-URL precision: for each post the
          ``last_edit_date`` field is compared with the stored crawl timestamp,
          and the post is skipped when the stored timestamp is more recent —
          catching the case where a post was fetched as part of a wide window
          but not actually changed.

        **SE filter.**  The default *se_filter* string was built at
        `api.stackexchange.com/docs/filters` and requests the following fields:
        ``body_markdown``, ``comments``, ``comments.body_markdown``,
        ``comments.link``, ``creation_date``, ``last_edit_date``, ``link``,
        ``title``.  Pass a custom filter string if you need additional fields.

        Arguments:
            site:
                Stack Exchange site name as used in the API, e.g. ``"photo"``,
                ``"stackoverflow"``, ``"unix"``, ``"electronics"``.
                Sites with standalone domains (``stackoverflow.com``,
                ``superuser.com``, ``serverfault.com``, ``askubuntu.com``,
                ``mathoverflow.net``) are resolved automatically.
            api_key:
                Optional Stack Exchange API key.  Raises daily quota from 300
                to 10 000 requests/day.  Obtain one free at
                https://stackapps.com/apps/oauth/register.
            category:
                Label applied to every indexed post.
            langs:
                Language codes passed to [get_immediate_links][core.crawler.Crawler.get_immediate_links] when following
                external links from post bodies.
            since:
                Only fetch posts whose ``creation_date`` or ``last_edit_date``
                is at or after this datetime.  Passed as ``fromdate`` to the
                API.  Overrides ``self.since`` when provided.
            window_days:
                Size (in days) of each date window used when doing a full crawl
                (i.e. when *since* is ``None``).  Smaller windows mean more API
                requests but fewer items per page, reducing the chance of
                hitting the 25-page cap.  Default ``90``.
            earliest_date:
                Stop the full-crawl backward walk when this date is reached.
                Defaults to 2010-01-01 (SE's approximate launch date).
            se_filter:
                Opaque SE filter string defining which fields are returned.
                Override only when you need fields beyond the defaults.

        Returns:
            list of [core.types.web_page][] objects.

        Example:
            ```python
            db = database.open_db("my-engine.db")
            with crawler.Crawler(delay=1.0) as cr:
                cr.load_known_urls(db)
                pages = cr.get_stackexchange_posts(
                    site     = "photo",
                    api_key  = "YOUR_SE_APP_KEY",
                    category = "forum",
                    since    = datetime.datetime(2025, 1, 1,
                                                 tzinfo=datetime.timezone.utc),
                )
            database.populate_db(db, pages)
            ```
        """
        # Standalone-domain sites — domain is not {site}.stackexchange.com
        _SE_STANDALONE = {
            "stackoverflow": "stackoverflow.com",
            "superuser":     "superuser.com",
            "serverfault":   "serverfault.com",
            "askubuntu":     "askubuntu.com",
            "mathoverflow":  "mathoverflow.net",
        }
        se_domain = _SE_STANDALONE.get(site.lower(), f"{site}.stackexchange.com")

        effective_since = since or self.since
        _earliest = earliest_date or datetime.datetime(2010, 1, 1, tzinfo=datetime.timezone.utc)
        written = 0

        # ── internal helpers ──────────────────────────────────────────────────

        def _item_to_pages(post: dict) -> int:
            """Convert a single SE API post dict to web_page objects."""
            post_url = post.get("link", "")
            if not post_url:
                return 0

            # Incremental: skip posts unchanged since last crawl
            last_edit_ts = post.get("last_edit_date") or post.get("creation_date")
            post_last_crawled = self._known_last_crawled(post_url) if last_edit_ts else None
            if post_last_crawled is not None:
                try:
                    post_dt = _normalize_tz(
                        datetime.datetime.fromtimestamp(last_edit_ts,
                                                        tz=datetime.timezone.utc)
                    )
                    if post_last_crawled >= post_dt:
                        return 0
                except (ValueError, OSError):
                    pass

            if hash_with_category(post_url, category) in self.crawled_URL:
                return 0
            self.crawled_URL.append(hash_with_category(post_url, category))

            # Mark comment URLs as already visited so recursive crawling skips them
            for comment in post.get("comments", []):
                if "link" in comment:
                    self.crawled_URL.append(
                        hash_with_category(comment["link"], category)
                    )

            # Build body: main post + all comments
            body_parts: list[str] = []
            if post.get("body_markdown"):
                body_parts.append(post["body_markdown"])
            for comment in post.get("comments", []):
                if comment.get("body_markdown"):
                    body_parts.append(comment["body_markdown"])

            full_body = "\n\n".join(body_parts)
            title = post.get("title", "")

            date_ts = post.get("last_edit_date") or post.get("creation_date")
            date    = (datetime.datetime
                       .fromtimestamp(date_ts, tz=datetime.timezone.utc)
                       .isoformat()
                       if date_ts else None)

            html = (
                f"<title>{title}</title><body>\n\n"
                + markdown.markdown(full_body)
                + "\n\n</body>"
            )
            entry, _, _ = get_page_content(None, self, content=html)
            if entry is None:
                return 0

            pages = self._emit(parse_page(entry, post_url, "en", "body", date, category, tokenizer=self.tokenizer))

            # Add bare URLs from Markdown text; exclude internal SE links
            # (those are covered by API pagination, not link-following)
            entry.links += [
                m.group(0)
                for m in patterns.URL_PATTERN.finditer(full_body, concurrent=True)
                if f"https://{se_domain}/" not in m.group(0)
            ]

            # Follow external links only
            unique_links = self.get_unique_internal_url(entry, se_domain, post_url)
            pages += self.get_immediate_links(
                unique_links, se_domain, "en", langs,
                "external", "", internal_links="external", mine_pdf=True,
            )

            return pages

        def _process_window(fromdate: datetime.datetime,
                             todate: datetime.datetime) -> tuple[bool, bool]:
            """Fetch all pages in one date window.

            Returns:
                (quota_exhausted, should_continue_outer_loop)
            """
            nonlocal written  # accumulate into the enclosing get_stackexchange_posts counter
            page = 1
            key_param = f"&key={api_key}" if api_key else ""
            from_ts = int(fromdate.timestamp())
            to_ts   = int(todate.timestamp())

            while True:
                url = (
                    f"https://api.stackexchange.com/2.3/posts"
                    f"?page={page}"
                    f"&fromdate={from_ts}&todate={to_ts}"
                    f"&order=desc&sort=activity"
                    f"&site={site}"
                    f"&filter={se_filter}"
                    f"{key_param}"
                )
                self.sleep(f"api.stackexchange.com")
                try:
                    resp = requests.get(url, timeout=60)
                    print(f"[SE:{site}] {url} → {resp.status_code}")
                except Exception as e:
                    print(f"[SE:{site}] Request failed: {e}")
                    return False, False

                if resp.status_code != 200:
                    print(f"[SE:{site}] HTTP {resp.status_code}, stopping")
                    return False, False

                data = json.loads(resp.content)

                for post in data.get("items", []):
                    written += _item_to_pages(post)

                quota_remaining = data.get("quota_remaining", 1)
                print(f"[SE:{site}] page {page}, quota remaining: {quota_remaining}")

                # Mandatory backoff — violating this can get the app suspended
                if "backoff" in data:
                    wait = int(data["backoff"])
                    print(f"[SE:{site}] Backoff requested: waiting {wait} s…")
                    time.sleep(wait)

                if quota_remaining <= 0:
                    print(f"[SE:{site}] Daily quota exhausted")
                    return True, False

                if not data.get("has_more", False):
                    return False, True   # window done, continue outer loop

                page += 1

        # ── main crawl logic ─────────────────────────────────────────────────

        if effective_since is not None:
            # Incremental mode: single forward pass from since → now
            now = datetime.datetime.now(datetime.timezone.utc)
            print(f"[SE:{site}] Incremental crawl from {effective_since} to {now}")
            _process_window(
                _normalize_tz(effective_since),
                now,
            )

        else:
            # Full crawl: slide a window backward from now to earliest_date
            todate   = datetime.datetime.now(datetime.timezone.utc)
            fromdate = todate - datetime.timedelta(days=window_days)

            print(f"[SE:{site}] Full crawl backward from {todate} to {_earliest}")

            while fromdate >= _normalize_tz(_earliest):
                quota_exhausted, carry_on = _process_window(fromdate, todate)

                if quota_exhausted:
                    break

                todate   = fromdate
                fromdate = todate - datetime.timedelta(days=window_days)

        print(f"[SE:{site}] Total pages indexed: {written}")
        return written


    def _parse_pdf_content(self, link, default_lang, delay: DelayedClass, category="", custom_header={}, ) -> int:
        return self._emit(get_pdf_content(link, default_lang, category=category, custom_header=custom_header, delay=delay))


    def _parse_original(self, page, url, default_lang, markup, date, category) -> int:
        return self._emit(parse_page(page, url, default_lang, markup=markup, date=date, category=category,
                                     tokenizer=self.tokenizer)) if page else 0


    def _parse_translations(self, page, domain, current_url, markup, date, langs, category) -> int:
        """Follow `<link rel="alternate" hreflang="lang" href="url">` tags declaring links to alternative language variants for the current HTML page and crawl the target pages. This works only for pages properly defining alternatives in HTML header."""
        written = 0

        if not page:
            return written

        for lang in langs:
            link_tag = page.find('link', {'rel': 'alternate', 'hreflang': lang})

            if link_tag and "href" in link_tag and link_tag["href"]:
                translatedURL = relative_to_absolute(link_tag["href"], domain, current_url)
                self.crawled_URL.append(hash_with_category(translatedURL, category))

                content_type, status, new_url, custom_header, status_code = get_content_type(translatedURL, self, bypass_robots_txt=True)
                translatedURL = self.update_link(translatedURL, new_url, category, status_code)

                if "text" in content_type and status:
                    translated_page, new_url, status_code = get_page_content(translatedURL, self, custom_header=custom_header)
                    translatedURL = self.update_link(translatedURL, new_url, category, status_code)

                    if translated_page is not None:
                        written += self._parse_original(translated_page, translatedURL, lang, markup, date, category)


        return written


    def _parse_internal_pdfs(self, page, domain, current_url, default_lang, category) -> int:
        written = 0
        pdfs = [relative_to_absolute(url, domain, current_url) for url in page.links]
        pdfs = [url for url in set(pdfs)
                if ".pdf" in url.lower() and not self.discard_link(url)]

        for currentURL in pdfs:
            content_type, status, new_url, custom_header, status_code = get_content_type(currentURL, self)
            currentURL = self.update_link(currentURL, new_url, category, status_code)

            if status and "pdf" in content_type:
                written += self._parse_pdf_content(currentURL, default_lang, category=category, custom_header=custom_header, delay=self)

        return written


    # ── External-source adapters ───────────────────────────────────────────────────────
    # Legitimate, bot-friendly routes to content whose live origin blocks automated access or has
    # gone offline: a scholarly-metadata API (OpenAlex) and the Internet Archive (Wayback). Neither
    # touches the protected origin; both are public services intended for exactly this use.

    @staticmethod
    def _reconstruct_abstract(inv: dict | None) -> str:
        """Rebuild plain text from OpenAlex's `abstract_inverted_index` {word: [positions]}."""
        if not inv:
            return ""
        n = 1 + max((p for ps in inv.values() for p in ps), default=-1)
        words = [""] * n
        for w, positions in inv.items():
            for p in positions:
                if 0 <= p < n:
                    words[p] = w
        return " ".join(w for w in words if w)

    def get_openalex_works(self, source_ids: list[str], category: str = "reference",
                           mailto: str = "", per_page: int = 200, max_works: int | None = None,
                           default_lang: str = "en") -> int:
        """Crawl works (title + reconstructed abstract) from the given OpenAlex source (journal) IDs
        straight into the DB. OpenAlex (api.openalex.org) is a free, unauthenticated index of
        scholarly metadata — a legitimate route to open-access academic content whose publisher
        origin blocks automated access (e.g. MDPI behind Akamai). Pass *mailto* for OpenAlex's
        faster "polite pool". *max_works* caps per source (None = all)."""
        written = 0
        # Preload the DOIs already indexed for this dataset ONCE, so the per-work resumable skip is
        # an in-memory O(1) test. (Calling _known_last_crawled per work would `WHERE url=? OR
        # wayback=?` full-scan the whole canonical every time — catastrophic across 80k works.)
        existing = set()
        if self.db is not None and self.dataset_name:
            existing = set(u for (u,) in self.db.execute(
                "SELECT url FROM pages WHERE dataset LIKE ?", (f"%,{self.dataset_name},%",)))
            print(f"[OpenAlex] {len(existing)} works already indexed for {self.dataset_name!r} — skipping those")

        for source_id in source_ids:
            cursor, indexed = "*", 0
            print(f"[OpenAlex] crawling source {source_id}")
            while cursor:
                flt = f"primary_location.source.id:{source_id},has_abstract:true,type:article"
                url = (f"https://api.openalex.org/works?filter={flt}"
                       f"&per-page={per_page}&cursor={cursor}"
                       + (f"&mailto={mailto}" if mailto else ""))
                # Fetch this page with retry/backoff on transient errors (429/5xx, network) so one
                # gateway hiccup mid-journal doesn't truncate an 80k-work source.
                data = None
                for attempt in range(6):
                    self.sleep("api.openalex.org", 1.0)
                    try:
                        resp = requests.get(url, timeout=60, headers=HEADER)
                        if resp.status_code == 200:
                            data = resp.json()
                            break
                        if resp.status_code in (429, 500, 502, 503, 504):
                            wait = min(90, 5 * 2 ** attempt)
                            print(f"[OpenAlex] HTTP {resp.status_code} for {source_id}, retry in {wait}s ({attempt+1}/6)")
                            time.sleep(wait)
                            continue
                        print(f"[OpenAlex] HTTP {resp.status_code} for {source_id}, stopping")
                        break
                    except Exception as e:
                        wait = min(90, 5 * 2 ** attempt)
                        print(f"[OpenAlex] request error ({e}), retry in {wait}s ({attempt+1}/6)")
                        time.sleep(wait)
                if data is None:
                    print(f"[OpenAlex] giving up on {source_id} after retries")
                    break
                for w in data.get("results", []):
                    link = w.get("doi") or w.get("id")
                    # Resumable: skip works already indexed (in-memory O(1)); a re-run picks up where
                    # a truncated crawl stopped without re-parsing anything.
                    if not link or link in existing:
                        continue
                    existing.add(link)  # also dedups a DOI shared across journals within this run
                    title = (w.get("display_name") or "").strip()
                    abstract = self._reconstruct_abstract(w.get("abstract_inverted_index"))
                    if not title or not abstract:
                        continue
                    date = w.get("publication_date")
                    lang = w.get("language") or default_lang
                    safe = abstract.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html = f"<title>{title}</title><body>\n\n{safe}\n\n</body>"
                    entry, _, _ = get_page_content(None, self, content=html)
                    if entry is None:
                        continue
                    written += self._emit(parse_page(entry, link, lang, "body", date, category,
                                                     tokenizer=self.tokenizer))
                    indexed += 1
                    if max_works and indexed >= max_works:
                        break
                print(f"[OpenAlex] {source_id}: {indexed} works indexed so far")
                if max_works and indexed >= max_works:
                    break
                cursor = data.get("meta", {}).get("next_cursor")
        return written

    def get_wayback_pages(self, url_prefix: str, default_lang: str = "en",
                          markup: str | tuple | list = "body", category: str = "reference",
                          limit: int | None = None, clean_url=None) -> int:
        """Crawl pages archived in the Internet Archive (Wayback Machine) whose ORIGINAL URL matches
        *url_prefix* (a CDX prefix, e.g. ``site.org/path*``). A legitimate route to content whose
        live origin is offline or bot-protected: it enumerates the archived HTML-200 captures via the
        public CDX API and fetches each RAW snapshot (``…/web/<ts>id_/<url>`` — no Wayback toolbar),
        storing it under its original URL. Already-indexed URLs are skipped (archived pages are
        stable). *clean_url(original)->str* optionally canonicalizes/dedupes noisy archived URLs."""
        written = 0
        cdx = ("http://web.archive.org/cdx/search/cdx?url=" + url_prefix +
               "&filter=statuscode:200&filter=mimetype:text/html&collapse=urlkey&fl=original,timestamp")
        # archive.org rate-limits the CDX API; under throttling it returns a non-200 or a truncated
        # body (a "successful" partial result that silently under-counts). Retry with backoff on any
        # error/non-200/empty. Truncation isn't perfectly detectable here, but this job skips
        # already-indexed URLs, so a later re-run picks up whatever a throttled response missed.
        lines = []
        for attempt in range(4):
            self.sleep("web.archive.org", 3.0)
            try:
                resp = requests.get(cdx, timeout=180, headers=HEADER)
                if resp.status_code == 200 and resp.text.strip():
                    lines = [ln for ln in resp.text.splitlines() if ln.strip()]
                    break
                print(f"[wayback] CDX attempt {attempt + 1}/4: HTTP {resp.status_code}, "
                      f"{len(resp.text)} bytes — retrying")
            except Exception as e:
                print(f"[wayback] CDX attempt {attempt + 1}/4 failed: {e}")
            time.sleep(5 * (attempt + 1))
        if not lines:
            print("[wayback] CDX query failed after retries")
            return written
        print(f"[wayback] {len(lines)} archived captures matching {url_prefix!r}")

        seen = set()
        for ln in lines:
            if limit is not None and written >= limit:
                break
            parts = ln.split()
            if len(parts) < 2:
                continue
            original, ts = parts[0], parts[1]
            stored = (clean_url(original) if clean_url else original)
            if not stored or stored in seen:
                continue
            seen.add(stored)
            # Archived pages are immutable: skip anything already indexed so re-runs are cheap.
            if self._known_last_crawled(stored) is not None:
                continue
            snap = f"https://web.archive.org/web/{ts}id_/{original}"
            # archive.org aggressively throttles bursts to the raw-snapshot endpoint (connection
            # refused). Retry each fetch with exponential backoff; under CrawlManager that wait is
            # spent crawling another source rather than idling.
            page = None
            for attempt in range(4):
                page, new_url, status_code = get_page_content(snap, self)
                if page is not None:
                    break
                time.sleep(min(90, 10 * 2 ** attempt))
            if page is not None:
                written += self._parse_original(page, stored, default_lang, markup, None, category)
        return written


def finalize_canonical(db, tokenizer, changed_urls, label: str = "") -> None:
    """Shared write-heavy finalization of a crawl delta: fill missing `parsed`, merge multi-source
    provenance, deduplicate *changed_urls* in place (URL + content election), and compress. No-ops
    the dedup when nothing changed. Used by Crawler.commit_dataset (per-source) and by CrawlManager's
    run-level dedup-once pass (the whole run's touched URLs at once). The CALLER holds any write lock;
    this does not lock itself."""
    from . import database, deduplicator, batching
    urls = list(changed_urls)
    if not urls:
        print(f"[{label}] nothing new/changed — DB unchanged")
        database.compress_db(db)
        return
    n_unparsed = db.execute("SELECT COUNT(*) FROM pages WHERE parsed IS NULL").fetchone()[0]
    if n_unparsed:
        print(f"[{label}] parsing {n_unparsed} rows still missing `parsed`…")
        batching.batch_parse_web_page(db, tokenizer, only_none=True)
    merged = database.merge_provenance_by_content_hash(db)
    print(f"[{label}] provenance merged across {merged} multi-source content groups")
    deduplicator.Deduplicator(threshold=1.0).run_incremental(db, changed_urls=urls)
    database.compress_db(db)


class CrawlManager:
    """Run several source crawls concurrently to hide per-site latency (network waits, timeouts,
    rate-limit backoffs), while serializing all DB writes so only ONE thread writes the shared
    canonical at a time.

    Each source gets its own Crawler — its own connection and its own per-domain rate limiting — so
    a slow or throttled site (e.g. archive.org backing off) doesn't stall the others: the time
    waiting on one is spent fetching another. A shared `write_lock` makes every crawler's
    `_flush_writes` / `commit_dataset` mutually exclusive, preserving the single-writer invariant
    the SQLite canonical requires. Fetching (the slow, I/O-bound part) is what runs in parallel.

    Usage:
        mgr = CrawlManager("chantal-canonical.db")
        mgr.add("electropedia", lambda cr: cr.get_wayback_pages("electropedia.org/iev/iev.nsf/display*", clean_url=fn))
        mgr.add("mdpi",         lambda cr: cr.get_openalex_works([...], mailto="..."))
        mgr.run()
    """

    def __init__(self, canonical_db: str, delay: float = 1.0, max_workers: int | None = None,
                 dedup_once: bool = True):
        self.canonical_db = canonical_db
        self.delay = delay
        self.max_workers = max_workers
        self.dedup_once = dedup_once     # True: fetch all, dedup the whole run ONCE at the end
        self.write_lock = threading.Lock()
        self._jobs: list = []
        self._all_urls: set = set()
        self._urls_lock = threading.Lock()

    def add(self, dataset_name: str, crawl_fn, **crawler_kwargs) -> "CrawlManager":
        """Register a source. *crawl_fn(cr)* does the fetching (calls cr.get_website_from_* /
        get_openalex_works / get_wayback_pages …); the crawl lifecycle is handled here."""
        self._jobs.append((dataset_name, crawl_fn, crawler_kwargs))
        return self

    def _run_one(self, dataset_name: str, crawl_fn, crawler_kwargs) -> None:
        cr = Crawler(delay=crawler_kwargs.pop("delay", self.delay), **crawler_kwargs)
        cr.write_lock = self.write_lock  # share the single-writer lock
        finished = False
        try:
            cr.begin_dataset(self.canonical_db, dataset_name)
            crawl_fn(cr)
            finished = True
        except Exception as e:
            print(f"[manager] {dataset_name!r} FETCH FAILED: {type(e).__name__}: {e}")
        finally:
            # Always finalize what was fetched (even on error): dedup-once mode just flushes+closes
            # and hands the touched URLs to the run-level dedup; else per-source commit_dataset.
            if cr.db is not None:
                try:
                    if self.dedup_once:
                        urls = cr.flush_dataset(dataset_name)
                        with self._urls_lock:
                            self._all_urls |= urls
                    else:
                        cr.commit_dataset(dataset_name)
                except Exception as e:
                    print(f"[manager] {dataset_name!r} finalize error: {e}")
        return finished

    def run(self) -> None:
        """Run all registered jobs concurrently (≤ max_workers fetching at once); then, in dedup-once
        mode, deduplicate the whole run's delta a SINGLE time over all sources' touched URLs."""
        if not self._jobs:
            return
        limit = self.max_workers or len(self._jobs)
        sem = threading.Semaphore(limit)

        def worker(name, fn, kw):
            with sem:
                self._run_one(name, fn, kw)

        threads = [threading.Thread(target=worker, args=(n, fn, kw), name=f"crawl-{n}")
                   for n, fn, kw in self._jobs]
        print(f"[manager] {len(threads)} sources → {self.canonical_db} "
              f"(≤{limit} fetching at once, single writer, dedup_once={self.dedup_once})")
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if self.dedup_once and self._all_urls:
            from . import database
            print(f"[manager] run-level finalize: dedup over {len(self._all_urls)} URLs across all sources")
            db = database.create_db(self.canonical_db, url_primary_key=False)
            database.ensure_incremental_autovacuum(db)
            finalize_canonical(db, Crawler().tokenizer, self._all_urls, label="run")
            database.close_db(db)
        print("[manager] all crawls finished")