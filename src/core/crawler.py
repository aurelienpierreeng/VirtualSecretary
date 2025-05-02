"""Module containing utilities to crawl websites for HTML, XML and PDF pages for their text content. PDF can be read from their text content if any, or through optical characters recognition for scans. Websites can be crawled from a `sitemap.xml` file or by following internal links recursively from and index page. Each page is aggregated on a list of [core.crawler.web_page][] objects, meant to be used as input to train natural language AI models and to index and rank for search engines.

© 2023-2024 - Aurélien Pierre
"""

import os
import time
import random
import json
import concurrent.futures

from urllib.parse import urljoin

import requests
import regex as re
import numpy as np


from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver import ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait

from bs4 import BeautifulSoup

from . import patterns, utils
from .pdf import get_pdf_content
from .types import web_page, get_web_pages_ram
from .network import check_response, try_url, get_url

def get_content_type(url: str, delay: int) -> tuple[str, bool, str, dict]:
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
    """
    try:
        response, header, new_url = try_url(url, timeout=10, delay=delay)
        return response.headers['content-type'], (response.status_code != 404), new_url, header
    except Exception as e:
        print(url, e)
        return "", False, url, {}


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
def get_content(url, custom_header, backend, driver, wait) -> tuple[str, str, int]:
    content, url, status, encoding, apparent_encoding = get_url(url, timeout=30, custom_header=custom_header, backend=backend, driver=driver, wait=wait)
    if(status != 200):
        raise(Exception("No page found"))

    if isinstance(content, bytes):
        # Of course some institutionnal websites don't use UTF-8, so let's guess
        try:
            # Try UTF-8. Note that the Selenium backend doesn't return encoding.
            content = content.decode()
        except (UnicodeDecodeError, AttributeError):
            try:
                content = content.decode(apparent_encoding)
            except (UnicodeDecodeError, AttributeError):
                content = content.decode(encoding)
    elif not isinstance(content, str):
        raise(Exception("Content is neither bytes nor string"))

    return content, url, status


def try_content(url, content, custom_header, backend, driver, wait):
    if content is None and url is not None:
        content, url, status = get_content(url, custom_header, backend, driver, wait)

    # Minified HTML doesn't have line breaks after block-level tags.
    # This is going to make sentence tokenization a nightmare because BeautifulSoup doesn't add them in get_text()
    # Re-introduce here 2 carriage-returns after those tags to create paragraphs.
    unminified = re.sub(r"(\<\/(?:div|section|main|section|aside|header|footer|nav|time|article|h[1-6]|p|ol|ul|li|details|pre|dl|dt|dd|table|tr|th|td|blockquote|style|img|audio|video|iframe|embed|figure|canvas|fieldset|hr|caption|figcaption|address|form|noscript|select)\>)",
                        r"\1\n\n\n\n", content, timeout=60)
    # Same with inline-level tags, but only insert space, except for superscript and subscript
    unminified = re.sub(r"(\<\/(?:a|span|time|sup|abbr|b|i|em|strong|code|dfn|big|kbd|label|textarea|input|option|var|q|tt)\>)",
                        r"\1 ", unminified, timeout=60)

    handler = BeautifulSoup(unminified, "html5lib")

    # In case of recursive crawling, we need to milk the links out before we remove <nav> at the next step
    handler.links = list({url["href"] for url in handler.find_all('a', href=True) if url["href"]})

    # Same with h1 because we will remove <header> and that's where it might be
    # Doe h2 as well since we are at it.
    handler.h1 = {tag.get_text().strip(" \n\t\r#↑🔗") for tag in handler.find_all("h1")}
    handler.h2 = {tag.get_text().strip(" \n\t\r#↑🔗") for tag in handler.find_all("h2")}

    # Same with date: sometimes put in <header>
    handler.date = get_date(handler)

    # Remove any kind of machine code and symbols from the HTML doctree because we want natural language only
    # That will also make subsequent parsing slightly faster.
    # Remove blockquotes too because they can duplicate internal content of forum pages.
    # Basically, the goal is to get only the content body of the article/page.
    for element in handler.select('pre, math, style, script, svg, img, picture, audio, video, iframe, embed, aside, nav, input, header, button, form, fieldset, footer, summary, dialog, textarea, select, option, sup'):
        element.decompose()

    # Remove inline style and useless attributes too
    for attribute in ["data", "style", "media"]:
        for tag in handler.find_all(attrs={attribute: True}):
            del tag[attribute]

    return handler, url


def get_page_content(url: str, content: str = None, custom_header={}, backend="requests", driver=None, wait=None) -> [BeautifulSoup | None, str]:
    """Request an (x)HTML page through the network with HTTP GET and feed its response to a BeautifulSoup handler. This needs a functionnal network connection.

    The DOM is pre-filtered as follow to keep only natural language and avoid duplicate strings:

    - media tags are removed (`<iframe>`, `<embed>`, `<img>`, `<svg>`, `<audio>`, `<video>`, etc.),
    - code and machine language tags are removed (`<script>`, `<style>`, `<code>`, `<pre>`, `<math>`),
    - menus and sidebars are removed (`<nav>`, `<aside>`),
    - forms, fields and buttons are removed(`<select>`, `<input>`, `<button>`, `<textarea>`, etc.)

    The HTML is un-minified to help end-of-sentences detections in cases where sentences don't end with punctuation (e.g. in titles).

    Arguments:
        url: a valid URL that can be fetched with an HTTP GET request.
        content: a string buffer used as HTML source. If this argument is passed, we don't fetch `url` from network and directly use this input.
        backend: `"requests"` uses the Python package requests and is the fastest option for pure HTML websites but doesn't support Javascript.
        `"selenium"` uses a Chrome browser driver, it is slower but handles AJAX-driven websites that require Javascript to work.

    Returns:
        a tuple with:
            1. [bs4.BeautifulSoup][] object initialized with the page DOM for further text mining. `None` if the HTML response was empty or the URL could not be reached. The list of URLs found in page before removing meaningless markup is stored as a list of strings in the `object.links` member. `object.h1` and `object.h2` contain a set of headers 1 and 2 found in the page before removing any markup. `object.date` contains the best-guess for the date.
            2. the final URL of the retrieved page, which might be different from the input URL if HTTP redirections happened,
    """

    try:
        return try_content(url, content, custom_header, backend, driver, wait)
    except Exception as e:
        print(e)
        return None, url


def get_page_markup(page: BeautifulSoup, markup: str|tuple|list[str]|list[tuple]|None) -> str:
    """Extract the text content of an HTML page DOM by targeting only the specific tags.

    Arguments:
        page: a [bs4.BeautifulSoup][] handler with pre-filtered DOM,
        markup: any kind of tags supported by [bs4.BeautifulSoup.find_all][]:

            - (str): the single tag to select. For example, `"body"` will select `<body>...</body>`.
            - (tuple): the tag and properties to select. For example, `("div", { "class": "right" })` will select `<div class="right">...</div>`.
            - all combinations of the above can be chained in lists.
            - None: don't parse the page internal content. Links,
            h1 and h2 headers will still be parsed.

    Returns:
        The text content of all instances of all tags in markup as a single string, if any, else an empty string.

    Examples:
        >>> get_page_markup(page, "article")

        >>> get_page_markup(page, ["h1", "h2", "h3", "article"])

        >>> get_page_markup(page, [("div", {"id": "content"}), "details", ("div", {"class": "comment-reply"})])
    """
    output = ""

    if markup is None:
        return output

    if not isinstance(markup, list):
        markup = [markup]

    for item in markup:
        if isinstance(item, tuple):
            # Unroll additional params (classes, ids, etc.)
            elements = page.find_all(item[0], item[1])
        else:
            elements = page.find_all(item)

        print(f"found {len(elements)} {item}")

        if elements:
            # Get the inner text
            results = [tag.get_text() for tag in elements]
            output += "\n\n".join(results)

    return output


def get_excerpt(html: BeautifulSoup) -> str | None:
    """Find HTML tags possibly containing the shortened version of the page content.

    Looks for HTML tags:

    - `<meta name="description" content="...">`
    - `<meta property="og:description" content="...">`

    Arguments:
        page: a [bs4.BeautifulSoup][] handler with pre-filtered DOM,

    Returns:
        The content of the meta tag if any.
    """

    excerpt_options = [ ("meta", {"property": "og:description"}),
                        ("meta", {"name": "description"}) ]

    excerpt = None
    i = 0

    while not excerpt and i < len(excerpt_options):
        excerpt = html.find(excerpt_options[i][0], excerpt_options[i][1])
        i += 1

    return excerpt["content"] if excerpt and "content" in excerpt else None


def get_date(html: BeautifulSoup):
    """Find HTML tags possibly containing the page date.

    Looks for HTML tags:

    - `<meta name="date" content="...">`
    - `<meta name="publishDate" content="...">`
    - `<meta property="article:published_time" content="...">`
    - `<meta property="article:modified_time" content="...">`
    - `<meta name="dc.date" content="...">`
    - `<time datetime="...">`
    - `<relative-time datetime="...">`
    - `<div class="dateline">...</div>`
    - `<script type="application/ld+json">{"dateModified":"...", }</script>` (Wikipedia)

    Arguments:
        page: a [bs4.BeautifulSoup][] handler with pre-filtered DOM,

    Returns:
        The content of the meta tag if any.
    """
    def method_0(html: BeautifulSoup):
        test = html.find("meta", {"name": "date", "content": True})
        return test["content"] if test else None

    def method_1(html: BeautifulSoup):
        test = html.find("meta", {"name": "publishDate", "content": True})
        return test["content"] if test else None

    def method_2(html: BeautifulSoup):
        test = html.find("meta", {"property": "article:modified_time", "content": True})
        return test["content"] if test else None

    def method_3(html: BeautifulSoup):
        test = html.find("meta", {"property": "article:published_time", "content": True})
        return test["content"] if test else None

    def method_4(html: BeautifulSoup):
        test = html.find("meta", {"name": "dc.date", "content": True})
        return test["content"] if test else None

    def method_5(html: BeautifulSoup):
        test = html.find("time", {"datetime": True})
        return test["datetime"] if test else None

    def method_6(html: BeautifulSoup):
        test = html.find("relative-time", {"datetime": True})
        return test["datetime"] if test else None

    def method_7(html):
        test = html.find("div", {"class": "dateline"})
        return test.get_text() if test else None

    def method_9(html):
        # Rich snippets
        test = html.find("span", {"class": "updated rich-snippet-hidden"})
        return test.get_text() if test else None

    def method_8(html):
        """
        Wikipedia example JSON:
        ```
        {
            "@context":"https://schema.org",
            "@type":"Article",
            "name":"Purple fringing","url":"https://en.wikipedia.org/wiki/Purple_fringing",
            "sameAs":"http://www.wikidata.org/entity/Q1154488",
            "mainEntity":"http://www.wikidata.org/entity/Q1154488",
            "author":{
                "@type":"Organization",
                "name":"Contributors to Wikimedia projects"
            },
            "publisher":{
                "@type":"Organization",
                "name":"Wikimedia Foundation, Inc.",
                "logo":{
                    "@type":"ImageObject",
                    "url":"https://www.wikimedia.org/static/images/wmf-hor-googpub.png"
                }
            },
            "datePublished":"2005-10-07T04:26:05Z",
            "dateModified":"2023-11-30T04:38:53Z",
            "image":"https://upload.wikimedia.org/wikipedia/commons/c/c1/Purple_fringing.jpg",
            "headline":"type of chromatic aberration in photography"
         }
         ```
         """
        test = html.find("script", {"type": "application/ld+json"})
        if test:
            inner = json.loads(test.contents[0])
            if "dateModified" in inner:
                return inner["dateModified"]
        return None

    date = None
    bag_of_methods = (method_0, method_1, method_2, method_3, method_4, method_5, method_6, method_7, method_8, method_9)

    i = 0
    while not date and i < len(bag_of_methods):
        date = bag_of_methods[i](html)
        i += 1

    return date


def get_lang(html: BeautifulSoup) -> str:
    """Attempt to find the page language"""

    def method_0(html):
        return html.html["lang"] if "lang" in html.html and html.html["lang"] else None

    def method_1(html):
        test = html.find("meta", {"property": "og:locale", "content": True})
        return test["content"] if test else None

    lang = None
    bag_of_methods = (method_0, method_1)

    i = 0
    while not lang and i < len(bag_of_methods):
        lang = bag_of_methods[i](html)
        i += 1

    return lang


def parse_page(page: BeautifulSoup, url: str,
               lang: str, markup: str | list[str],
               date: str = None,
               category: str = None) -> list[web_page]:
    """Get the requested markup from the requested page URL.

    This chains in a single call:

    - [core.crawler.get_page_markup][]
    - [core.crawler.get_date][]
    - [core.crawler.get_excerpt][]

    Arguments:
        page: a [bs4.BeautifulSoup][] handler with pre-filtered DOM,
        url: the valid URL accessible by HTTP GET request of the page
        lang: the provided or guessed language of the page,
        markup: the markup to search for. See [core.crawler.get_page_markup][] for details.
        date: if the page was retrieved from a sitemap, usually the date is available in ISO format (`yyyy-mm-ddTHH:MM:SS`) and can be passed directly here. Otherwise, several attempts will be made to extract it from the page content (see [core.crawler.get_date][]).
        category: arbitrary category or label defined by user

    Returns:
        The content of the page, including metadata, in a [core.crawler.web_page][] singleton.
    """
    # Get excerpt in metadata - hard : several ways of declaring it.
    excerpt = get_excerpt(page)

    # Get date - hard if no sitemap with timestamps.
    if not date:
        date = page.date

    # Get content - easy : user-request
    if ".wikipedia.org" in url and markup == "body":
        # Wikipedia is massive enough to appear in at least one external link.
        # For external links, the default behaviour is to get the whole <body>.
        # For Wikipedia, make an special case here by restricting markup to meaningfull stuff.
        # TODO: is this portable to all pages generated by MediaWiki ?
        markup = ("div", {"id": "mw-content-text"})

    content = get_page_markup(page, markup=markup)
    lang = get_lang(page)

    # Get title - easy : it's standard
    title = page.find("title")
    if title:
        title = title.get_text()
    elif len(page.h1) > 0:
        title = list(page.h1)[0]
    elif len(content) > 50:
        title = content[0:50]

    if content and title:
        result = web_page(title=title,
                          url=url,
                          date=date,
                          content=content,
                          excerpt=excerpt,
                          h1=page.h1,
                          h2=page.h2,
                          lang=lang,
                          category=category)
        print(result)
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


class Crawler:
    no_follow: list[str] = [
        "api.whatsapp.com/share",
        "api.whatsapp.com/send",
        "pinterest.fr/pin/create",
        "pinterest.com/pin/create",
        "facebook.com/sharer",
        "twitter.com/intent/tweet",
        "twitter.com/share",
        "reddit.com/submit",
        "t.me/share", # Telegram share
        "linkedin.com/share",
        "vk.com/share.php",
        "bufferapp.com/add",
        "getpocket.com/edit",
        "tumblr.com/share",
        "translate.google.com/translate", # Machine-translated pages
        "flickr.com",
        "instagram.com",
        "mailto:",
        "/profile/",
        "/login/",
        "/signup/",
        "/login?",
        "/signup?"
        "/user/",
        "/member/",
        "/register?",
        ".css",
        ".js",
        ".json",
        ]
    """List of URLs sub-strings that will disable crawling if they are found in URLs. Mostly social networks sharing links."""

    executor = None
    futures = []

    def __init__(self, delay: float = 1., no_follow: list[str] = []):
        """Crawl a website from its sitemap or by following internal links recusively from an index page.
        This creates a pool of threads to parallelize network I/O. The pool needs to be freed after use.
        This class needs therefore to be used within a `with` statement that will take care of resources
        allocations and releases in background.

        Parameters:
            delay: time in seconds to wait before 2 HTTP requests. Keep in mind that crawling is multi-threaded,
            so as many concurrent requests can happen at the same time against a server as you have threads.
            The right delay will prevent the crawler from being throttled by anti-DoS rules while making it as fast as possible.
            Set to `0.0` if you are crawling your own servers and they have no DoS protection.
            no_follow: list of URL parts to completely ignore, that is not index them but not even crawl them for internal links.
            no_follow: list of URLs parts that will discard pages from crawling

        Example:
            ```
            with crawler.Crawler() as cr:
                output = cr.get_website_from_sitemap("https://domain.com")

                # Can be called more than once.
                # The list of already-crawled pages will be shared between calls
                # so pages are not crawled more than once.
                output += cr.get_website_from_crawling("https://forum.domain.com")
            ```

        """
        self.crawled_URL: list[str] = []
        """List of URLs already visited"""

        self.no_follow += no_follow
        self.delay = delay

        # Start an headless Chromium
        options = ChromeOptions()
        options.headless = True
        options.add_argument("--headless=new")

        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.driver.set_page_load_timeout(30)
        self.driver.implicitly_wait(30)
        self.wait = WebDriverWait(self.driver, 15)

        self.errors = []
        """URLs that couldn't be accessed due to blocking or throttling"""

        self.notfound = []
        """URLs returning error 404 - not found"""


    def __enter__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count())
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown()
        print("PROCESSED URLS:", len(set(self.crawled_URL)))
        print("404 ERRORS:", len(set(self.notfound)))
        print("OTHER ERRORS:", len(set(self.errors)))
        print(set(self.errors))
        print(set(self.notfound))


    def discard_link(self, url):
        """Returns True if the url is found in the `self.no_follow` list"""
        for elem in self.no_follow:
            if elem in url:
                return True

        return False


    def get_immediate_links(self, links: list[str], domain, default_lang, langs, category, contains_str, internal_links: str = "any", mine_pdf = False) -> list[web_page]:
        """Follow internal and external links contained in a webpage only to one recursivity level,
        including PDF files and HTML pages. This is useful to index references docs linked from a page.

        Args:
            internal_links: defines what to do with links found inside the HTML page body/content:
                - `any`: follow and include all links found in page, no matter what domain they point to,
                - `internal`: follow and include links found in page only if they point to the same domain as the current page,
                - `external`: follow and include links found in page only if they point to a different domain than the current page,
                - `ignore`: don't follow internal links

        Returns:
            This method returns nothing but adds asynchronous crawling jobs on the [][crawler.Crawler.futures] stack.
        """
        output = []
        if internal_links == "ignore":
            return output

        for nextURL in links:
            current_address = patterns.URL_PATTERN.search(nextURL, concurrent=True)
            if not current_address or nextURL in self.crawled_URL:
                continue

            current_protocol = current_address.group(1) if current_address.group(1) else "https"
            current_domain = current_address.group(2)
            current_page = current_address.group(3)
            current_params = current_address.group(4) if current_address.group(4) else ""

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
                time.sleep(self.delay)
                if domain not in current_domain:
                    # If the current URL doesn't belong to the same domain as the parent,
                    # we don't pass on the category of the parent page
                    # because we have no idea what the external URL is.
                    # And we multi-thread
                    category = "external"

                # No multithreading if we stay on the same domain to comply with rates thresholds
                output += self.get_website_from_crawling(current_protocol + "://" + current_domain + current_page + current_params, default_lang, "", langs, max_recurse_level=1, category=category, contains_str=contains_str, mine_pdf=mine_pdf, _recursion_level=0, _mainthread=False)

        return output



    def get_website_from_crawling(self,
                                  website: str,
                                  default_lang: str = "en",
                                  child: str = "/",
                                  langs: tuple = ("en", "fr"),
                                  markup: str = "body",
                                  contains_str: str | list[str] = "",
                                  max_recurse_level: int = -1,
                                  category: str = None,
                                  restrict_section: bool = False,
                                  mine_pdf: bool = False,
                                  _recursion_level: int = 0,
                                  _mainthread: bool = True) -> list[web_page]:
        """Recursively crawl all pages of a website from internal links found starting from the `child` page. This applies to all HTML pages hosted on the domain of `website` and to PDF documents either from the current domain or from external domains but referenced on HTML pages of the current domain.

        Arguments:
            website: root of the website, including `https://` or `http://` without trailing slash.
            default_lang: provided or guessed main language of the website content. Not used internally.
            child: page of the website to use as index to start crawling for internal links.
            langs: ISO-something 2-letters code of the languages for which we attempt to fetch the translation if available, looking for the HTML `<link rel="alternate" hreflang="...">` tag.
            contains_str: a string or a list of strings that should be contained in a page URL for the page to be indexed. On a forum, you could for example restrict pages to URLs containing `"discussion"` to get only the threads and avoid user profiles or archive pages.
            markup: see [core.crawler.get_page_markup][]
            max_recursion_level: this method will call itself recursively on each internal link found in the current page, starting from the `website/child` page. The `max_recursion_level` defines how many times it calls itself until it is stopped, if it is stopped. When set to `-1`, it stops when all the internal links have been crawled.
            category: arbitrary category or label set by user.
            restrict_section: set to `True` to limit crawling to the website section defined by `://website/child/*`. This is useful when indexing parts of very large websites when you are only interested in a small subset.
            mine_pdf: set to `True` to aggressively try to crawl PDF linked on external HTML pages. This may increase RAM consumption dramatically.
            _recursion_level: __DON'T USE IT__. Everytime this method calls itself recursively, it increments this variable internally, and recursion stops when the level is equal to the `max_recurse_level`.

        Returns:
            a list of all valid pages found. Invalid pages (wrong markup, empty HTML response, 404 errors) will be silently ignored.

        Examples:
            >>> from core import crawler
            >>> cr = crawler.Crawler()
            >>> pages = cr.get_website_from_crawling("https://aurelienpierre.com", default_lang="fr", markup=("div", { "class": "post-content" }))
        """
        output = []
        index_url = radical_url(website + child)
        #print("trying", index_url)

        if self.discard_link(index_url):
            #print("no follow")
            return output

        # Abort now if the page was already crawled or recursion level reached
        if index_url in self.crawled_URL:
            #print("already crawled")
            return output

        if max_recurse_level > -1 and _recursion_level >= max_recurse_level:
            #print("max recursivity level reached")
            return output

        # Extract the domain name, to prepend it if we find relative URL while parsing
        split_domain = patterns.URL_PATTERN.search(website)
        if split_domain is None:
            print("%s can't be parsed as URL" % website)
            return output

        domain = split_domain.group(2)
        include = check_contains(contains_str, index_url)
        #print("processing", index_url, "include", include)

        # Fetch and parse current (top-most) page
        time.sleep(self.delay)
        content_type, status, new_url, custom_header = get_content_type(index_url, self.delay)
        print("HEADERS:", status, new_url, content_type, custom_header)

        # Recall we passed there, whether or not we actually mined something
        self.crawled_URL.append(index_url)

        if not status:
            self.notfound += list({new_url, index_url})

        if index_url != new_url:
            self.crawled_URL.append(new_url)
            index_url = new_url

        if self.discard_link(index_url) or not status:
            #print("no follow")
            return output

        # FIXME: we nest 7 levels of if here. It's ugly but I don't see how else
        # to cover so many cases.
        if "text" in content_type \
            and "javascript" not in content_type \
            and "css" not in content_type \
            and "json" not in content_type:

            time.sleep(self.delay)
            index, new_url = get_page_content(index_url, backend="requests", custom_header=custom_header, driver=self.driver, wait=self.wait)

            if index is None:
                self.errors += list({new_url, index_url})
                return output

            # Account for HTTP redirections
            if new_url != index_url:
                self.crawled_URL.append(new_url)
                index_url = new_url

            if include or _recursion_level == 0 or ".pdf" in index_url.lower():
                # For the first recursion level, ignore "url contains" rule to allow parsing index pages
                if index:
                    # Valid HTML response
                    output += self._parse_original(index, index_url, default_lang, markup, None, category)
                    output += self._parse_translations(index, domain, index_url, markup, None, langs, category)
                    #print("page object")
                else:
                    # Some websites display PDF in web applets on pages
                    # advertising content-type=text/html but UTF8 codecs
                    # fail to decode because it's actually not HTML but PDF.
                    # If we end up here, it's most likely what we have.
                    output += self._parse_pdf_content(index_url, default_lang, category=category)
                    #print("no page object")

            # Follow internal links whether or not this page was mined, if we didn't reach the final recursion level
            if index and _recursion_level + 1 != max_recurse_level:
                for currentURL in self.get_unique_internal_url(index, domain, index_url):
                    current_address = patterns.URL_PATTERN.search(currentURL)
                    if not current_address or currentURL in self.crawled_URL:
                        continue

                    current_protocol = current_address.group(1) if current_address.group(1) else "https"
                    current_domain = current_address.group(2)
                    current_page = current_address.group(3)
                    current_params = current_address.group(4) if current_address.group(4) else ""

                    #print(current_page, current_page, current_params)
                    # Note: we use multi-threading only for outer domains links, so we
                    # can keep track of the throttling rate on the source domain,
                    # and avoid being rejected by the server.
                    if not restrict_section and domain == current_domain:
                        # Recurse only through local pages, aka :
                        # 1. domains match
                        #print("recursing")
                        output += self.get_website_from_crawling(
                            website, default_lang, child=current_page + current_params, langs=langs, markup=markup, contains_str=contains_str,
                            _recursion_level=_recursion_level + 1, max_recurse_level=max_recurse_level, restrict_section=restrict_section, category=category,
                            _mainthread=False)
                    elif restrict_section and domain == current_domain and child in current_page:
                        # Recurse only through local subsections, aka :
                        # 1. domains match
                        # 2. current page is in a subsection of current child
                        #print("recursing")
                        output += self.get_website_from_crawling(
                            website, default_lang, child=current_page + current_params, langs=langs, markup=markup, contains_str=contains_str,
                            _recursion_level=_recursion_level + 1, max_recurse_level=max_recurse_level, restrict_section=restrict_section, category=category,
                            _mainthread=False)
                    elif include:# and domain == current_domain:
                        # Follow internal links on only one recursivity level.
                        # Aka HTML reference pages (Wikipedia) and attached PDF (docs, manuals, spec sheets)
                        #print("following")
                        output += self.get_website_from_crawling(
                            current_protocol + "://" + current_domain + current_page + current_params, default_lang, "", langs, contains_str="", max_recurse_level=1,
                            restrict_section=restrict_section, category=category, _recursion_level=0, _mainthread=False)
                        """
                        elif include and domain != current_domain:
                            # Follow external links on only one recursivity level.
                            # Aka HTML reference pages (Wikipedia) and attached PDF (docs, manuals, spec sheets)
                            #print("following")
                            self.futures.append(self.executor.submit(self.get_website_from_crawling,
                                current_protocol + "://" + current_domain + current_page + current_params, default_lang, "", langs, contains_str="", max_recurse_level=1,
                                restrict_section=restrict_section, category=category, _recursion_level=0, _mainthread=False))
                        """
                    else:
                        #print("discarding")
                        pass

            elif index and mine_pdf:
                # Scenario : we are on the terminating page. That's :
                # case 1 : we are at the last stage of recursion.
                # case 2 : we are on an external link, followed from sitemap page.
                # Terminating page can be an index page for PDF files containing
                # the actual content (ex: ArXiv). Do an exception : crawl one step further for PDFs only.
                output += self._parse_internal_pdfs(index, domain, index_url, default_lang, category)
                pass

        elif "pdf" in content_type:
            #print("got pdf")
            time.sleep(self.delay)
            output += self._parse_pdf_content(index_url, default_lang, category=category)
            # No link to follow from PDF docmuents
        else:
            # Got an image, video, compressed file, binary, etc.
            #print("nothing done")
            pass

         # Process internal links found in pages
        if _mainthread:
            # Get pages content from the pool
            output += self.wait_for_crawling()

            print("OUTPUT", type(output))
            print("FINAL NUMBER of POSTS:", len(output))

        return output

    def wait_for_crawling(self) -> list[web_page]:
        """Wait for all crawling parallel threads to return their page object

        Return:
            the list of webpages crawled
        """
        output = []

        # Can't use built-in methods because we don't know the size of self.futures ahead
        # since we append dynamically.
        while len(self.futures) > 0 and self.futures[0]:
            output += self.futures[0].result()
            del(self.futures[0])

        return output


    def get_website_from_sitemap(self,
                                 website: str,
                                 default_lang: str,
                                 sitemap: str = "/sitemap.xml",
                                 langs: tuple[str] = ("en", "fr"),
                                 markup: str | tuple[str] = "body",
                                 category: str = None,
                                 contains_str: str | list[str] = "",
                                 internal_links: str = "any",
                                 mine_pdf: bool = False,
                                 _recursion_level: int = 0) -> list[web_page]:
        """Recursively crawl all pages of a website from links found in a sitemap. This applies to all HTML pages hosted on the domain of `website` and to PDF documents either from the current domain or from external domains but referenced on HTML pages of the current domain. Sitemaps of sitemaps are followed recursively.

        Arguments:
            website: root of the website, including `https://` or `http://` without trailing slash.
            default_lang: provided or guessed main language of the website content. Not used internally.
            sitemap: relative path of the XML sitemap.
            langs: ISO-something 2-letters code of the languages for which we attempt to fetch the translation if available, looking for the HTML `<link rel="alternate" hreflang="...">` tag.
            markup: see [core.crawler.get_page_markup][]
            category: arbitrary category or label
            contains_str: limit recursive crawling from sitemap-defined pages to pages containing this string or list of strings. Will get passed as-is to [get_website_from_crawling][].
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
        output = []

        time.sleep(self.delay)
        index_url = website + sitemap

        self.crawled_URL.append(index_url)
        content_type, status, new_url, custom_header = get_content_type(index_url, self.delay)
        print("HEADERS:", status, new_url, content_type, custom_header)

        if not status:
            self.notfound += list({index_url, new_url})

        if new_url != index_url:
            self.crawled_URL.append(new_url)
            index_url = new_url

        if not status:
            return output

        index_page, new_url = get_page_content(index_url, custom_header=custom_header, backend="requests", driver=self.driver, wait=self.wait)
        if index_page is None:
            self.errors += list({index_url, new_url})
            return output

        split_domain = patterns.URL_PATTERN.search(website)
        domain = split_domain.group(2)

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
            output += self.get_website_from_sitemap(website, default_lang, sitemap=_sitemap, langs=langs, markup=markup, category=category, internal_links=internal_links, _recursion_level=_recursion_level+1)

        # Process pages
        for link in index_page.find_all('url'):
            output += self._sitemap_process(domain, website, sitemap, link, default_lang, langs, markup, category, internal_links, contains_str, mine_pdf, _recursion_level)

        # Process internal links found in pages
        if _recursion_level == 0:
            output += self.wait_for_crawling()

        return output


    def get_unique_internal_url(self, page: BeautifulSoup, domain: str, currentURL:str) -> list[str]:
        """Grab the internal links found in page, except PDF, and return only the ones we don't already know"""
        # Get a set of unique absolute URLs
        links = {relative_to_absolute(url, domain, currentURL) for url in page.links}

        # Get URLs that are neither already crawled or already on the list nor PDF
        return list({url for url in links.difference(set(self.crawled_URL))
                if not self.discard_link(url) and ".pdf" not in url.lower()})


    def _sitemap_process(self, domain, website, sitemap, link, default_lang, langs, markup, category, internal_links, contains_str, mine_pdf, _recursion_level) -> list[web_page]:
        output = []
        url = link.find("loc")
        date = link.find("lastmod")

        if not url:
            print("No URL found in ", link)
            # Nothing to process, ignore this item
            return output

        date = date.get_text() if date else None

        currentURL = relative_to_absolute(url.get_text(), domain, website + sitemap)
        print(currentURL, date)

        if self.discard_link(currentURL):
            return output

        self.crawled_URL.append(currentURL)
        content_type, status, new_url, custom_header = get_content_type(currentURL, self.delay)

        if not status:
            self.notfound += list({currentURL, new_url})

        # Account for HTTP redirections
        if new_url != currentURL:
            self.crawled_URL.append(new_url)
            currentURL = new_url

        if not status:
            return output

        time.sleep(self.delay)
        page, new_url = get_page_content(currentURL, backend="requests", custom_header=custom_header, driver=self.driver, wait=self.wait)

        if page:
            # We got a proper web page, parse it
            output += self._parse_original(page, currentURL, default_lang, markup, date, category)
            output += self._parse_translations(page, domain, currentURL, markup, date, langs, category)
            output += self._parse_internal_pdfs(page, domain, currentURL, default_lang, category)

            # Follow internal and external links found in body
            output += self.get_immediate_links(self.get_unique_internal_url(page, domain, currentURL), domain, default_lang, langs, category, contains_str, internal_links=internal_links, mine_pdf=mine_pdf)

        else:
            self.errors += list({currentURL, new_url})

        return output


    def _parse_pdf_content(self, link, default_lang, category="", custom_header={}):
        time.sleep(self.delay)
        return get_pdf_content(link, default_lang, category=category, custom_header=custom_header)


    def _parse_original(self, page, url, default_lang, markup, date, category):
        return parse_page(page, url, default_lang, markup=markup, date=date, category=category) if page else []


    def _parse_translations(self, page, domain, current_url, markup, date, langs, category):
        """Follow `<link rel="alternate" hreflang="lang" href="url">` tags declaring links to alternative language variants for the current HTML page and crawl the target pages. This works only for pages properly defining alternatives in HTML header."""
        output = []

        if not page:
            return output

        for lang in langs:
            link_tag = page.find('link', {'rel': 'alternate', 'hreflang': lang})

            if link_tag and "href" in link_tag and link_tag["href"]:
                translatedURL = relative_to_absolute(link_tag["href"], domain, current_url)

                self.crawled_URL.append(translatedURL)
                content_type, status, new_url, custom_header = get_content_type(translatedURL, self.delay)

                if not status:
                    self.notfound += list({translatedURL, new_url})

                if translatedURL != new_url:
                    self.crawled_URL.append(new_url)
                    translatedURL = new_url

                if not status:
                    return output

                if "text" in content_type:
                    time.sleep(self.delay)
                    translated_page, new_url = get_page_content(translatedURL, backend="requests", custom_header=custom_header, driver=self.driver, wait=self.wait)

                    # Account for HTTP redirections
                    if new_url != translatedURL:
                        self.crawled_URL.append(new_url)
                        translatedURL = new_url

                    if translated_page is None:
                        self.errors += list({translatedURL, new_url})
                        return output
                    else:
                        output += self._parse_original(translated_page, translatedURL, lang, markup, date, category)


        return output


    def _parse_internal_pdfs(self, page, domain, current_url, default_lang, category):
        output = []
        pdfs = [relative_to_absolute(url, domain, current_url) for url in page.links]
        pdfs = [url for url in set(pdfs).difference(set(self.crawled_URL))
                if ".pdf" in url.lower() and not self.discard_link(url)]

        for currentURL in pdfs:
            time.sleep(self.delay)
            content_type, status, new_url, custom_header = get_content_type(currentURL, self.delay)
            self.crawled_URL.append(currentURL)

            if new_url != currentURL:
                currentURL = new_url
                self.crawled_URL.append(new_url)

            if not status:
                self.notfound += list({currentURL, new_url})
                return output

            if "pdf" in content_type:
                time.sleep(self.delay)
                output += self._parse_pdf_content(currentURL, default_lang, category=category, custom_header=custom_header)

        return output
