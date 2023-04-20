import re
import time

import requests
from bs4 import BeautifulSoup
from core.utils import typography_undo

from typing import TypedDict

web_page = TypedDict("web_page", {"title": str,     # Title of the page
                                  "url": str,       # Where to find the page on the network
                                  "date": str,      # Date of last modification of the page, to assess relevance of the content.
                                  "content": str,   # The actual content of the page. Needs fine-tuning depending of HTML templates.
                                  "excerpt": str,   # Shortened version of the content for search results previews.
                                  "h1": set[str],   # Title of the post if any. There should be only one h1 per page, but some templates wrongly use h1 for section titles.
                                  "h2": set[str],   # Section titles if any
                                  "lang": str       # 2-letters code of the page language
                                  })
"""Dictionnary representing a web page and its metadata"""

def relative_to_absolute(URL: str, domain: str) -> str:
    """Convert a relative path to absolute by prepending the domain"""
    if URL.startswith("/"):
        # relative URL: prepend domain
        return domain + URL
    else:
        return URL


def radical_url(URL: str) -> str:
    """Trim an URL to the page (radical) part, removing anchors if any (internal links)"""
    anchor = re.match(r"(.+?)#(.+)", URL)
    if anchor:
        URL = anchor.groups()[0]
    return URL


def get_page_content(url) -> BeautifulSoup:
    """Request a page through the network and feed its response to a BeautifulSoup handler"""
    # Prevent being thresholded on some servers
    time.sleep(1)

    try:
        page = requests.get(url, timeout=30)
        print(f"{url}: {page.status_code}")
        handler = BeautifulSoup(page.content, 'html.parser')

        # Remove any kind of machine code and symbols from the HTML doctree because we want natural language only
        # That will also make subsequent parsing slightly faster
        [element.decompose() for element in handler.select('code, pre, math, style, script, svg, img, audio, video, iframe, embed')]

        # Remove inline style and useless attributes too
        for attribute in ["data", "style", "media"]:
            for tag in handler.find_all(attrs={attribute: True}):
                del tag[attribute]

        return handler

    except Exception as e:
        print(e)
        return BeautifulSoup("<html></html>", 'html.parser')


def get_page_markup(page: BeautifulSoup, markup: str|list) -> str:
    """Extract the text content of the specified `markup`"""
    output = ""

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
            # Each HTML tag is semantically equivalent to a "sentence".
            # When extracting pure text out of HTML markup, we instruct the parser
            # to separate tags content with dot/space to hint sentences splitters.
            results = [tag.get_text(separator=". ") for tag in elements]
            output += "\n\n".join(results)

    return output


def get_excerpt(html: BeautifulSoup):
    """Find HTML tags possible containing the shortened version of the page content."""

    excerpt_options = [ ("meta", {"property": "og:description"}),
                        ("meta", {"name": "description"}) ]

    excerpt = None
    i = 0

    while not excerpt and i < len(excerpt_options):
        excerpt = html.find(excerpt_options[i][0], excerpt_options[i][1])
        i += 1

    return excerpt["content"] if excerpt else None


def get_date(html: BeautifulSoup):
    """Find HTML tags possibly containing the page date."""

    def method_1(html: BeautifulSoup):
        test = html.find("meta", {"property": "article:modified_time", "content": True})
        return test["content"] if test else None

    def method_2(html: BeautifulSoup):
        test = html.find("time", {"datetime": True})
        return test["datetime"] if test else None

    def method_3(html):
        test = html.find("div", {"class": "dateline"})
        return test.get_text() if test else None

    date = None
    bag_of_methods = (method_1, method_2, method_3)

    i = 0
    while not date and i < len(bag_of_methods):
        date = bag_of_methods[i](html)
        i += 1

    return date


def parse_page(page: BeautifulSoup, url: str, lang: str, markup, date=None) -> list[tuple[str, str, str]]:
    """Get the requested markup from the requested page URL.

    Returns:
        A tuple containing the page parsed content, the page lang and the page URL, ready for machine-learning.
    """
    # Get title - easy : it's standard
    title = page.find("title")
    if title:
        title = title.get_text()

    # Get excerpt in metadata - hard : several ways of declaring it.
    excerpt = get_excerpt(page)
    if excerpt:
        excerpt = typography_undo(excerpt)

    # Get date - hard if no sitemap with timestamps.
    if not date:
        date = get_date(page)

    h1 = {typography_undo(tag.get_text()) for tag in page.find_all("h1")}
    h2 = {typography_undo(tag.get_text()) for tag in page.find_all("h2")}

    # Get content - easy : user-request
    content = typography_undo(get_page_markup(page, markup=markup))

    if content and title:
        result = web_page(title=title,
                          url=url,
                          date=date,
                          content=content,
                          excerpt=excerpt,
                          h1=h1,
                          h2=h2,
                          lang=lang)
        print(result)
        return [result]
    else:
        return []

class Crawler:
    def __init__(self):
        self.crawled_URL = []

    def get_website_from_crawling(self,
                                  website: str,
                                  default_lang,
                                  child: str = "/",
                                  langs: tuple = ("en", "fr"),
                                  markup: str = "body",
                                  contains_str: str = "",
                                  recurse: bool = True) -> list[tuple[str, str, str]]:
        """Crawl all found pages of a website from the index. Intended for word2vec training.

        Arguments:
          website (str): root of the website, including `https://` or `http://` without trailing slash.
          sitemap (str): relative path of the sitemap
          langs (tuple[str]): ISO-something 2-letters code for alternate language
          default_lang (str): the ISO-something 2-letters code for the base language of the website
          contains_str (str): a string that should be contained in a page URL for the page to be indexed.
          recurse (bool): crawl recursively all the links found from the given `website/child` page. If `False`, we fetch only the top-level page and return.

        Returns:
          list of tuples: `(page content, page language, page URL)`
        """
        domain = re.search(r"(https?://[a-z0-9]+?\.[a-z0-9]{2,})", website).group(0)
        index_url = website + child
        index = get_page_content(index_url)
        output = []

        # Parse index page
        if index_url not in self.crawled_URL:
            output += parse_page(index, index_url, default_lang, markup)
            self.crawled_URL.append(index_url)

        # Don't recurse : crawl index/top-most page and return
        if not recurse:
            return output

        # Recurse
        for url in index.find_all('a', href=True):
            currentURL = relative_to_absolute(radical_url(url["href"]), domain)
            if website in currentURL and currentURL not in self.crawled_URL:
                if contains_str in currentURL:
                    page = get_page_content(currentURL)
                    output += parse_page(page, currentURL, default_lang, markup)

                    # Find translations if any
                    for lang in langs:
                        link_tag = page.find(
                            'link', {'rel': 'alternate', 'hreflang': lang})

                        if link_tag and link_tag["href"]:
                            translatedURL = relative_to_absolute(link_tag["href"], domain)
                            t_page = get_page_content(translatedURL)
                            output += parse_page(t_page, translatedURL, lang, markup)

                # Remember we crawled this
                self.crawled_URL.append(currentURL)

                # Follow internal links once content is scraped
                _child = currentURL.replace(website, "")
                output += self.get_website_from_crawling(
                    website, default_lang, child=_child, langs=langs, markup=markup, contains_str=contains_str, recurse=recurse)

        return output

    def get_website_from_sitemap(self, website: str,
                                 default_lang: str,
                                 sitemap: str = "/sitemap.xml",
                                 langs: tuple[str] = ("en", "fr"),
                                 markup: str = "body") -> list[tuple[str, str, str]]:
        """Crawl all pages of a website from an XML sitemap. Intended for word2vec training.

        Supports recursion through sitemaps of sitemaps.

        [get_website_from_crawling][.Crawler.get_website_from_crawling]

        Arguments:
          website (str): root of the website, including `https://` or `http://` but without trailing slash.
          sitemap (str): relative path of the sitemap
          langs (list[str]): ISO-something 2-letters code for alternate language

        Returns:
          list of tuples: `(page content, page language, page URL)`
        """
        index_page = get_page_content(website + sitemap)
        domain = re.search(
            r"(https?:\/\/[a-z0-9\-\_]+?\.[a-z0-9]{2,})", website).group(0)

        # Sitemaps of sitemaps enclose elements in `<sitemap> </sitemap>`
        # While sitemaps of pages enclose them in `<url> </url>`.
        # In both cases, we find URL in `<loc>` and dates in `<lastmod>`
        # Blindly look for both and concatenate the lists
        links = index_page.find_all('sitemap') + index_page.find_all('url')

        print("%i URLs found in sitemap" % len(links))
        output = []

        for link in links:
            url = link.find("loc")
            date = link.find("lastmod")

            if not url:
                print("No URL found in ", link)
                # Nothing to process, ignore this item
                pass

            if not date:
                # Defaults to dummy date
                date = None
            else:
                date = date.get_text()

            currentURL = relative_to_absolute(url.get_text(), domain)
            print(currentURL, date)

            if '.xml' not in currentURL:
                page = get_page_content(currentURL)
                output += parse_page(page, currentURL, default_lang, markup=markup, date=date)

                # Find translations if any
                for lang in langs:
                    link_tag = page.find(
                        'link', {'rel': 'alternate', 'hreflang': lang})

                    if link_tag and link_tag["href"]:
                        translatedURL = relative_to_absolute(
                            link_tag["href"], domain)

                        t_page = get_page_content(translatedURL)
                        output += parse_page(page, translatedURL, lang, markup=markup, date=date)

                        # Remember we crawled this
                        self.crawled_URL.append(translatedURL)

                # Remember we crawled this
                self.crawled_URL.append(currentURL)

            else:
                # We got a sitemap of sitemaps, recurse over the sub-sitemaps
                _sitemap = currentURL.replace(website, "")
                output += self.get_website_from_sitemap(
                    website, default_lang, sitemap=_sitemap, langs=langs, markup=markup)

        return output
