import re
import time

import requests
from bs4 import BeautifulSoup
from core import patterns
from pypdf import PdfReader
from io import BytesIO


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

headers = {
                'User-Agent': 'Virtual Secretary 0.1 Unix',
                'From': 'youremail@domain.example'  # This is another valid field
            }

def get_content_type(url: str) -> tuple[str, bool]:
    """Probe an URL for headers only to see what type of content it returns.

    Returns:
        type, status (tuple): the type of content (`str`) and the status state (`bool`). Status is `True` if the URL can be reached and fetched, `False` if there is some kind of error or empty response.
    """
    try:
        response = requests.head(url, timeout=30, headers=headers)
        content_type = response.headers['content-type']
        status = response.status_code != 404 and response.status_code != 403
        print(url, content_type, status)
        return content_type, status
    except:
        return "", False


def relative_to_absolute(URL: str, domain: str) -> str:
    """Convert a relative path to absolute by prepending the domain"""
    if "://" in URL:
        return URL
    elif URL.startswith("/"):
        # relative URL: prepend domain
        return "://" + domain + URL
    else:
        # relative URL without /
        return "://" + domain + "/" + URL


def radical_url(URL: str) -> str:
    """Trim an URL to the page (radical) part, removing anchors if any (internal links)"""
    anchor = re.match(r"(.+?)\#(.+)", URL)
    return anchor.groups()[0] if anchor else URL


def _recurse_pdf_outline(reader: PdfReader, outline_elem, document_title: str):
    chapters_titles = []
    chapters_bounds = []

    if isinstance(outline_elem, dict):
        chapters_bounds += [reader._get_page_number_by_indirect(outline_elem.page)]
        chapters_titles += [document_title + " | " + outline_elem.title]

    elif isinstance(outline_elem, list):
        for elem in outline_elem:
            chapters_t, chapters_b = _recurse_pdf_outline(reader, elem, document_title)
            chapters_titles += chapters_t
            chapters_bounds += chapters_b

    return chapters_titles, chapters_bounds


def _get_pdf_outline(reader: PdfReader, document_title:str) -> tuple[list[str], list[int]]:
    chapters_titles = []
    chapters_bounds = []

    for out in reader.outline:
        chapters_t, chapters_b = _recurse_pdf_outline(reader, out, document_title)
        chapters_titles += chapters_t
        chapters_bounds += chapters_b

    # The list of titles and page bounds start at the first section title,
    # we need to handle the content between that and the very beginning of the document.
    return [document_title] + chapters_titles, [0] + chapters_bounds


def get_pdf_content(url: str, lang: str, file_path: str = None) -> list[web_page]:
    """Retrieve a PDF document and parse its content.

    Arguments:
        url: the online address of the document, or the downloading page if the doc is not directly accessible from a GET request (for some old-schools website where downloads are inited from a POST request to some PHP form handler, or publications behind paywall).
        lang: the ISO code of the language
        file_path: local path to the PDF file if the URL can't be directly fetched by GET request. The content will be extracted from the local file but the original/remote URL will still be referenced as the source.
    """
    try:
        if not file_path:
            page = requests.get(url, timeout=30, headers=headers)
            print(f"{url}: {page.status_code}")

            if page.status_code != 200:
                return []

            reader = PdfReader(BytesIO(page.content))
        else:
            reader = PdfReader(file_path)

        # Beware: pypdf converts date from string assuming fixed format without catching exceptions
        try:
            date = reader.metadata.creation_date
        except:
            date = None

        title = reader.metadata.title if reader.metadata.title else url.split("/")[-1]
        print(title)

        if reader.outline:
            results = []
            chapters_titles, chapters_bounds = _get_pdf_outline(reader, title)

            for i in range(0, len(chapters_bounds) - 1):
                print(chapters_titles[i], chapters_bounds[i], chapters_bounds[i + 1])
                n_start = chapters_bounds[i]
                n_end = min(chapters_bounds[i + 1] + 1, len(reader.pages) - 1)
                content = "\n".join([elem.extract_text() for elem in reader.pages[n_start:n_end]])

                if content:
                    # Make up a dummy anchor to make URLs to document sections unique
                    # since that's what is used as key for dictionaries
                    result = web_page(title=chapters_titles[i],
                                        url=f"{url}#{i}",
                                        date=date,
                                        content=content,
                                        excerpt=None,
                                        h1=[],
                                        h2=[],
                                        lang=lang)
                    print(result)
                    results.append(result)

            return results

        else:
            excerpt = reader.metadata.subject
            content = "\n".join([elem.extract_text() for elem in reader.pages])

            if content:
                result = web_page(title=title,
                                    url=url,
                                    date=date,
                                    content=content,
                                    excerpt=excerpt,
                                    h1=[],
                                    h2=[],
                                    lang=lang)
                print(result)
                return [result]
            return []

    except Exception as e:
        print(e)
        return []


def get_page_content(url) -> BeautifulSoup:
    """Request a page through the network and feed its response to a BeautifulSoup handler"""
    # Prevent being thresholded on some servers
    #time.sleep(0.5)

    try:
        page = requests.get(url, timeout=30, headers=headers)
        print(f"{url}: {page.status_code}")

        if page.status_code != 200:
            return None

        # Of course some institutionnal websites don't use UTF-8, so let's guess
        content = page.content
        try:
            # Try UTF-8
            content = content.decode()
        except (UnicodeDecodeError, AttributeError):
            try:
                content = content.decode(page.apparent_encoding)
            except (UnicodeDecodeError, AttributeError):
                try:
                    content = content.decode(page.encoding)
                except (UnicodeDecodeError, AttributeError):
                    # We just have to hope it's pure text
                    pass

        # Minified HTML doesn't have line breaks after block-level tags.
        # This is going to make sentence tokenization a nightmare because BeautifulSoup doesn't add them in get_text()
        # Re-introduce here 2 carriage-returns after those tags to create paragraphs.
        unminified = re.sub(r"(\<\/(?:div|section|main|section|aside|header|footer|nav|time|article|h[1-6]|p|ol|ul|li|details|pre|dl|dt|dd|table|tr|th|td|blockquote|style|img|audio|video|iframe|embed|figure|canvas|fieldset|hr|caption|figcaption|address|form|noscript|select)\>)",
                            r"\1\n\n\n\n", content)
        # Same with inline-level tags, but only insert space, except for superscript and subscript
        unminified = re.sub(r"(\<\/(?:a|span|time|abbr|b|i|em|strong|code|dfn|big|kbd|label|textarea|input|option|var|q|tt)\>)",
                            r"\1 ", unminified)

        handler = BeautifulSoup(unminified, "html5lib")

        # Remove any kind of machine code and symbols from the HTML doctree because we want natural language only
        # That will also make subsequent parsing slightly faster.
        # Remove blockquotes too because they can duplicate internal content of forum pages
        for element in handler.select('code, pre, math, style, script, svg, img, audio, video, iframe, embed, blockquote, quote, aside, nav'):
            element.decompose()

        # Remove inline style and useless attributes too
        for attribute in ["data", "style", "media"]:
            for tag in handler.find_all(attrs={attribute: True}):
                del tag[attribute]

        return handler

    except Exception as e:
        print(e)
        return None


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
            # Get the inner text
            results = [tag.get_text() for tag in elements]
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

    return excerpt["content"] if excerpt and "content" in excerpt else None


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

    # Get date - hard if no sitemap with timestamps.
    if not date:
        date = get_date(page)

    h1 = {tag.get_text() for tag in page.find_all("h1")}
    h2 = {tag.get_text() for tag in page.find_all("h2")}

    # Get content - easy : user-request
    content = get_page_markup(page, markup=markup)

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
                                  max_recurse_level = -1,
                                  recursion_level = 0) -> list[tuple[str, str, str]]:
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
        output = []
        index_url = radical_url(website + child)

        # Abort now if the page was already crawled or recursion level reached
        if index_url in self.crawled_URL or \
            (max_recurse_level > -1 and recursion_level >= max_recurse_level):
            return output

        # Extract the domain name, to prepend it if we find relative URL while parsing
        split_domain = patterns.URL_PATTERN.search(website)
        domain = split_domain.group(1)

        # Fetch and parse current (top-most) page
        content_type, status = get_content_type(index_url)

        if "text" in content_type and status:
            index = get_page_content(index_url)

            if (contains_str in index_url or recursion_level == 0):
                # For the first recursion level, ignore "url contains" rule to allow parsing index pages
                if index:
                    # Valid HTML response
                    output += self.parse_original(index, index_url, default_lang, markup, None)
                    output += self.parse_translations(index, domain, markup, None, langs)
                else:
                    # Some websites display PDF in web applets on pages
                    # advertising content-type=text/html but UTF8 codecs
                    # fail to decode because it's actually PDF.
                    # If we end up here, it's most likely what we have.
                    output += get_pdf_content(index_url, default_lang)

            # Recall we passed there, whether or not we actually mined something
            self.crawled_URL.append(index_url)

            # Follow internal links whether or not this page was mined
            if index and recursion_level + 1 != max_recurse_level:
                for url in index.find_all('a', href=True):
                    currentURL = radical_url(relative_to_absolute(url["href"], domain))
                    if (domain in currentURL or currentURL.lower().endswith(".pdf")) \
                        and currentURL not in self.crawled_URL:
                        # Parse only local pages unless they are PDF docs
                        _child = re.sub(r"(http)?s?(\:\/\/)?%s" % domain, "", currentURL)
                        output += self.get_website_from_crawling(
                            website, default_lang, child=_child, langs=langs, markup=markup, contains_str=contains_str,
                            recursion_level=recursion_level + 1, max_recurse_level=max_recurse_level)

        elif "pdf" in content_type and status:
            output += get_pdf_content(index_url, default_lang)
            self.crawled_URL.append(index_url)
            # No link to follow from PDF docmuents
        else:
            # Got an image, video, compressed file, binary, etc.
            self.crawled_URL.append(index_url)

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
        output = []

        index_page = get_page_content(website + sitemap)
        if not index_page:
            return output

        split_domain = patterns.URL_PATTERN.search(website)
        domain = split_domain.group(1)

        # Sitemaps of sitemaps enclose elements in `<sitemap> </sitemap>`
        # While sitemaps of pages enclose them in `<url> </url>`.
        # In both cases, we find URL in `<loc>` and dates in `<lastmod>`
        # Blindly look for both and concatenate the lists
        links = index_page.find_all('sitemap') + index_page.find_all('url')

        print("%i URLs found in sitemap" % len(links))

        for link in links:
            url = link.find("loc")
            date = link.find("lastmod")

            if not url:
                print("No URL found in ", link)
                # Nothing to process, ignore this item
                continue

            date = date.get_text() if date else None

            currentURL = relative_to_absolute(url.get_text(), domain)
            print(currentURL, date)

            if '.xml' not in currentURL:
                # We got a proper web page, parse it
                page = get_page_content(currentURL)
                self.crawled_URL.append(currentURL)
                output += self.parse_original(page, currentURL, default_lang, markup, date)
                output += self.parse_translations(page, domain, markup, date, langs)

                # Follow internal links to PDF documents, not necessarily hosted on the same server
                output += self.crawl_pdf(page, domain, default_lang)
            else:
                # We got a sitemap of sitemaps, recurse over the sub-sitemaps
                _sitemap = re.sub(r"(http)?s?(\:\/\/)?%s" % domain, "", currentURL)
                output += self.get_website_from_sitemap(
                    website, default_lang, sitemap=_sitemap, langs=langs, markup=markup)

        return output


    def crawl_pdf(self, page, domain, default_lang):
        """Try to crawl all PDF documents from links referenced in an HTML page."""
        output = []

        if page:
            for url in page.find_all('a', href=True):
                link = radical_url(relative_to_absolute(url["href"], domain))
                if link not in self.crawled_URL:
                    content_type, status = get_content_type(link)
                    if "pdf" in content_type and status:
                        output += get_pdf_content(link, default_lang)

                    self.crawled_URL.append(link)

        return output


    def parse_original(self, page, url, default_lang, markup, date):
        return parse_page(page, url, default_lang, markup=markup, date=date) if page else []


    def parse_translations(self, page, domain, markup, date, langs):
        """Follow `<link rel="alternate" hreflang="lang" href="url">` tags declaring links to alternative language variants for the current HTML page and crawl the target pages. This works only for pages properly defining alternatives in HTML header."""
        output = []

        if not page:
            return output

        for lang in langs:
            link_tag = page.find('link', {'rel': 'alternate', 'hreflang': lang})

            if link_tag and link_tag["href"]:
                translatedURL = relative_to_absolute(link_tag["href"], domain)
                content_type, status = get_content_type(translatedURL)

                if "text" in content_type and status:
                    translated_page = get_page_content(translatedURL)
                    output += self.parse_original(translated_page, translatedURL, lang, markup, date)

                self.crawled_URL.append(translatedURL)

        return output
