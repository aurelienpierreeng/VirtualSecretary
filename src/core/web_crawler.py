import re
import pickle
import os

import requests
from bs4 import BeautifulSoup


def get_data_folder(filename: str) -> str:
    """Resolve the path of a training data saved under `filename`. These are stored in `../../data/`.
    The `.pickle` extension is added automatically.

    Warning:
        This does not check the existence of the file and root folder.
    """
    current_path = os.path.abspath(__file__)
    install_path = os.path.dirname(
                        os.path.dirname(
                            os.path.dirname(current_path)))
    models_path = os.path.join(install_path, "data")
    return os.path.abspath(os.path.join(models_path, filename + ".pickle"))


def save_data(data, filename: str):
    """Save scraped data to a pickle file in data folder. Folder and file extension are handled automtically."""
    with open(get_data_folder(filename), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def relative_to_absolute(URL, domain):
    """Convert a relative path to absolute by prepending the domain"""
    if URL.startswith("/"):
        # relative URL: prepend domain
        return domain + URL
    else:
        return URL


def get_page_content(url) -> BeautifulSoup:
    """Request a page through the network and feed its response to a BeautifulSoup handler"""
    try:
        page = requests.get(url)
        print(f"{url}: {page.status_code}")
        return BeautifulSoup(page.content, 'html.parser')
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
            for body in elements:
                # Remove pre and code markup from the HTML doctree because we want natural language only
                for element in body.select('code, pre, math'):
                    element.decompose()

                body = body.get_text()

                # Replace special unicode spaces
                body = body.replace("\u2002", " ")
                body = body.replace("\u2008", " ")
                body = body.replace("\u202f", " ")
                body = body.replace("\xa0", " ")

                # Append each element as a new line
                output += "\n\n" + body

    return output

def parse_page(page: BeautifulSoup, url: str, lang: str, markup) -> list[tuple[str, str, str]]:
    """Get the requested markup from the requested page URL.

    Returns:
        A tuple containing the page parsed content, the page lang and the page URL, ready for machine-learning.
    """
    content = get_page_markup(page, markup=markup)

    if content:
        return [(content, lang, url)]
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
            currentURL = relative_to_absolute(url["href"], domain)
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

        urls = get_page_content(website + sitemap).find_all('loc')
        print("%i URLs found in sitemap" % len(urls))
        domain = re.search(
            r"(https?:\/\/[a-z0-9\-\_]+?\.[a-z0-9]{2,})", website).group(0)
        output = []

        for url in urls:
            currentURL = relative_to_absolute(url.get_text(), domain)
            print(currentURL)

            if '.xml' not in currentURL:
                page = get_page_content(currentURL)
                output.append(
                    (get_page_markup(page, markup=markup), default_lang, currentURL))

                # Find translations if any
                for lang in langs:
                    link_tag = page.find(
                        'link', {'rel': 'alternate', 'hreflang': lang})

                    if link_tag and link_tag["href"]:
                        translatedURL = relative_to_absolute(
                            link_tag["href"], domain)

                        t_page = get_page_content(translatedURL)
                        output.append(
                            (get_page_markup(t_page, markup=markup), lang, translatedURL))

                        # Remember we crawled this
                        self.crawled_URL.append(translatedURL)

                # Remember we crawled this
                self.crawled_URL.append(currentURL)

            else:
                _sitemap = currentURL.replace(website, "")
                output += self.get_website_from_sitemap(
                    website, default_lang, sitemap=_sitemap, langs=langs, markup=markup)

        return output
