import requests
from bs4 import BeautifulSoup


def get_page_content(url) -> BeautifulSoup:
  page = requests.get(url)
  return BeautifulSoup(page.content, 'html.parser')


def get_page_markup(page: BeautifulSoup, markup: str = "body") -> str:

  output = ""

  if not isinstance(markup, list):
    markup = [markup]

  for item in markup:
    if isinstance(item, tuple):
      # Unroll additional params (classes, ids, etc.)
      elements = page.find_all(item[0], item[1])
    else:
      elements = page.find_all(item)

    print(len(elements))

    if elements:
      for body in elements:
        body = body.get_text()

        # Replace special unicode spaces
        body = body.replace("\u2002", " ")
        body = body.replace("\u2008", " ")
        body = body.replace("\u202f", " ")
        body = body.replace("\xa0", " ")

        output += "\n\n" + body

  return output

class Crawler:
  def __init__(self):
    self.crawled_URL = []

  def get_website_from_crawling(self, website: str, default_lang, child="/", langs=["en", "fr"], markup="body") -> list[tuple[str, str, str]]:
    """Crawl all found pages of a website from the index. Intended for word2vec training.

    Arguments:
      website (str): root of the website, including `https://` or `http://`
      sitemap (str): relative path of the sitemap
      langs (list[str]): ISO-something 2-letters code for alternate language

    Returns:
      list of tuples: `(page content, page language, page URL)`
    """

    urls = get_page_content(website + child).find_all('a', href=True)
    output = []

    for url in urls:
      currentURL = url["href"]

      if website in currentURL and currentURL not in self.crawled_URL:
        print(currentURL)

        page = get_page_content(currentURL)
        output.append((get_page_markup(page, markup=markup), default_lang, currentURL))

        # Find translations if any
        for lang in langs:
          link_tag = page.find('link', {'rel': 'alternate', 'hreflang': lang})

          if link_tag and link_tag["href"]:
            translatedURL = link_tag["href"]

            if translatedURL.startswith("/"):
              # Relative link. Prepend root.
              translatedURL = website + translatedURL

            t_page = get_page_content(translatedURL)
            output.append((get_page_markup(t_page, markup=markup), lang, translatedURL))

        # Remember we crawled this
        self.crawled_URL.append(currentURL)

        # Follow internal links once content is scraped
        _child = currentURL.replace(website, "")
        self.get_website_from_crawling(website, default_lang, _child, langs=langs, markup=markup)

    return output


  def get_website_from_sitemap(self, website: str, default_lang, sitemap="/sitemap.xml", langs=["en", "fr"], markup="body") -> list[tuple[str, str, str]]:
    """Crawl all pages of a website from an XML sitemap. Intended for word2vec training.

    Supports recursion through sitemaps of sitemaps.

    Arguments:
      website (str): root of the website, including `https://` or `http://`
      sitemap (str): relative path of the sitemap
      langs (list[str]): ISO-something 2-letters code for alternate language

    Returns:
      list of tuples: `(page content, page language, page URL)`
    """

    urls = get_page_content(website + sitemap).find_all('loc')
    output = []

    for url in urls:
      currentURL = url.get_text()
      print(currentURL)

      if '.xml' not in currentURL:
        page = get_page_content(currentURL)
        output.append((get_page_markup(page, markup=markup), default_lang, currentURL))

        # Find translations if any
        for lang in langs:
          link_tag = page.find('link', {'rel': 'alternate', 'hreflang': lang})

          if link_tag and link_tag["href"]:
            translatedURL = link_tag["href"]

            if translatedURL.startswith("/"):
              # Relative link. Prepend root.
              translatedURL = website + translatedURL

            t_page = get_page_content(translatedURL)
            output.append((get_page_markup(t_page, markup=markup), lang, translatedURL))

            # Remember we crawled this
            self.crawled_URL.append(translatedURL)

        # Remember we crawled this
        self.crawled_URL.append(currentURL)

      else:
        _sitemap = currentURL.replace(website, "")
        output += self.get_website_from_sitemap(website, default_lang, sitemap=_sitemap, langs=langs, markup=markup)

    return output
