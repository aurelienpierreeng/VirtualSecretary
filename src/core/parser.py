from bs4 import BeautifulSoup
from dataclasses import dataclass, field

import json
import copy

from .utils import clean_whitespaces

@dataclass(slots=True)
class ParsedHTML:
    """
    Wrapper around BeautifulSoup with precomputed crawler metadata.
    """

    soup: BeautifulSoup

    links: list[str] = field(default_factory=list)
    h1: set[str] = field(default_factory=set)
    h2: set[str] = field(default_factory=set)

    content: str | None = field(default_factory=str)
    title: str | None = field(default_factory=str)
    excerpt: str | None = field(default_factory=str)

    date: str | None = field(default_factory=str)
    lang: str | None = field(default_factory=str)

    scripts: list[str] = field(default_factory=list)

    @classmethod
    def from_html(cls, html: str, parser: str = "html5lib") -> "ParsedHTML":
        """
        Build ParsedHTML from raw HTML string.
        """

        soup = BeautifulSoup(html, parser)

        obj = cls(soup=soup)

        # In case of recursive crawling, we need to milk the links out
        # before removing navigation/header elements later.
        obj.links = list({
            href
            for tag in soup.find_all("a", href=True)
            if (href := tag["href"].strip())
            and not href.startswith(("#", "javascript:", "mailto:", "tel:"))
        })

        # Preserve headings before cleaning DOM.
        obj.h1 = {
            tag.get_text().strip(" \n\t\r#↑🔗")
            for tag in soup.find_all("h1")
        }

        obj.h2 = {
            tag.get_text().strip(" \n\t\r#↑🔗")
            for tag in soup.find_all("h2")
        }

        # Extract date before destructive cleanup.
        obj.date = obj.get_date()
        obj.lang = obj.get_lang()

        # Extract inline script contents.
        obj.scripts = copy.deepcopy([
            element.decode_contents()
            for element in soup.select("script")
        ])

        # Remove any kind of machine code and symbols from the HTML doctree because we want natural language only
        # That will also make subsequent parsing slightly faster.
        # Remove blockquotes too because they can duplicate internal content of forum pages.
        # Basically, the goal is to get only the content body of the article/page.
        for element in soup.select('style, script, svg, img, picture, audio, video, iframe, embed, aside, nav, input, header, button, form, fieldset, footer, summary, dialog, textarea, select, option'):
            element.decompose()

        # Remove inline style and useless attributes too
        for attribute in ["data", "style", "media"]:
            for tag in soup.find_all(attrs={attribute: True}):
                del tag[attribute]

        return obj

    def __getattr__(self, name):
        """
        Transparently proxy missing attributes/methods to BeautifulSoup.
        """

        return getattr(self.soup, name)

    def __str__(self) -> str:
        return str(self.soup)

    def __repr__(self) -> str:
        return (
            f"ParsedHTML("
            f"links={len(self.links)}, "
            f"h1={len(self.h1)}, "
            f"h2={len(self.h2)}, "
            f"scripts={len(self.scripts)}"
            f")"
        )

    def get_page_markup(self, markup: str|tuple|list[str]|list[tuple]|None) -> str | None:
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
        seen = set()
        results = []

        if markup is None:
            return None

        if not isinstance(markup, list):
            markup = [markup]

        for item in markup:
            if isinstance(item, tuple):
                # Unroll additional params (classes, ids, etc.)
                elements = self.find_all(item[0], item[1])
            else:
                elements = self.find_all(item)

            print(f"found {len(elements)} {item}")

            for tag in elements:
                # Get the inner text
                text = tag.get_text()
                if text not in seen:
                    results.append(text)
                    seen.add(text)

        return clean_whitespaces("\n\n".join(results))


    def get_excerpt(self) -> str | None:
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
            excerpt = self.find(excerpt_options[i][0], excerpt_options[i][1])
            i += 1

        return clean_whitespaces(excerpt["content"]) if excerpt and "content" in excerpt else None


    def get_date(self) -> str | None:
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

        def method_7(html: BeautifulSoup):
            test = html.find("div", {"class": "dateline"})
            return test.get_text() if test else None

        def method_9(html: BeautifulSoup):
            test = html.find("span", {"class": "updated rich-snippet-hidden"})
            return test.get_text() if test else None

        def method_8(html: BeautifulSoup):
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
            if not test:
                return None
            
            try:
                inner = json.loads(test.get_text())
                if isinstance(inner, dict):
                    return inner.get("dateModified") or inner.get("datePublished")

            except Exception:
                pass

            return None

        date = None
        bag_of_methods = (method_0, method_1, method_2, method_3, method_4, method_5, method_6, method_7, method_8, method_9)

        i = 0
        while not date and i < len(bag_of_methods):
            date = bag_of_methods[i](self.soup)
            i += 1

        return date


    def get_lang(self) -> str | None:
        """Attempt to find the page language"""

        def method_0(html: BeautifulSoup):
            if html.html and html.html.has_attr("lang"):
                lang = html.html["lang"]
                return lang or None

            return None

        def method_1(html: BeautifulSoup):
            test = html.find("meta", {"property": "og:locale", "content": True})
            return test["content"] if test else None

        lang = None
        bag_of_methods = (method_0, method_1)

        i = 0
        while not lang and i < len(bag_of_methods):
            lang = bag_of_methods[i](self.soup)
            i += 1

        return lang
    
    
    def get_title(self) -> str | None:
        title = self.find("title")
        if title:
            title = title.get_text()
        elif len(self.h1) > 0:
            title = list(self.h1)[0]
        elif self.content and len(self.content) > 50:
            title = self.content[0:50]
        
        if title:
            return clean_whitespaces(title)
        else:
            return None
        

    def parse(self, markup: str|tuple|list[str]|list[tuple]|None):
        self.content = self.get_page_markup(markup)
        self.excerpt = self.get_excerpt()
        self.title = self.get_title()