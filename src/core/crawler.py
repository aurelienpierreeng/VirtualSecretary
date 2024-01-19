"""Module containing utilities to crawl websites for HTML, XML and PDF pages for their text content. PDF can be read from their text content if any, or through optical characters recognition for scans. Websites can be crawled from a `sitemap.xml` file or by following internal links recursively from and index page. Each page is aggregated on a list of [core.crawler.web_page][] objects, meant to be used as input to train natural language AI models and to index and rank for search engines.

© 2023 - Aurélien Pierre
"""

import regex as re
import time

import requests
import copy
from bs4 import BeautifulSoup
from core import patterns
from pypdf import PdfReader
from io import BytesIO
import pytesseract
import cv2
import pdf2image
from io import BytesIO
from PIL import Image
import numpy as np
from urllib.parse import urljoin
import multiprocessing

from core import utils

from typing import TypedDict

class web_page(TypedDict):
    """Typed dictionnary representing a web page and its metadata. It can also be used for any text document having an URL/URI"""

    title: str
    """Title of the page"""

    url: str
    """Where to find the page on the network. Can be a local or distant URI, with or without protocol, or even an unique identifier."""

    date: str
    """Date of the last modification of the page, to assess relevance of the content."""

    content: str
    """The actual content of the page."""

    excerpt: str
    """Shortened version of the content for search results previews. Typically provided as `description` meta tag by websites."""

    h1: set[str]
    """Title of the post if any. There should be only one h1 per page, matching title, but some templates wrongly use h1 for section titles."""

    h2: set[str]
    """Section titles if any"""

    lang: str
    """2-letters code of the page language. Not used internally, it's important only if you need to use it in implementations."""

    category: str
    """Arbitrary category or label set by user"""


headers = {
            'User-Agent': 'Virtual Secretary 0.1 Unix',
            'From': 'youremail@domain.example'  # This is another valid field
            }

def get_content_type(url: str) -> tuple[str, bool]:
    """Probe an URL for HTTP headers only to see what type of content it returns.

    Returns:
        type (str): the type of content, like `plain/html`, `application/pdf`, etc.
        status (bool): the state flag:

            - `True` if the URL can be reached and fetched,
            - `False` if there is some kind of error or empty response.
    """
    try:
        response = requests.head(url, timeout=60, headers=headers, allow_redirects=True)
        content_type = response.headers['content-type']
        status = response.status_code == 200
        return content_type, status
    except:
        return "", False


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
    if "://" in URL:
        return URL
    else:
        # relative URL from current page
        if current_page is None:
            raise TypeError("`current_page` should be defined")

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
        else:
            # relative path declared from current page. Hard
            test_url = urljoin(current_page, URL)

        return test_url


def radical_url(URL: str) -> str:
    """Trim an URL to the page (radical) part, removing anchors if any (internal links)

    Examples:
        >>> radical_url("http://me.com/page#section-1")
        "http://me.com/page"
    """
    anchor = re.match(r"(.+?)\#(.+)", URL)
    return anchor.groups()[0] if anchor else URL


def ocr_pdf(document: bytes, output_images: bool = False, path: str = None,
            repair: int = 1, upscale: int = 3, contrast: float = 1.5, sharpening: float = 1.2, threshold: float = 0.4,
            tesseract_lang: str = "eng+fra+equ", tesseract_bin: str = None) -> str:
    """Extract text from PDF using OCR through [Tesseract](https://github.com/tesseract-ocr/tesseract). Both the binding [Python package PyTesseract](https://pypi.org/project/pytesseract/#installation) __and__ the [Tesseract binaries](https://tesseract-ocr.github.io/tessdoc/Installation.html) need to be installed.

    To run on a server where you don't have `sudo` access to install package, you will need to download the [AppImage package](https://tesseract-ocr.github.io/tessdoc/Installation.html#appimage) and pass its path to the `tesseract_bin` argument.

    Tesseract uses machine-learning to identify words and needs the relevant language models to be installed on the system as well. Linux packaged version of Tesseract seem to generally ship French, English and equations (math) models by default. Other languages need to be installed manually,  see [Tesseract docs](https://tesseract-ocr.github.io/tessdoc/Data-Files#data-files-for-version-400-november-29-2016) for available packages. Use `pytesseract.get_languages(config='')` to list available language packages installed locally.

    The OCR is preceeded by an image processing step aiming at text reconstruction, by sharpening, increasing contrast and iteratively reconstructing holes in letters using an inpainting method in wavelets space. This is computationaly expensive, which may not be suitable to run on server.

    Arguments:
        document: the PDF document to open.
        output_images: set to `True`, each page of the document is saved as PNG in the `path` directory before and after contrast enhancement. This is useful to tune the image contrast and sharpness enhancements, prior to OCR.
        repair: number of iterations of enhancements (sharpening, contrast and inpainting) to perform. More iterations take longer, too many iterations might simplify their geometry (as if they were fluid and would drip, removing corners and pointy ends) in a way that actually degrades OCR.
        upscale: upscaling factor to apply before enhancement. This can help recovering ink leaks but takes more memory and time to compute.
        contrast: `1.0` is the neutral value. Moves RGB values farther away from the threshold.
        sharpening: `1.0` is the neutral value. Increases sharpness. Values too high can produce ringing (replicated ghost edges).
        threshold: the reference value (fulcrum) for contrast enhancement. Good values are typically in the range 0.20-0.50.
        tesseract_lang: the Tesseract command argument `-l` defining languages models to use for OCR. Languages are referenced by their 3-letters ISO-something code. See [Tesseract doc](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html#using-multiple-languages) for syntax and meaning. You can mix several languages by joining them with `+`.
        tesseract_bin: the path to the Tesseract executable if it is not in the global CLI path. This is passed as-is to `pytesseract.pytesseract.tesseract_cmd` of the [PyTesseract](https://pypi.org/project/pytesseract/) binding library.

    Returns:
        All the retrieved text from all the PDF pages as a single string. No pagination is done.

    Raises:
        RuntimeError: when using a language package is attempted while Tesseract has no such package installed.
    """
    count = 0
    content = ""

    if tesseract_bin:
        pytesseract.pytesseract.tesseract_cmd = tesseract_bin

    tesseract_langs = tesseract_lang.split("+")
    for _lang in tesseract_langs:
        if _lang not in pytesseract.get_languages(config=''):
            raise RuntimeError("The Tesseract language package `%s` is not installed on this system. Visit https://tesseract-ocr.github.io/tessdoc/Data-Files" % _lang)

    for image in pdf2image.convert_from_bytes(document):
        if output_images and path:
            image.save(path + "-" + str(count) + "-in.png")

        # Convert image to grayscale if it's RGB(a)
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Convert to float, un-gamma and invert
        gray = (gray / 255.)**2.4

        # Upsample
        gray = cv2.resize(gray, (gray.shape[1] * upscale,
                                 gray.shape[0] * upscale))

        # Iterative laplacian pyramid sharpening + 4th order diffusing
        for iter in range(repair):
            LF = gray
            residual = np.zeros_like(gray)
            for i in [3, 5, 9]:
                LF_2 = cv2.GaussianBlur(LF, (i, i), 0)
                HF = LF - LF_2
                residual += (cv2.GaussianBlur(HF, (i, i), 0) + HF) / 2 * sharpening
                LF = LF_2

            # Reconstruct the pyramid
            gray = np.clip(residual + LF, 0, np.inf)

            # Contraste : engraisse le noir
            gray = (gray / threshold)**contrast * threshold

        # Convert back to uint8 and redo gamma
        gray = (np.clip(gray, 0, 1)**(1./2.4) * 255).astype(np.uint8)

        if output_images and path:
            to_save = Image.fromarray(cv2.resize(gray, (img.shape[1], img.shape[0])))
            to_save.save(path + "-" + str(count) + "-out.png")
            count += 1

        # OCR
        page = pytesseract.image_to_string(gray, lang=tesseract_lang).strip("\n ")
        content += "\n" + page
        print(page)

    return content.strip("\n ")


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


hyphenized = re.compile(r"-[\n\r](?=\w)")


def get_pdf_content(url: str,
                    lang: str,
                    file_path: str = None,
                    process_outline: bool = True,
                    category: str = None,
                    ocr: int = 1,
                    **kwargs) -> list[web_page]:
    """Retrieve a PDF document through the network with HTTP GET or from the local filesystem, and parse its text content, using OCR if needed. This needs a functionnal network connection if `file_path` is not provided.

    Arguments:
        url: the online address of the document, or the downloading page if the doc is not directly accessible from a GET request (for some old-schools website where downloads are inited from a POST request to some PHP form handler, or publications behind a paywall).
        lang: the ISO code of the language.
        file_path: local path to the PDF file if the URL can't be directly fetched by GET request. The content will be extracted from the local file but the original/remote URL will still be referenced as the source.
        process_outline: set to `True` to split the document according to its outline (table of content), so each section will be in fact a document in itself. PDF pages are processed in full, so sections are at least equal to a page length and there will be some overlapping.
        category: arbitrary category or label set by user
        ocr:
            - `0` disables any attempt at using OCR,
            - `1` enables OCR through Tesseract if no text was found in the PDF document
            - `2` forces OCR through Tesseract even when text was found in the PDF document.
        See [core.crawler.ocr_pdf][] for info regarding the Tesseract environment. You will need to manually disable

    Other parameters:
        **kwargs: directly passed-through to [core.crawler.ocr_pdf][]. See this function documentation for more info.

    Returns:
        a list of [core.crawler.web_page][] objects holding the text content and the PDF metadata
    """
    try:
        # Open the document from local or remote storage
        document : BytesIO
        if not file_path:
            page = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
            print(f"{url}: {page.status_code}")

            if page.status_code != 200:
                print("couldn't download %s" % url)
                return []

            document = BytesIO(page.content)
        else:
            document = open(file_path, "rb")

        blob = document.read() # need to backup PDF content here because PdfReader kills it next
        reader = PdfReader(document)

        # Beware: pypdf converts date from string assuming fixed format without catching exceptions.
        # need to catch them here to avoid plain crash.
        try:
            date = reader.metadata.creation_date
        except:
            date = None

        title = reader.metadata.title if reader.metadata.title else url.split("/")[-1]
        excerpt = reader.metadata.subject

        print(title)

        # Check if the PDF contains text
        content = "\n".join([elem.extract_text() for elem in reader.pages]).strip("\n ")

        if (ocr == 1 and len(content) < 20) or ocr == 2:
            # No text, retry with OCR
            try:
                content = ocr_pdf(blob, path=file_path, **kwargs)
            except Exception as e:
                # pylint: disable=invalid-name
                print(e)

        if not (reader.outline and process_outline):
            # Whether or not text comes from OCR, if we save it in one chunk, do it now and exit.
            content = hyphenized.sub("", content)

            if content:
                result = web_page(title=title,
                                    url=url,
                                    date=date,
                                    content=content,
                                    excerpt=excerpt,
                                    h1=[],
                                    h2=[],
                                    lang=lang,
                                    category=category)
                print(result)
                return [result]

        else:
            # Save each outline section in a different document
            results = []
            chapters_titles, chapters_bounds = _get_pdf_outline(reader, title)

            for i in range(0, len(chapters_bounds) - 1):
                print(chapters_titles[i], chapters_bounds[i], chapters_bounds[i + 1])
                n_start = chapters_bounds[i]
                n_end = min(chapters_bounds[i + 1] + 1, len(reader.pages) - 1)
                content = "\n".join([elem.extract_text() for elem in reader.pages[n_start:n_end]]).strip("\n ")
                content = hyphenized.sub("", content)

                if content:
                    # Make up a page anchor to make URLs to document sections unique
                    # since that's what is used as key for dictionaries. Also, Chrome and Acrobat
                    # will be able to open PDF files at the right page with this anchor.
                    result = web_page(title=chapters_titles[i],
                                        url=f"{url}#page={i}",
                                        date=date,
                                        content=content,
                                        excerpt=None,
                                        h1=[],
                                        h2=[],
                                        lang=lang,
                                        category=category)
                    print(result)
                    results.append(result)

            return results

        return []

    except Exception as e:
        print(e)
        return []


@utils.exit_after(120)
def get_page_content(url: str, content: str = None) -> [BeautifulSoup | None, list[str]]:
    """Request an (x)HTML page through the network with HTTP GET and feed its response to a BeautifulSoup handler. This needs a functionnal network connection.

    The DOM is pre-filtered as follow to keep only natural language and avoid duplicate strings:

    - media tags are removed (`<iframe>`, `<embed>`, `<img>`, `<svg>`, `<audio>`, `<video>`, etc.),
    - code and machine language tags are removed (`<script>`, `<style>`, `<code>`, `<pre>`, `<math>`),
    - menus and sidebars are removed (`<nav>`, `<aside>`),
    - forms, fields and buttons are removed(`<select>`, `<input>`, `<button>`, `<textarea>`, etc.)
    - quotes tags are removed (`<quote>`, `<blockquote>`).

    The HTML is un-minified to help end-of-sentences detections in cases where sentences don't end with punctuation (e.g. in titles).

    Arguments:
        url: a valid URL that can be fetched with an HTTP GET request.
        content: a string buffer used as HTML source. If this argument is passed, we don't fetch `url` from network and directly use this input.

    Returns:
        a [bs4.BeautifulSoup][] object initialized with the page DOM for further text mining. `None` if the HTML response was empty or the URL could not be reached. The list of URLs found in page before removing meaningless markup is stored as a list of strings in the `object.links` member. `object.h1` and `object.h2` contain a set of headers 1 and 2 found in the page before removing any markup.
    """

    try:
        if content is None and url is not None:
            page = requests.get(url, timeout=30, headers=headers, allow_redirects=True)
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
                            r"\1\n\n\n\n", content, timeout=30)
        # Same with inline-level tags, but only insert space, except for superscript and subscript
        unminified = re.sub(r"(\<\/(?:a|span|time|abbr|b|i|em|strong|code|dfn|big|kbd|label|textarea|input|option|var|q|tt)\>)",
                            r"\1 ", unminified, timeout=30)

        handler = BeautifulSoup(unminified, "html5lib")

        # In case of recursive crawling, we need to milk the links out before we remove <nav> at the next step
        handler.links = [url["href"] for url in handler.find_all('a', href=True) if url["href"]]

        # Same with h1 because we will remove <header> and that's where it might be
        # Doe h2 as well since we are at it.
        handler.h1 = {tag.get_text().strip(" \n\t\r#↑🔗") for tag in handler.find_all("h1")}
        handler.h2 = {tag.get_text().strip(" \n\t\r#↑🔗") for tag in handler.find_all("h2")}

        # Remove any kind of machine code and symbols from the HTML doctree because we want natural language only
        # That will also make subsequent parsing slightly faster.
        # Remove blockquotes too because they can duplicate internal content of forum pages.
        # Basically, the goal is to get only the content body of the article/page.
        for element in handler.select('code, pre, math, style, script, svg, img, picture, audio, video, iframe, embed, blockquote, quote, aside, nav, input, header, button, form, fieldset, footer, summary, dialog, textarea, select, option'):
            element.decompose()

        # Remove inline style and useless attributes too
        for attribute in ["data", "style", "media"]:
            for tag in handler.find_all(attrs={attribute: True}):
                del tag[attribute]

        return handler

    except Exception as e:
        print(e)
        return None


def get_page_markup(page: BeautifulSoup, markup: str|tuple|list[str]|list[tuple]) -> str:
    """Extract the text content of an HTML page DOM by targeting only the specific tags.

    Arguments:
        page: a [bs4.BeautifulSoup][] handler with pre-filtered DOM,
        markup: any kind of tags supported by [bs4.BeautifulSoup.find_all][]:

            - (str): the single tag to select. For example, `"body"` will select `<body>...</body>`.
            - (tuple): the tag and properties to select. For example, `("div", { "class": "right" })` will select `<div class="right">...</div>`.
            - all combinations of the above can be chained in lists.

    Returns:
        The text content of all instances of all tags in markup as a single string, if any, else an empty string.

    Examples:
        >>> get_page_markup(page, "article")

        >>> get_page_markup(page, ["h1", "h2", "h3", "article"])

        >>> get_page_markup(page, [("div", {"id": "content"}), "details", ("div", {"class": "comment-reply"})])
    """
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

    return utils.clean_whitespaces(output)


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

    - `<meta property="article:modified_time" content="...">`
    - `<time datetime="...">`
    - `<div class="dateline">...</div>`

    Arguments:
        page: a [bs4.BeautifulSoup][] handler with pre-filtered DOM,

    Returns:
        The content of the meta tag if any.
    """

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
        date = get_date(page)

    # Get content - easy : user-request
    content = get_page_markup(page, markup=markup)

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
        "reddit.com/submit",
        "t.me/share", # Telegram share
        "linkedin.com/share",
        "bufferapp.com/add",
        "getpocket.com/edit",
        "tumblr.com/share",
        "mailto:",
        "/profile/",
        "/login/",
        "/user/",
        "/member/",
        ".css",
        ".js",
        ".json",
        ]
    """List of URLs sub-strings that will disable crawling if they are found in URLs. Mostly social networks sharing links."""

    def __init__(self):
        """Crawl a website from its sitemap or by following internal links recusively from an index page."""

        self.crawled_URL: list[str] = []
        """List of URLs already visited"""

    def discard_link(self, url):
        """Returns True if the url is found in the `self.no_follow` list"""
        for elem in self.no_follow:
            if elem in url:
                return True

        return False


    def get_immediate_links(self, page, domain, currentURL, default_lang, langs, category, contains_str) -> list[web_page]:
        """Follow internal and external links contained in a webpage only to one recursivity level,
        including PDF files and HTML pages. This is useful to index references docs linked from a page.
        """
        output = []
        for url in page.find_all('a', href=True):
            nextURL = radical_url(relative_to_absolute(url["href"], domain, currentURL))
            if nextURL not in self.crawled_URL:
                output += self.get_website_from_crawling(nextURL.rstrip("/#"), default_lang, "", langs, max_recurse_level=1, category=category, contains_str=contains_str, _recursion_level=0)

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
                                  _recursion_level: int = 0) -> list[web_page]:
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

        # Abort now if the page was already crawled or recursion level reached
        if self.discard_link(index_url) or index_url in self.crawled_URL:
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
        content_type, status = get_content_type(index_url)

        # Recall we passed there, whether or not we actually mined something
        self.crawled_URL.append(index_url)

        # FIXME: we nest 7 levels of if here. It's ugly but I don't see how else
        # to cover so many cases.
        if status and "text" in content_type \
            and "javascript" not in content_type \
            and "css" not in content_type \
            and "json" not in content_type:

            index = get_page_content(index_url)

            if include or _recursion_level == 0 or ".pdf" in index_url:
                # For the first recursion level, ignore "url contains" rule to allow parsing index pages
                if index:
                    # Valid HTML response
                    output += self._parse_original(index, index_url, default_lang, markup, None, category)
                    output += self._parse_translations(index, domain, index_url, markup, None, langs, category)
                else:
                    # Some wgebsites display PDF in web applets on pages
                    # advertising content-type=text/html but UTF8 codecs
                    # fail to decode because it's actually not HTML but PDF.
                    # If we end up here, it's most likely what we have.
                    output += get_pdf_content(index_url, default_lang, category=category)

            # Follow internal links whether or not this page was mined
            if index and _recursion_level + 1 != max_recurse_level:
                for url in index.links:
                    currentURL = radical_url(relative_to_absolute(url, domain, index_url))
                    if currentURL in self.crawled_URL:
                        continue

                    current_address = patterns.URL_PATTERN.search(currentURL)
                    if not current_address:
                        continue

                    current_domain = current_address.group(2)
                    current_page = current_address.group(3)
                    current_url_params = current_address.group(4) if current_address.group(4) else ""

                    if not restrict_section and domain == current_domain:
                        # Recurse only through local pages, aka :
                        # 1. domains match
                        output += self.get_website_from_crawling(
                            website, default_lang, child=current_page + current_url_params, langs=langs, markup=markup, contains_str=contains_str,
                            _recursion_level=_recursion_level + 1, max_recurse_level=max_recurse_level, restrict_section=restrict_section, category=category)
                    elif restrict_section and domain == current_domain and child in current_page:
                        # Recurse only through local subsections, aka :
                        # 1. domains match
                        # 2. current page is in a subsection of current child
                        output += self.get_website_from_crawling(
                            website, default_lang, child=current_page + current_url_params, langs=langs, markup=markup, contains_str=contains_str,
                            _recursion_level=_recursion_level + 1, max_recurse_level=max_recurse_level, restrict_section=restrict_section, category=category)
                    elif include:
                        # Follow internal and external links on only one recursivity level.
                        # Aka HTML reference pages (Wikipedia) and attached PDF (docs, manuals, spec sheets)
                        output += self.get_website_from_crawling(
                            currentURL, default_lang, "", langs, contains_str="", max_recurse_level=1, restrict_section=restrict_section, category=category,
                            _recursion_level=0)

        elif "pdf" in content_type and status:
            output += get_pdf_content(index_url, default_lang, category=category)
            self.crawled_URL.append(index_url)
            # No link to follow from PDF docmuents
        else:
            # Got an image, video, compressed file, binary, etc.
            self.crawled_URL.append(index_url)

        return output

    def get_website_from_sitemap(self,
                                 website: str,
                                 default_lang: str,
                                 sitemap: str = "/sitemap.xml",
                                 langs: tuple[str] = ("en", "fr"),
                                 markup: str = "body",
                                 category: str = None,
                                 contains_str: str | list[str] = "") -> list[web_page]:
        """Recursively crawl all pages of a website from links found in a sitemap. This applies to all HTML pages hosted on the domain of `website` and to PDF documents either from the current domain or from external domains but referenced on HTML pages of the current domain. Sitemaps of sitemaps are followed recursively.

        Arguments:
            website: root of the website, including `https://` or `http://` without trailing slash.
            default_lang: provided or guessed main language of the website content. Not used internally.
            sitemap: relative path of the XML sitemap.
            langs: ISO-something 2-letters code of the languages for which we attempt to fetch the translation if available, looking for the HTML `<link rel="alternate" hreflang="...">` tag.
            markup: see [core.crawler.get_page_markup][]
            category: arbitrary category or label
            contains_str: limit recursive crawling from sitemap-defined pages to pages containing this string or list of strings. Will get passed as-is to [get_website_from_crawling][].

        Returns:
            a list of all valid pages found. Invalid pages (wrong markup, empty HTML response, 404 errors) will be silently ignored.

        Examples:
            >>> from core import crawler
            >>> cr = crawler.Crawler()
            >>> pages = cr.get_website_from_sitemap("https://aurelienpierre.com", default_lang="fr", markup=("div", { "class": "post-content" }))
        """
        output = []

        index_page = get_page_content(website + sitemap)
        if not index_page:
            return output

        split_domain = patterns.URL_PATTERN.search(website)
        domain = split_domain.group(2)

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

            currentURL = relative_to_absolute(url.get_text(), domain, website + sitemap)
            print(currentURL, date)

            if '.xml' not in currentURL:
                # TODO: discard JSON content
                page = get_page_content(currentURL)
                self.crawled_URL.append(currentURL)

                if page:
                    # We got a proper web page, parse it
                    output += self._parse_original(page, currentURL, default_lang, markup, date, category)
                    output += self._parse_translations(page, domain, currentURL, markup, date, langs, category)

                    # Follow internal and external links on only one recursivity level
                    output += self.get_immediate_links(page, domain, currentURL, default_lang, langs, category, contains_str)
            else:
                # We got a sitemap of sitemaps, recurse over the sub-sitemaps
                _sitemap = re.sub(r"(http)?s?(\:\/\/)?%s" % domain, "", currentURL)
                output += self.get_website_from_sitemap(
                    website, default_lang, sitemap=_sitemap, langs=langs, markup=markup, category=category)

        return output


    def _crawl_pdf(self, page, domain, default_lang, category):
        """Try to crawl all PDF documents from links referenced in an HTML page."""
        output = []

        if page:
            for url in page.find_all('a', href=True):
                link = radical_url(relative_to_absolute(url["href"], domain, None))
                if link not in self.crawled_URL:
                    content_type, status = get_content_type(link)
                    if "pdf" in content_type and status:
                        output += get_pdf_content(link, default_lang, category=category)

                    self.crawled_URL.append(link)

        return output


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
                content_type, status = get_content_type(translatedURL)

                if "text" in content_type and status:
                    translated_page = get_page_content(translatedURL)
                    output += self._parse_original(translated_page, translatedURL, lang, markup, date, category)

                self.crawled_URL.append(translatedURL)

        return output
