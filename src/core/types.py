from typing import TypedDict
from datetime import datetime as dt
from .utils import guess_date
import numpy as np
import sys

class web_page(TypedDict):
    """Typed dictionnary representing a web page and its metadata. It can also be used for any text document having an URL/URI.

    The database module automatically uses this dictionnary's keys to create DB columns when generating the `web_page` DB.
    Therefore, keys needs to be kept in sync along all the modules, that is modules should not add their own custom keys.

    [PEP 705](https://discuss.python.org/t/pep-705-read-only-typeddict-items/37867/40) adds the ability to declare read-only
    typed dictionnaries, which should be used here as soon as it gets merged to stable Python to forbid custom keys definitions
    in `web_pages` instances from other modules.
    """

    title: str
    """Title of the page"""

    url: str = ""
    """Where to find the page on the network. Can be a local or distant URI, with or without protocol, or even an unique identifier."""

    domain: str = ""
    """Domain of the url"""

    date: str = ""
    """Date of the last modification of the page, to assess relevance of the content, as a string."""

    content: str = ""
    """The actual content of the page in a human-readable way."""

    excerpt: str = ""
    """Shortened version of the content for search results previews. Typically provided as `description` meta tag by websites."""

    h1: list = []
    """Title of the post if any. There should be only one h1 per page, matching title, but some templates wrongly use h1 for section titles."""

    h2: list = []
    """Section titles if any"""

    lang: str = "en"
    """2-letters ISO code of the page language. Not used internally, it's important only if you need to use it in implementations."""

    category: str = None
    """Arbitrary category, tag or label set by user, to be reused for example in AI document tagging."""

    datetime: dt = None
    """The page date as a `datetime.datetime` object directly usable"""

    length: int = 0
    """Characters length of `content`"""

    parsed: str = ""
    """The normalized content of the page (lowercase, possibly converted to simple ASCII characters) for machine view over the content."""

    tokenized: list = [[]]
    """List of parsed content text tokens, including metatokens, if needed, as a list of sentences, where sentences are themselves a list of string tokens."""

    vectorized: np.ndarray = np.empty(0, dtype=np.float32)
    """Precomputed vector representation of the tokenized content."""


def sanitize_web_page(page: web_page, to_db: bool = False) -> web_page:
    """Ensure existence and validity of `web_page` keys/values.

    Params:
        to_db: set to `True` to convert arrays of strings into semicolon-separated strings.
    """
    # Handle legacy code : h1 and h2 used to be sets, now they are lists

    if "h1" in page:
        if isinstance(page["h1"], str):
            page["h1"] = [page["h1"]]
        else:
            page["h1"] = list(page["h1"])

    if "h2" in page:
        page["h2"] = list(page["h2"])

    # Handle legacy code : fields added recently
    if "vectorized" not in page:
        page["vectorized"] = np.empty(0, dtype=np.float32)

    if "tokenized" not in page:
        page["tokenized"] = [[]]

    if "parsed" not in page:
        page["parsed"] = ""

    if "domain" not in page:
        page["domain"] = ""

    if "datetime" not in page:
        page["datetime"] = guess_date(str(page["date"])) if page["date"] else None

    if "length" not in page:
        page["length"] = 0

    if "excerpt" not in page or not page["excerpt"] or len(page["excerpt"]) < 800:
        page["excerpt"] = str(page["content"])[0:min(len(page["content"]), 800)]

    if "category" not in page:
        page["category"] = None

    # Some fields here may be bytes when read from web crawling.
    page["title"] = str(page["title"])
    page["content"] = str(page["content"])

    # Dict are ordered starting with Python 3.7. Problem is, even for a typeddict,
    # the order is the one of key/value assignation. Re-order everything as in the
    # typeddict to ensure a predictable order here (and handle back/for-ward compatibility) :
    page = {k: page[k] for k in web_page.__annotations__.keys()}

    return page


def db_row_to_web_page(row: list[tuple[any]]) -> web_page:
    """Turn an SQL extraction of a full row containing a `web_page`. Columns are matched to keys in the same order.
    The database needs to be saved with columns in the right order, call [core.types.sanitize_web_page][] first"""
    keys = web_page.__annotations__.keys()
    return { k: row[i] for i, k in enumerate(keys) }


# Can't add the following as methods of the web_page TypedDict
# see https://github.com/python/mypy/issues/4201
def get_web_page_ram(item: web_page) -> int:
    """Get RAM usage of a web_page in bytes"""
    memory = sys.getsizeof(item)

    for key, value in item.items():
        memory += sys.getsizeof(key) + sys.getsizeof(value)

    if "parsed" not in item:
        # In case the parsed variant is not already stored in page,
        # estimate it by upper value through the content length
        memory += sys.getsizeof(item["content"])

    return memory


def get_web_pages_ram(web_pages: list[web_page]) -> int:
    """Get RAM usage of a list of web_pages in bytes"""
    memory = sys.getsizeof(web_pages)
    for elem in web_pages:
          memory += get_web_page_ram(elem)
    return memory
