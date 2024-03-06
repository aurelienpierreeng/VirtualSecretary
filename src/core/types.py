from typing import TypedDict
import sys

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
