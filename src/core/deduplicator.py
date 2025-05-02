"""Find and remove duplicates and near-duplicates in a list of [core.types.web_pages][]

© 2024 - Aurélien Pierre.
"""
import os
from collections import Counter
import concurrent
from datetime import datetime, timezone, timedelta

import requests
import Levenshtein

import numpy as np
from guppy import hpy
h=hpy()

from . import patterns
from . import nlp
from .types import web_page, get_web_pages_ram
from .utils import guess_date, typography_undo, clean_whitespaces, get_available_ram, get_models_folder, timeit

class Deduplicator():
    urls_to_ignore: list[str] = [
        "/tag/",
        "/tags/",
        "/category/",
        "/categories/",
        "/author/",
        "/authors/",
        "/profil/",
        "/profiles/",
        "/user/",
        "/users/",
        "/login/",
        "/signup/",
        "/member/",
        "/members/",
        "/cart/",
        "/shop/",
        "/register",
    ]
    """URL substrings to find in URLs and remove matching web pages: mostly WordPress archive pages, user profiles and login pages."""

    executor = None

    @staticmethod
    def discard_post(url, discard):
        for elem in discard:
            if elem in url:
                return True

        return False

    @classmethod
    def prepare_posts_parallel(cls, elem, discard_params, urls_to_ignore, fix_urls):
        url = patterns.URL_PATTERN.match(elem["url"].rstrip("/"), concurrent=True)
        if url and not cls.discard_post(elem["url"], urls_to_ignore):
            # Canonify the URL: remove params and anchors
            protocol = url.group(1)
            domain = url.group(2)
            page = url.group(3)
            params = url.group(4) if url.group(4) else ""
            anchor = url.group(5) if url.group(5) else ""

            # See if an https variant of the http page is available.
            # This avoids http/https duplicates.
            if fix_urls and protocol == "http":
                test_url = "https://" + domain + page + params + anchor
                try:
                    response = requests.head(test_url, timeout=2, allow_redirects=True)
                    if response.status_code == 200:
                        # Found a valid page -> convert to https
                        protocol = "https"
                except:
                    pass # timeout

            # Wikipedia mobile version: redirect to desktop version
            domain = domain.replace(".m.wikipedia.org", ".wikipedia.org")

            # See if a non-www. variant of the domain is available
            # This avoids www.domain.ext/domain.ext duplicates
            if fix_urls and domain.startswith("www."):
                test_url = protocol + domain.lstrip("www.") + page + params + anchor
                try:
                    response = requests.head(test_url, timeout=2, allow_redirects=True)
                    if response.status_code == 200:
                        # Found a valid page -> remove www.
                        domain = domain.lstrip("www.")
                except:
                    pass # timeout

            elem["domain"] = domain

            if "/#/" in elem["url"]:
                # Matrix chat links use # as a "page" and make anchor detection fail big time
                new_url = elem["url"]
            else:
                new_url = protocol + "://" + domain + page

            if params and (params.startswith("?lang=") or params.startswith("?v=") \
                or not discard_params):
                # Non SEO-friendly way of translating pages and Youtube videos
                # Need to keep it
                new_url += params

            if anchor and anchor.startswith("#page="):
                # Long PDF are indexed by page. Keep it.
                new_url += anchor

            # Replace URL by canonical stuff
            elem["url"] = new_url

            # elem["parsed"] will need to have been prepared earlier

            if "length" not in elem or elem["length"] == 0:
                elem["length"] = len(elem["parsed"])

            # Get datetime for age comparison
            if "datetime" not in elem or elem["datetime"] is None:
                elem["datetime"] = guess_date(elem["date"])

            return elem

        return None


    @staticmethod
    def get_unique_urls_parallel(candidates):
        elected = candidates[0]
        category = candidates[0]["category"] if "category" in candidates[0] else ""
        length = 0
        date = datetime.fromtimestamp(0, tz=timezone(timedelta(0)))

        for candidate in candidates:
            cand_date = candidate["datetime"]
            cand_length = candidate["length"]
            cand_category = candidate["category"] if "category" in candidate else ""
            vote = False

            if cand_length > length:
                # Replace by longer content if any
                length = cand_length
                vote = True

            if cand_date > date:
                # Replace by more recent content if any
                date = cand_date
                vote = True
            elif cand_date < date:
                # Cancel replacement if candidate is older
                vote = False
            # else: same age or both undefined date, let length decide

            # Cancel replacement if candidate is external (aka followed recursively from internal links)
            # and we already have a variant indexed from within (from sitemap or Rest API) that
            # might have less noise
            if cand_category == "external" and category != "external":
                vote = False

            if vote:
                elected = candidate
                category = cand_category

        # Replace the list of candidates by the elected one for this URL
        return elected


    def get_unique_urls(self, posts: list[web_page]) -> list[web_page]:
        """Pick the most recent, or otherwise the longer, candidate for each canonical URL.
        """
        # 1. Find canonical URL (and prepare content) for each post
        # 2. Create a dictionnary where keys are canonical URLs and values are a list of candidate pages
        cleaned_set = {}
        for i, elem in enumerate(posts):
            # Note: we can't process that in parallel because then the full list gets copied as many times
            # as cores used, and the RAM has every opportunity to overflow
            posts[i] = self.prepare_posts_parallel(elem, self.discard_params, self.urls_to_ignore, self.fix_urls)

            # Trick to ensure elem is a reference to post[i] hoping this will avoid duplicating memory
            # FIXME: is that really useful AND needed ?
            elem = posts[i]

            if elem and "parsed" in elem and len(elem["parsed"]) > 0:
                # Create a dict where the key is the canonical URL
                # and we aggregate the list of matching objects sharing the same URL.
                cleaned_set.setdefault(elem["url"], [])
                cleaned_set[elem["url"]].append(elem)

        # 3. Extract the best candidate for each canonical URL, aka most recent, or longest, or most accurate
        return [self.get_unique_urls_parallel(item) for item in cleaned_set.values()]


    @staticmethod
    def get_unique_content_parallel(candidates):
        elected = candidates[0]
        date = datetime.fromtimestamp(0, tz=timezone(timedelta(0)))

        for candidate in candidates:
            if candidate["datetime"] > date:
                # Replace by more recent content if any
                date = candidate["datetime"]
                elected = candidate

        # Replace the list of candidates by the elected one for this URL
        return elected


    def get_unique_content(self, posts: list[web_page]) -> list[web_page]:
        """Pick the most recent candidate for each canonical content.

        Return:
            `canonical content: web_page` dictionnary

        """
        cleaned_set = {}
        # 1. Create a dictionnary where keys are canonical parsed content and values are a list of candidate pages sharing the same content
        for elem in posts:
            content = elem["parsed"]
            cleaned_set.setdefault(content, [])
            cleaned_set[content].append(elem)

        del posts

        # 2. Extract the most recent page for each canonical content
        return [self.get_unique_urls_parallel(item) for item in  cleaned_set.values()]


    def get_close_content(self, posts: list[web_page], threshold: float = 0.90, distance: float = 500) -> list[web_page]:
        """Find near-duplicate by computing the Levenshtein distance between pages contents.

        Params:
            posts: dictionnary mapping an unused key to a liste of `crawler.web_page`
            threshold: the minimum distance ratio of Lenvenshtein metric for 2 contents to be assumed duplicates
            distance: for efficiency, the list of web_page is first sorted alphabetically by URL, assuming duplicates
            will share at least the beginning of their URL. From there, duplicates are searched ahead in the list up
            to this distance.

        """

        # Sort posts by URL since we have the most probability
        # to find duplicates at similar URLs
        posts = {post["url"]: post for post in posts}
        posts = dict(sorted(posts.items()))

        elements = [value for value in posts.values()]
        replacements = np.arange(len(elements), dtype=np.int64)

        for i in range(len(elements)):
            if replacements[i] == i:
                # Collect the indices of the near-duplicates
                # The similarity matrix is symmetric,
                # no need to process the lower triangle
                indices = [j for j in range(i, min(len(posts), i + distance))
                           if i == j
                           or (replacements[j] == j
                               and Levenshtein.ratio(elements[i]["parsed"], elements[j]["parsed"]) > threshold)]

                if len(indices) > 1:
                    print(i, "found", len(indices) - 1, "duplicates")

                    length = 0
                    date = datetime.fromtimestamp(0, tz=timezone(timedelta(0)))
                    elected = -1

                    # If duplicates, find the most recent or the longest
                    for idx in indices:
                        vote = False
                        if elements[idx]["length"] > length:
                            length = elements[idx]["length"]
                            vote = True

                        if elements[idx]["datetime"] > date:
                            date = elements[idx]["datetime"]
                            vote = True
                        elif elements[idx]["datetime"] < date:
                            vote = False

                        if vote:
                            elected = idx

                    if elected > -1:
                        # Write the index of the best candidate for the current position
                        replacements[i] = elected

                        # Void the other candidates
                        # Note : idx should be always > i since we test forward
                        for idx in indices:
                            if idx != elected:
                                replacements[idx] = -1

                    # else : replacements[i] = i still
                # else : replacements[i] = i still
            # else: element already removed

        return [elements[i] for i in replacements if i > -1]

    @timeit()
    def __call__(self, posts: list[web_page]) -> list[web_page]:
        """Launch the actual duplicate finder. Note that `posts` will be destroyed in the process, to save RAM."""
        print("Initial number of posts: ", len(posts))
        posts = self.get_unique_urls(posts)

        print("After URL deduplication: ", len(posts))

        posts = self.get_unique_content(posts)

        print("After content deduplication: ", len(posts))

        if self.threshold < 1.0:
            posts = self.get_close_content(posts, threshold=self.threshold, distance=self.distance)

        print("After content near-duplicates removal: ", len(posts))

        # List all unique domains with their frequency
        counts = Counter([post["domain"] for post in posts])
        print(f"got {len(counts)} unique domains")

        # Sort domains by frequency
        counts = dict(sorted(counts.items(), key=lambda counts: counts[1]))

        # Remove domains below page number threshold
        discard_list = []
        if self.n_min > 0:
            discard_list = [domain for domain, counts in counts.items() if counts < self.n_min]
            posts = [item for item in posts if item["domain"] not in discard_list]

        print(len(posts))

        with open(get_models_folder("domains"), 'w', encoding='utf8') as f:
            for key, value in counts.items():
                if key not in discard_list:
                    f.write(f"{key}: {value}\n")

        return posts


    def __init__(self, threshold: float = 0.9, distance: int = 500, discard_params: bool = True, n_min: int = 0, fix_urls: bool = True):
        """Instanciate a depduplicator object.

        The duplicates factorizing takes a list of [core.types.web_page][]

        Duplication detection is done using canonical URLs (removing
        query parameters and anchors) and lowercased, ASCII-converted content.

        You can edit (append or replace) the list of URLs to ignore
        [core.deduplicator.Deduplicator.urls_to_ignore][] before doing the actual process.

        Optionaly, near-duplicates are detected too by computing the
        Levenshtein distance between pages contents (lowercased and
        ASCII-converted). This brings a significant performance penalty
        on large datasets.

        Arguments:
            threshold: the minimum Levenshtein distance ratio between 2 pages contents
                for those pages to be considered near-duplicates and be factorized. If set to
                1.0, the near-duplicates detection is bypassed which results in a huge speed up.
            distance: the near-duplicates search is performed on the nearest elements after the
                [core.crawler.web_page][] list has been ordered alphabetically by URL, for performance, assuming near-duplicates
                will most likely be found on the same domain and at a resembling path.
                The distance parameters defines how many elements ahead we will look into.
            discard_params: on modern CMS that enable "pretty URLs" (URL rewriting), pages will be indexed
                by a `domain/section/subsection/page` and URL query parameters will most likely be used my meaningless
                pages like social sharing links or search results page so this parameter can be set to `True`
                to discard those.
                On Rest-API-driven websites, streaming websites and old CMS using "ugly URLS",
                pages will be indexed by `domain?content=id` and the query parameters need to be kept by setting
                this parameter to `False`
            n_min: domains that have a number of indexed pages below this threshold will be discarded entirely.
                This avoids indexing random dude's website, under the assumption that relevant and reliable domains
                will have several pages indexed.
            fix_urls: attempt to convert `http` to `https` URLs and remove leading `www.`. This sends DNS requests
                to assess if the `https` and `www.`-less variants can be reached, which takes a most 2 s per URL.
                Set to `False` to speed things up.
        """

        self.threshold = threshold
        self.distance = distance
        self.discard_params = discard_params
        self.n_min = n_min
        self.fix_urls = fix_urls
