import requests
import time
import random

import regex as re

from urllib import robotparser

from . import utils

# disable warnings regarding bypass of checks for SSL certs
# assuming we only download public pages here
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from . import patterns

def check_response(url, status_code):
    if status_code == 429:
        print(f"SERVER {url} has throttled us")
    elif status_code == 503:
        print(f"SERVER {url} did not respond. We may be throttled.")
    elif status_code != 200:
        print(f"SERVER {url} did not respond: error {status_code}")


@utils.exit_after(120)
def try_url(url, timeout=30, delay=1.) -> tuple[requests.request, dict, str]:
    """Try URLs with some sanitization (protocol & path), until we found something. Probe only headers, not the actual page."""

    UA = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:129.0) Gecko/20100101 Firefox/129.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 OPR/112.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edge/127.0.2651.98",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 OPR/112.0.0.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    ]
    ua1 = {"User-Agent": random.choice(UA), 'Connection': 'keep-alive'}
    ua2 = {"User-Agent": "Virtual Secretary", 'Connection': 'keep-alive'}
    ua3 = {"User-Agent": "Twitterbot", 'Connection': 'keep-alive'}
    ua4 = {"User-Agent": "Googlebot", 'Connection': 'keep-alive' }
    ua5 = {'Connection': 'keep-alive'}
    agents = [ua1, ua2, ua3, ua4, ua5]

    # URL inside non-processed Markdown syntax can have an orphan/unmatched trailing )
    # But beware of Wikipedia links that can have non-orphan final ) for disambiguation
    url = patterns.remove_unmatched_parentheses(url).rstrip("/.,: ")
    link = patterns.URL_PATTERN.match(url, concurrent=True)

    if link is not None:
        # Canonify the URL: remove params and anchors
        protocol = link.group(1)
        domain = link.group(2)
        page = link.group(3)
        params = link.group(4) if link.group(4) else ""
        anchor = link.group(5) if link.group(5) else ""
    else:
        # Couldn't parse URL, still try it just in case
        return requests.head(url, timeout=timeout, allow_redirects=True, verify=False), {}, url

    # 0: try to see if there is a robots.txt file preventing us from accessing
    robots_txt = protocol + "://" + domain.rstrip("/") + "/robots.txt"
    robot = robotparser.RobotFileParser(robots_txt)
    robot.read()

    # 1: try the given URL with varying headers
    for HEADER in agents:
        if "User-Agent" in HEADER and robot.can_fetch(HEADER["User-Agent"], url) \
            or robot.can_fetch("*", url):
            for test_url in [url, url + "/"]:
                time.sleep(delay)
                result = requests.head(url, timeout=timeout, headers=HEADER, allow_redirects=True, verify=False)
                if result.status_code == 200:
                    return result, HEADER, result.url

    # 2. bruteforce all possible unique combinations in case we have a semi-wrong URL
    # Try : http/https, with/without www., trailing / or not, URL params/anchors or not.
    # Also try to remove trailing () because that might be Markdown link syntax caught in URL detection.
    final_url = url
    for HEADER in agents:
        for PROTOCOL in set([protocol + "://", "http://", "https://"]):
            for DOMAIN in set([domain, patterns.remove_www(domain), domain.rstrip("/")]):
                for PAGE in set([page, page.rstrip("()"), page.rstrip("/"), page + "/" ]):
                    for PARAMS in set([params, "", params.rstrip("()")]):
                        for ANCHOR in set([anchor, "", anchor.rstrip("()")]):
                            test_url = PROTOCOL + DOMAIN + PAGE + PARAMS + ANCHOR
                            if test_url != url and test_url != url + "/":
                                time.sleep(delay)
                                result = requests.head(test_url, timeout=timeout, headers=HEADER, allow_redirects=True, verify=False)
                                if result.status_code == 200:
                                    # valid result
                                    return result, HEADER, result.url
                                elif result.status_code != 404:
                                    # invalid result but at least we know the page exists
                                    # maybe we are just being thresholded or forbidden
                                    final_url = test_url

    # Try the Wayback machine in final resort: "https://web.archive.org/web/"
    result = requests.head("https://web.archive.org/web/" + final_url, timeout=timeout, allow_redirects=True, verify=False)
    return result, {}, result.url


def get_url(url: str, timeout=30, custom_header={}, backend="selenium", driver=None, wait=None) -> tuple[bytes, str, int]:
    """
    Get the content of an URL using requests or selenium.
    `.pdf`, `.xml` and `.txt` URLs always use requests.

    Return:
        content: the raw DOM,
        url: the final URL after possible redirections
        status: the HTTP code (integer). Always 200 for the selenium backend, which has no way of retrieving it.

    """
    if backend == "requests" or url.lower().endswith(".xml") or url.lower().endswith(".txt") or ".pdf" in url.lower():
        page = requests.get(url, timeout=30, headers=custom_header, allow_redirects=True, verify=False)
        url = page.url
        status_code = page.status_code
        content = page.content
        encoding = page.encoding
        apparent_encoding = page.apparent_encoding

    elif backend == "selenium" and driver is not None and wait is not None:
        driver.get(url)

        # Wait for AJAX calls to return
        try:
            wait.until(lambda driver: driver.execute_script('return document.readyState') == 'complete')
        except Exception as e:
            pass

        url = driver.current_url
        content = driver.page_source

        # selenium doesn't handle these, so guess them:
        status_code = 200
        encoding = apparent_encoding = "utf-8"
    else:
        raise(Exception("wrong backend chosen or no driver configured"))

    check_response(url, status_code)

    return content, url, status_code, encoding, apparent_encoding
