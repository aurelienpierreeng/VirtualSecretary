#import requests
from curl_cffi import requests
from curl_cffi import CurlHttpVersion
from curl_cffi.curl import CurlOpt

import time
from datetime import datetime
import random
import httpx

import regex as re
from dataclasses import dataclass

from protego import Protego

from . import utils

# disable warnings regarding bypass of checks for SSL certs
# assuming we only download public pages here
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from . import patterns

from abc import ABC, abstractmethod

class DelayedClass(ABC):
    """Abstract class for any implementation
    having an internal timer that won't trigger a given action more often
    than `delay` seconds
    """

    delay: float = 1.0
    """Timeout between two requests"""

    main_domain: str | None = None
    """Main domain from where we crawl, either the one holding the sitemap or the one at the root of the recursion."""

    robots_txt: Protego | None = None
    """robots.txt file associated to the main_domain"""

    last_requests: dict[str, float] = {}
    """Dictionnary of domain/timestamp for the last request sent to a domain"""

    domain_thresholds: dict[str, float] = {}
    """Dictionnary of known domains remembering the robots.txt rate thresholds"""

    def sleep(self, domain: str, overwrite: float | None = None):
        """Sleep for at most the remaining timeout time.

        Args:
            overwrite: temporary timeout overwrite
        """

        time_elapsed = 9999999999999999999999999
        if domain in self.last_requests:
            time_elapsed = datetime.now().timestamp() - self.last_requests[domain]

        threshold = overwrite or self.delay
        if domain in self.domain_thresholds:
            threshold = self.domain_thresholds[domain]

        if time_elapsed < threshold:
            time.sleep(threshold - time_elapsed)

        self.last_requests[domain] = datetime.now().timestamp()


    def get_robots_txt(self, protocol: str, domain: str, timeout: float) -> Protego | None:
        robots_txt = protocol + "://" + domain.rstrip("/") + "/robots.txt"

        try:
            self.sleep(domain)
            robots = get_page(robots_txt, timeout, HEADER)
            return Protego.parse(robots.text)
        except Exception as e:
            print("Error fetching robots.txt", e)
            return None
        

    def can_crawl(self, url: str, robots: Protego | None) -> bool:
        if not robots:
            return True
        else:
            return robots.can_fetch(url, USER_AGENT)


    def get_crawling_rate(self, domain: str, robots: Protego | None) -> float:
        if not robots or domain == "":
            return self.delay
        
        if domain in self.domain_thresholds:
            return self.domain_thresholds[domain]
        
        crawling_delay = robots.crawl_delay(USER_AGENT) or self.delay
        
        rate = robots.request_rate(USER_AGENT)
        if rate and rate.requests and rate.seconds:
            crawling_delay = max(crawling_delay, rate.requests / rate.seconds)

        # Sanitize delay to something reasonable to prevent assholes servers
        # from blocking the whole crawling forever. 
        # If you can't serve 2 requests per minute, either don't bother making a website
        # or disallow your website fair and square.
        # Fuck inkscape.org and its 1 request / 86400.0 s
        crawling_delay = min(crawling_delay, 30)

        self.domain_thresholds["domain"] = crawling_delay
        return crawling_delay
    

    def __init__(self, protocol:str, domain: str, delay: float, timeout: float):
        self.delay = delay
        self.main_domain = domain
        self.robots_txt = self.get_robots_txt(protocol, domain, timeout)
        self.delay = self.get_crawling_rate(domain, self.robots_txt)
        if self.robots_txt:
            print(f"Main robots.txt parsed for {domain}. Crawling rate: 1 request / {self.delay} s")


@dataclass
class HTTPResponse:
    url: str
    """Redirected URL, if redirection, else initial URL"""

    status_code: int
    """HTTP response return code"""

    headers: dict
    """Server response HTTP headers"""

    content: bytes | None
    """Page content as bytes"""

    encoding: str

    apparent_encoding: str

    text: str
    """Page content as text, if possible"""

    raw_response: object
    """Original curl_cffi or httpx Response object"""


def wrap_response(r: requests.Response | httpx.Response) -> HTTPResponse:
    """Unify responses from HTTPx and cURL into uniform types"""
    return HTTPResponse(
        url=str(r.url),
        status_code=r.status_code,
        headers=dict(r.headers),
        encoding=r.encoding or "utf-8",
        apparent_encoding=r.encoding or "utf-8",
        content=r.content,
        text=getattr(r, "text", ""),
        raw_response=r,
    )


def check_response(prefix: str, old_url: str, new_url: str, status_code):
    print(f"{prefix}: {old_url} -> {new_url} : {status_code}")


USER_AGENT = "VirtualSecretary/1.0 (+https://github.com/aurelienpierreeng/VirtualSecretary)"

HEADER = {
    "User-Agent": USER_AGENT, 
    "Accept": "application/pdf,text/html,application/xhtml+xml,text/xml,application/xml"
}

def _curl_request(method, url, timeout, headers=None) -> requests.Response:
    kwargs = {
        "timeout": (5, timeout), # connect_timiout, and real timeout
        "allow_redirects": True,
        "verify": False,
        "curl_options": {
            CurlOpt.IPRESOLVE: 1,
            CurlOpt.LOW_SPEED_LIMIT: 0,
            CurlOpt.LOW_SPEED_TIME: 0,
        },
    }

    if headers:
        kwargs["headers"] = headers
        kwargs["http_version"] = CurlHttpVersion.V1_1
    else:
        kwargs["impersonate"] = "chrome146"

    return requests.request(method, url, **kwargs)


def _httpx_request(method, url, timeout, headers=None) -> httpx.Response:
    headers = headers or {"User-Agent": USER_AGENT}
    kwargs = {
        "timeout": httpx.Timeout(timeout),
        "follow_redirects": True,
        "verify": False,
    }

    if headers:
        kwargs["http2"] = False

    with httpx.Client(**kwargs) as client:
        return client.request(method, url, headers=headers)


def request(method, url, timeout=30, headers=None) -> HTTPResponse:
    """
    Try curl_cffi first, fallback to httpx on ANY transport-level failure.
    """

    try:
        return wrap_response(_curl_request(method, url, timeout, headers))

    except Exception as e:
        # Optional: log curl failure reason
        print(f"[curl failed] {url}: {e}")

        try:
            return wrap_response(_httpx_request(method, url, timeout, headers))

        except Exception as e2:
            print(f"[httpx failed] {url}: {e2}")
            return HTTPResponse(
                url=url,
                status_code=-1,
                headers=headers or {},
                encoding="utf-8",
                apparent_encoding="utf-8",
                content=None,
                text="",
                raw_response=None,
            )


def get_head(url, timeout, headers=None) -> HTTPResponse:
    return request("HEAD", url, timeout, headers)


def get_page(url, timeout, headers=None) -> HTTPResponse:
    return request("GET", url, timeout, headers)



@utils.exit_after(120)
def try_url(url, delay: DelayedClass, timeout: int | float = 30, bypass_robots_txt: bool = False) -> tuple[HTTPResponse | None, dict | None, str]:
    """Probe the URL head, without getting the content.

    This will:
        1. resolve redirections
        2. check with robots.txt (if any) if we have permission to crawl and at what rate,
        3. fallback to web.archive.org if hitting 404 (not found) error,
        4. handle requests rate thresholding,
        5. find out what headers spoofing combination is accepted by the server (or by fucking Cloudflare),
        when robots.txt didn't block us explicitely, but server/proxy returned 403 (unauthorized) error.

    Args:
        delay: class holding a thresholding timer/delay method,
        timeout: 
            abort any connection that takes longer than this (in seconds) to finish. That might cancel loading 
            large PDFs if too small.
        bypass_robots_txt:
            don't check if robots.txt allows us to crawl the current page. That makes us spare some requests.
            When crawling pages from `sitemap.xml`, we can safely assume that all pages there are allowed or the webmaster
            is an idiot.

    Returns:
        response: the HTTP response object,
        headers: the HTTP client headers that succeeded in spoofing the server, if any,
        url: the final, redirected, URL (can be the same as the input one).
    """

    # URL inside non-processed Markdown syntax can have an orphan/unmatched trailing )
    # But beware of Wikipedia links that can have non-orphan final ) for disambiguation
    url = patterns.remove_unmatched_parentheses(url).rstrip("/.,: ")

    # Note: web.archive.org doesn't have a robots.txt, but will threshold us anyway.
    # Be nice with them.
    crawling_delay = delay.delay
    if "web.archive.org" in url:
        bypass_robots_txt = True
        crawling_delay = 2.0

    # Start with resolving redirections, for doi.org or URL shorteners,
    # because we need the target domain to get the proper robots.txt later
    valid_url = False
    result = None

    link = patterns.split_url(url)
    domain = ""
    if link is not None:
        protocol, domain, page, params, anchor = link

    try:
        delay.sleep(domain, crawling_delay)
        result = get_head(url, timeout, HEADER)
        url = str(result.url)
        valid_url = result.status_code > -1 and result.status_code < 400
    except Exception as e:
        print(f"{url} failed: {e}")

    # Parse (redirected ?) URL
    link = patterns.split_url(url)
    wayback = "https://web.archive.org/web"
    archive_url = f"{wayback}/0/{url}"
    is_archive_url = url.startswith(wayback)

    if link is not None:
        protocol, domain, page, params, anchor = link
    else:
        return None, None, url
    
    # 0: try to see if there is a robots.txt file preventing us from accessing
    allowed = True

    if bypass_robots_txt and valid_url and result:
        # If we bypass robots and already got a valid URL at the first head ping, 
        # return immediately: we have everything we need already.
        return result, HEADER, url

    elif not bypass_robots_txt:
        robots_txt = None
        try:
            if delay.main_domain == domain and delay.robots_txt:
                robots_txt = delay.robots_txt
            else:
                robots_txt = delay.get_robots_txt(protocol, domain, timeout)
                crawling_delay = max(delay.get_crawling_rate(domain, robots_txt), crawling_delay)
                print(f"robots.txt parsed for {domain}. Crawling rate: 1 request / {crawling_delay} s")

            allowed = delay.can_crawl(url, robots_txt)

        except Exception as e:
            print(f"{domain}/robots.txt failed: {e}")

        if not allowed:
            print(f"robots.txt forbids us to crawl {url}")
            return None, None, url

    
    # 1: try the given URL
    # At this point, robots.txt gave explicit authorization to crawl to our user agent.
    # Yet some servers will still block us (403 error), so enable spoofing if they do.
    for test_url in [url, url + "/"]:
        try:
            for header in [None, HEADER]:
                delay.sleep(domain, crawling_delay)
                result = get_head(test_url, timeout, header)
                if result.status_code > -1 and result.status_code < 400:
                    # Assuming the next request is to get the full page content,
                    # and we don't keep the domain-wise crawling rates in memory, 
                    # except for the main website recursion,
                    # handle rate thresholding now, so page content runs immediately on call.
                    delay.sleep(domain, crawling_delay)

                    return result, header, str(result.url)
                
        except Exception as e:
            # DNS resolution issue, timeout, server unreachable, etc.
            print(test_url, e)  

            # Try the Wayback machine in final resort: "https://web.archive.org/web/"
            if is_archive_url:
                return None, None, url
            else:
                return try_url(archive_url, delay, timeout=timeout)
                    
    # 2. bruteforce all possible unique combinations in case we have a semi-wrong URL
    # Try : 
    # 1. http/https/no protocol, 
    # 2. with www./without www., trailing / or not, 
    # 3. URL params or not
    # 4. anchors or not.
    # Also try to remove trailing () because that might be Markdown link syntax caught in URL detection.
    for PROTOCOL in set([protocol + "://", "http://", "https://"]):
        for DOMAIN in set([domain, patterns.remove_www(domain), domain.rstrip("/")]):
            for PAGE in set([page, page.rstrip("()"), page.rstrip("/"), page + "/" ]):
                for PARAMS in set([params, "", params.rstrip("()")]):
                    for ANCHOR in set([anchor, "", anchor.rstrip("()")]):
                        test_url = PROTOCOL + DOMAIN + PAGE + PARAMS + ANCHOR
                        if test_url != url and test_url != url + "/":
                            try:
                                for header in [None, HEADER]:
                                    delay.sleep(domain, crawling_delay)
                                    result = get_head(test_url, timeout, header)
                                    if result.status_code > -1 and result.status_code < 400:
                                        # Assuming the next request is to get the full page content,
                                        # and we don't keep the domain-wise crawling rates in memory, 
                                        # except for the main website recursion,
                                        # handle rate thresholding now, so page content runs immediately on call.
                                        delay.sleep(domain, crawling_delay)

                                        return result, header, str(result.url)
                                    
                            except Exception as e:
                                # DNS resolution issue, timeout, server unreachable, etc.
                                print(test_url, e)
                                break


    # Try the Wayback machine in final resort: "https://web.archive.org/web/"
    if is_archive_url:
        return None, None, url
    else:
        return try_url(archive_url, delay, timeout=timeout)


def get_url(url: str, 
            delay: DelayedClass, 
            timeout=60, 
            custom_header={}) -> tuple[bytes | None, str, int, str, str]:
    """
    Get the content of an URL using requests.
    `.pdf`, `.xml` and `.txt` URLs always use requests.

    Return:
        content: the raw DOM,
        url: the final URL after possible redirections
        status: the HTTP code (integer)
        encoding: 
        apparent encoding: 
    """

    page = get_page(url, timeout, custom_header)
    new_url = page.url
    status_code = page.status_code
    content = page.content
    encoding = page.encoding
    apparent_encoding = page.apparent_encoding
    check_response("Content", url, new_url, status_code)
    return content, new_url, status_code, encoding, apparent_encoding
