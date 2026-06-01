"""Find and remove duplicates and near-duplicates in a list of [core.types.web_page][]

© 2024 - Aurélien Pierre.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta

import requests
import Levenshtein

from curl_cffi import requests
import numpy as np
from guppy import hpy
h=hpy()
import sqlite3
import os
from concurrent import futures
import concurrent.futures
import os

from . import patterns
from . import nlp
from . import database
from .types import web_page, sanitize_web_page
from .utils import guess_date, get_models_folder, timeit

def _normalise_date(d) -> str:
    """Normalise any date value to a comparable ISO-8601 string.

    The list path stores ``datetime`` as Python :class:`datetime.datetime`
    objects set by :func:`core.utils.guess_date`; the DB path reads them back
    from SQLite as plain TEXT.  By converting both to ISO strings before any
    comparison, :func:`_elect_group` works uniformly without caring which path
    called it.

    ``None``, empty strings, and other falsy values map to ``""`` which sorts
    before any real ISO date, treating missing dates as "oldest possible".
    """
    if not d:
        return ""
    if isinstance(d, str):
        return d
    return d.isoformat()   # datetime / date → "YYYY-MM-DDTHH:MM:SS±HH:MM"


from typing import NamedTuple

class _UrlResult(NamedTuple):
    """Return value of :meth:`Deduplicator._canonicalize_url`."""
    canonical_url: str
    domain:        str
    wayback:       "str | None"   # original Wayback/archive URL, or None


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
    def _canonicalize_url(
        cls,
        original_url: str,
        discard_params: bool,
        urls_to_ignore: list,
        fix_urls: bool,
    ) -> "_UrlResult | None":
        """Canonicalize a single URL and extract its domain.

        This is the authoritative URL-processing core, shared by both the list
        path (:meth:`prepare_posts_parallel`) and the DB path
        (:meth:`_fill_prepared`).  It is the **only** place where URL
        normalization rules live; changing the rules here affects both paths.

        Processing steps (in order):

        1. Pre-filter: discard if the original URL matches :attr:`urls_to_ignore`.
        2. Wayback / Internet Archive unwrapping (``web.archive.org`` prefix).
        3. URL structure parsing via :func:`patterns.split_url`.
        4. ``http`` → ``https`` upgrade probe (when ``fix_urls=True`` and the URL
           is not already from a Wayback snapshot).
        5. Wikipedia mobile redirect (``*.m.wikipedia.org`` → ``*.wikipedia.org``).
        6. ``www.`` stripping probe (when ``fix_urls=True``).
        7. Canonical URL assembly (protocol + domain + path), retaining only
           ``?lang=`` / ``?v=`` params and ``#page=N`` anchors.
        8. Post-filter: discard if the **canonical** URL matches
           :attr:`urls_to_ignore` (catches patterns revealed by unwrapping).

        Args:
            original_url:   Raw URL as stored by the crawler.
            discard_params: When ``True``, strip all query parameters except
                            ``?lang=`` and ``?v=`` (YouTube / translated pages).
            urls_to_ignore: Substrings that mark a URL as discardable.
            fix_urls:       Probe for ``https``/non-``www.`` variants via
                            ``HEAD`` requests (I/O-bound; use threads).

        Returns:
            A :class:`_UrlResult` with ``canonical_url``, ``domain``, and
            ``wayback`` (the original archive wrapper URL, or ``None``); or
            ``None`` if the URL must be discarded.
        """
        if cls.discard_post(original_url, urls_to_ignore):
            return None

        input_url = original_url.rstrip("/")

        # Wayback / Internet Archive unwrapping
        wayback = None
        canonical = patterns.wayback_extract_url(input_url)
        if canonical:
            wayback   = input_url   # remember the archive wrapper
            input_url = canonical

        url_parts = patterns.split_url(input_url)
        if not url_parts:
            return None

        protocol, domain, page, params, anchor = url_parts

        # See if an https variant of the http page is available.
        # This avoids http/https duplicates.
        if fix_urls and protocol == "http" and not canonical:
            test_url = "https://" + domain + page + params + anchor
            try:
                response = requests.head(test_url, timeout=2, allow_redirects=True, impersonate="chrome120")
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
            test_url = protocol + "://" + domain.lstrip("www.") + page + params + anchor
            try:
                response = requests.head(test_url, timeout=2, allow_redirects=True, impersonate="chrome120")
                if response.status_code == 200:
                    # Found a valid page -> remove www.
                    domain = domain.lstrip("www.")
            except Exception:
                pass # timeout

        # Canonical URL assembly.
        # Matrix chat uses /#/ as a virtual path; preserve the original form.
        if "/#/" in original_url:
            new_url = original_url
        else:
            new_url = protocol + "://" + domain + page

        if params and (params.startswith("?lang=") or params.startswith("?v=") or not discard_params):
            new_url += params   # translation params and YouTube video IDs must be kept

        if anchor and anchor.startswith("#page="):
            new_url += anchor   # PDF page anchors are semantically significant

        # Post-canonicalization discard check (e.g. /tag/ revealed by Wayback unwrap)
        if cls.discard_post(new_url, urls_to_ignore):
            return None

        return _UrlResult(canonical_url=new_url, domain=domain, wayback=wayback)


    @classmethod
    def prepare_posts_parallel(cls, elem, discard_params, urls_to_ignore, fix_urls):
        """Canonicalize a :class:`~core.types.web_page` dict for the list path.

        Delegates URL normalization to :meth:`_canonicalize_url` and adds
        list-path-specific fallbacks for ``length`` and ``datetime`` (which are
        guaranteed to be pre-computed on the DB path by ``batch_parse_web_page``
        but may be absent on hand-assembled lists).

        Returns the mutated *elem* dict, or ``None`` if the URL must be
        discarded.
        """
        result = cls._canonicalize_url(elem["url"], discard_params, urls_to_ignore, fix_urls)
        if result is None:
            return None

        elem["url"]    = result.canonical_url
        elem["domain"] = result.domain
        if result.wayback is not None:
            elem["wayback"] = result.wayback

        # List-path fallbacks — not needed on the DB path because
        # batch_parse_web_page already populates these columns.
        if not elem.get("length") and elem.get("parsed"):
            elem["length"] = len(elem["parsed"])

        if not elem.get("datetime"):
            elem["datetime"] = guess_date(elem.get("date"))

        return elem
    

    @staticmethod
    def get_unique_urls_parallel(candidates):
        elected = candidates[0]
        category = candidates[0]["category"] if "category" in candidates[0] else ""
        length = 0
        date = datetime.fromtimestamp(0, tz=timezone(timedelta(0)))

        for candidate in candidates:
            cand_date = candidate.get("datetime", guess_date(candidate.get("date")))
            cand_length = candidate["length"]
            cand_category = candidate["category"] if "category" in candidate else ""
            vote = False

            if cand_length > length:
                # Replace by longer content if any
                length = cand_length
                vote = True

            if cand_date and date and cand_date > date:
                # Replace by more recent content if any
                date = cand_date
                vote = True
            elif cand_date and date and cand_date < date:
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

        # Use threads to avoid duplicating the full `posts` list in separate processes.
        # `prepare_posts_parallel` is I/O-heavy (requests.head), so threads are appropriate
        # and they operate on shared memory references without copying `posts`.
        max_workers = os.cpu_count() or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                lambda elem: self.prepare_posts_parallel(elem, self.discard_params, self.urls_to_ignore, self.fix_urls),
                posts,
            )

            for elem in results:
                if elem and "parsed" in elem and len(elem["parsed"]) > 0:
                    # Create a dict where the key is the canonical URL
                    # and we aggregate the list of matching objects sharing the same URL.
                    cleaned_set.setdefault(elem["url"], [])
                    cleaned_set[elem["url"]].append(elem)

        # 3. Extract the best candidate for each canonical URL, aka most recent, or longest, or most accurate
        return [self.get_unique_urls_parallel(item) for item in cleaned_set.values()]


    # ── Election priority constants ───────────────────────────────────────────────────────
    #
    # Two separate ORDER BY clauses because URL conflicts and content conflicts have
    # different tiebreakers (length vs. URL length) as the third criterion.
    #
    # Common prefix for both:
    #   1. Non-external category  — internal crawl data has less noise
    #   2. Newer content datetime — primary quality signal (pre-computed by batch_parse_web_page)
    #   3. Newer crawled datetime — secondary freshness signal (set by the crawler)
    #
    # Then they diverge:
    #   URL  conflict tiebreaker  → longer content   (more information)
    #   Content conflict tiebreaker → shorter URL    (cleaner / more canonical address)
    #
    # Final tiebreaker for both: lower source_rowid (deterministic, reproducible).
    #
    # NULLS LAST: rows lacking a date sort after rows that have one, so a dated
    # row always beats an undated one at the same content/length level.
    # SQLite supports NULLS LAST since 3.30.0 (2019-10-04).

    _ELECTION_ORDER_URL = """
        CASE WHEN category = 'external' THEN 1 ELSE 0 END  ASC,
        datetime                                             DESC NULLS LAST,
        crawled                                              DESC NULLS LAST,
        length                                               DESC NULLS LAST,
        source_rowid                                         ASC
    """

    _ELECTION_ORDER_CONTENT = """
        CASE WHEN category = 'external' THEN 1 ELSE 0 END  ASC,
        datetime                                             DESC NULLS LAST,
        crawled                                              DESC NULLS LAST,
        LENGTH(canonical_url)                                ASC,
        source_rowid                                         ASC
    """

    def run_on_db(self, db: sqlite3.Connection, chunksize: int = 4096) -> None:
        """Deduplicate the ``pages`` table in-place, matching the full ``__call__`` pipeline.

        The method runs four sequential phases that mirror ``__call__``:

        1. **URL canonicalization** – stream every row through
           :meth:`prepare_posts_parallel` (threaded, I/O-bound), normalise URLs,
           compute a SHA-1 content hash, and write results to the temporary
           ``_prepared`` table.
        2. **URL deduplication** – for each canonical URL keep the single best row
           using SQL window functions with :attr:`_ELECTION_ORDER`.
        3. **Exact-content deduplication** – among URL winners, collapse rows that
           share the same SHA-1 hash using the same election order.
        4. **Near-duplicate removal** (skipped when ``threshold == 1.0``) – load the
           survivors into memory, run the Levenshtein window scan with parallelised
           comparisons (threaded; ``python-Levenshtein`` releases the GIL), write
           the final winner set back to a temp table.

        The ``pages`` table is atomically replaced by the winner set at the end.
        All intermediate ``_prepared / _url_winners / _content_winners /
        _near_winners`` temp tables are cleaned up on success.

        Assumptions:
        - ``pages`` has at least the columns: ``url, title, content, date,
          datetime, parsed, category``.
        - ``datetime`` values, when present, are ISO-8601 strings (SQLite TEXT).
          NULL is treated as "oldest possible" in the election.
        - The ``external`` category label means the page was crawled by following
          external links and contains the full ``<body>``; any other category means
          it was crawled from a sitemap / REST-API and contains cleaner markup.
          Non-external therefore wins over external in the election.

        Args:
            db:        Open ``sqlite3.Connection`` to the database.
            chunksize: Number of rows fetched per batch during Phase 1.
        """
        cursor = db.cursor()

        before = cursor.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        print(f"[dedup] Phase 0  – initial records                      : {before}")

        # ── Phase 1: URL canonicalization + hash computation ──────────────────
        self._fill_prepared(db, chunksize)
        after_prep = cursor.execute("SELECT COUNT(*) FROM _prepared").fetchone()[0]
        print(f"[dedup] Phase 1  – after URL canonicalization           : {after_prep}")
        database.compress_db(db)

        # ── Phase 2: URL deduplication ────────────────────────────────────────
        self._elect_by_url(db)
        after_url = cursor.execute("SELECT COUNT(*) FROM _url_winners").fetchone()[0]
        print(f"[dedup] Phase 2  – after URL deduplication              : {after_url}")
        database.compress_db(db)

        # ── Phase 3: Exact content deduplication ──────────────────────────────
        self._elect_by_content(db)
        after_content = cursor.execute("SELECT COUNT(*) FROM _content_winners").fetchone()[0]
        print(f"[dedup] Phase 3  – after exact-content deduplication    : {after_content}")
        database.compress_db(db)

        # ── Phase 4: Near-duplicate removal (optional) ────────────────────────
        if self.threshold < 1.0:
            self._elect_near_duplicates(db)
            final_table = "_near_winners"
            after_near = cursor.execute(f"SELECT COUNT(*) FROM {final_table}").fetchone()[0]
            print(f"[dedup] Phase 4  – after near-duplicate removal         : {after_near}")
            database.compress_db(db)
        else:
            final_table = "_content_winners"
            print("[dedup] Phase 4  – near-duplicate removal skipped (threshold=1.0)")

        # ── Phase 5: Rebuild pages table with winners ─────────────────────────
        self._rebuild_pages(db, final_table)
        database.compress_db(db)

        final = cursor.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        print(f"[dedup] Done     – {before - final} removed, {final} remain.")

    # ─────────────────────────────────────────────────────────────────────────────────────
    # Private DB pipeline helpers
    # ─────────────────────────────────────────────────────────────────────────────────────

    def _fill_prepared(self, db: sqlite3.Connection, chunksize: int) -> None:
        """Phase 1 – canonicalise every URL, extract domains, populate ``_prepared``.

        **What this phase does and does not do**

        *Does:*

        - Run every row's URL through :meth:`_canonicalize_url` (parallel,
          I/O-bound when ``fix_urls=True``): Wayback unwrapping, ``http``→``https``
          upgrade, ``www.`` stripping, ``urls_to_ignore`` filtering.
        - Write the canonical URL, domain, and (if applicable) the original
          Wayback wrapper URL back via ``_prepared``.
        - Copy ``content_hash``, ``datetime``, ``crawled``, ``length``, and
          ``category`` verbatim from ``pages``; these are expected to have been
          populated upstream by ``batch_parse_web_page`` / the crawler.

        *Does not:*

        - Re-compute ``content_hash``, ``length``, or ``datetime`` — those are
          the responsibility of ``batch_parse_web_page``.
        - Store ``parsed`` text — it is large, already in ``pages``, and only
          needed in Phase 4 where it is read back from ``pages`` directly.
        - Drop rows with ``NULL`` / empty ``parsed`` — such rows are kept for
          archival (the URL is known but the content could not be extracted).
          They participate in URL deduplication (Phase 2) but are excluded from
          content and near-duplicate deduplication (Phases 3–4) because their
          ``content_hash`` is ``NULL``.

        **Pre-conditions** (caller's responsibility)

        The ``pages`` table is expected to have the following columns populated
        before :meth:`run_on_db` is called:

        - ``parsed``       — normalised text produced by ``batch_parse_web_page``.
        - ``content_hash`` — SHA-1 of ``parsed``, set by ``batch_parse_web_page``.
        - ``length``       — character count of ``parsed``, set by the same.
        - ``datetime``     — timezone-aware UTC datetime parsed from ``date``,
                             set by ``batch_parse_web_page``.
        - ``crawled``      — UTC datetime set by the crawler at fetch time.
        - ``category``     — arbitrary label set by the crawler; ``"external"``
                             marks pages reached by recursive link-following.

        Columns that are **not** required before deduplication and are typically
        computed after it: ``tokenized``, ``stemmed``, ``vectorized``.

        Column ``excerpt`` should be populated by the crawler or a preprocessing
        step; the deduplicator does not generate it on the DB path.

        Args:
            db:        Open database connection.
            chunksize: Rows fetched per batch; tune to balance memory and throughput.
        """
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS _prepared")
        cursor.execute("""
            CREATE TABLE _prepared (
                source_rowid  INTEGER PRIMARY KEY,
                canonical_url TEXT    NOT NULL,
                domain        TEXT,
                wayback       TEXT,            -- original archive wrapper URL, or NULL
                content_hash  TEXT,            -- NULL for rows with no parsed content
                datetime      TEXT,            -- pre-computed upstream; NULL if unavailable
                crawled       TEXT,            -- set by crawler; NULL if unavailable
                length        INTEGER,         -- NULL for rows with no parsed content
                category      TEXT
            )
        """)
        # Partial index on content_hash: NULL rows are excluded so the index
        # only covers rows that will actually participate in content dedup.
        cursor.execute("CREATE INDEX idx_prep_url  ON _prepared (canonical_url)")
        cursor.execute("CREATE INDEX idx_prep_hash ON _prepared (content_hash) WHERE content_hash IS NOT NULL")
        db.commit()

        sel = db.execute("""
            SELECT rowid, url, content_hash, datetime, crawled, length, category
            FROM pages
        """)

        max_workers = os.cpu_count() or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                rows = sel.fetchmany(chunksize)
                if not rows:
                    break

                # Split URL column from the rest so canonicalization is parallelised
                # without copying heavy content columns into worker closures.
                url_jobs = [(row[0], row[1]) for row in rows]
                metadata = {row[0]: row[2:] for row in rows}   # rowid → (hash, dt, crawled, length, cat)

                url_results = list(executor.map(
                    lambda t: (t[0], self._canonicalize_url(
                        t[1], self.discard_params, self.urls_to_ignore, self.fix_urls
                    )),
                    url_jobs,
                ))

                inserts = []
                for rid, url_res in url_results:
                    if url_res is None:
                        continue    # URL matches urls_to_ignore — drop row
                    content_hash, dt, crawled, length, category = metadata[rid]
                    inserts.append((
                        rid,
                        url_res.canonical_url,
                        url_res.domain,
                        url_res.wayback,
                        content_hash,
                        dt,
                        crawled,
                        length or None,   # treat 0 as NULL (unparsed archival row)
                        category,
                    ))

                if inserts:
                    cursor.executemany(
                        """INSERT INTO _prepared
                               (source_rowid, canonical_url, domain, wayback,
                                content_hash, datetime, crawled, length, category)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        inserts,
                    )
                    db.commit()


    def _elect_by_url(self, db: sqlite3.Connection) -> None:
        """Phase 2 – keep one row per canonical URL using :attr:`_ELECTION_ORDER_URL`.

        Creates the ``_url_winners`` temp table.  Rows with ``NULL``
        ``content_hash`` (no extractable content) participate normally: if two
        such rows share a canonical URL, the one with the more recent
        ``crawled`` timestamp survives; if neither has a crawled timestamp the
        lower ``source_rowid`` wins.
        """
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS _url_winners")
        cursor.execute(f"""
            CREATE TABLE _url_winners AS
            SELECT source_rowid
            FROM (
                SELECT source_rowid,
                       ROW_NUMBER() OVER (
                           PARTITION BY canonical_url
                           ORDER BY {self._ELECTION_ORDER_URL}
                       ) AS rn
                FROM _prepared
            )
            WHERE rn = 1
        """)
        cursor.execute("CREATE INDEX idx_uw ON _url_winners (source_rowid)")
        db.commit()


    def _elect_by_content(self, db: sqlite3.Connection) -> None:
        """Phase 3 – keep one row per SHA-1 content hash using :attr:`_ELECTION_ORDER_CONTENT`.

        Only rows with a non-``NULL`` ``content_hash`` are deduplicated.  Rows
        without a hash (archival stubs whose ``parsed`` content is ``NULL``)
        have already been URL-deduplicated in Phase 2 and are passed through
        unchanged via a ``UNION ALL``.

        Assumption: ``content_hash`` was computed by ``batch_parse_web_page``
        (SHA-1 of normalised ``parsed`` text).  Rows that were never processed
        by that step will have ``content_hash = NULL`` and will be treated as
        archival stubs regardless of whether they have ``parsed`` text.

        Creates the ``_content_winners`` temp table.
        """
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS _content_winners")
        cursor.execute(f"""
            CREATE TABLE _content_winners AS

            -- Rows with content: elect one winner per identical hash
            SELECT source_rowid
            FROM (
                SELECT p.source_rowid,
                       ROW_NUMBER() OVER (
                           PARTITION BY p.content_hash
                           ORDER BY {self._ELECTION_ORDER_CONTENT}
                       ) AS rn
                FROM _prepared    p
                JOIN _url_winners u USING (source_rowid)
                WHERE p.content_hash IS NOT NULL
            )
            WHERE rn = 1

            UNION ALL

            -- Archival stubs (no content hash): pass through without deduplication
            SELECT u.source_rowid
            FROM _url_winners u
            JOIN _prepared    p USING (source_rowid)
            WHERE p.content_hash IS NULL
        """)
        cursor.execute("CREATE INDEX idx_cw ON _content_winners (source_rowid)")
        db.commit()


    def _elect_near_duplicates(self, db: sqlite3.Connection) -> None:
        """Phase 4 – Levenshtein near-duplicate detection on content winners.

        Loads only the rows that have actual content (``pages.parsed IS NOT
        NULL``) sorted by ``canonical_url`` for locality, then delegates the
        parallel window scan to :meth:`_close_content_scan`.

        Rows without parsed content (archival stubs) are never compared —
        they have no text to measure similarity against — and are passed
        straight into ``_near_winners`` via a separate INSERT.

        ``parsed`` text is read directly from ``pages`` rather than being
        cached in ``_prepared``: it can be very large, and ``_prepared`` is
        a compact index-oriented table that should not duplicate bulk text.

        Creates the ``_near_winners`` temp table.
        """
        cursor = db.cursor()

        rows = cursor.execute("""
            SELECT pr.source_rowid,
                   pr.canonical_url,
                   pg.parsed,
                   pr.datetime,
                   pr.crawled,
                   pr.length,
                   pr.category
            FROM _prepared        pr
            JOIN _content_winners cw USING (source_rowid)
            JOIN pages            pg ON pg.rowid = pr.source_rowid
            WHERE pg.parsed IS NOT NULL
              AND pg.parsed != ''
            ORDER BY pr.canonical_url
        """).fetchall()

        cursor.execute("DROP TABLE IF EXISTS _near_winners")
        cursor.execute("CREATE TABLE _near_winners (source_rowid INTEGER PRIMARY KEY)")

        if rows:
            elements = [
                {
                    "rowid":    r[0],
                    "url":      r[1],
                    "parsed":   r[2],
                    "datetime": r[3] or "",
                    "crawled":  r[4] or "",
                    "length":   r[5] or 0,
                    "category": r[6] or "",
                }
                for r in rows
            ]

            keep = self._close_content_scan(elements)
            cursor.executemany(
                "INSERT INTO _near_winners VALUES (?)",
                [(elements[i]["rowid"],) for i in range(len(elements)) if keep[i]],
            )

        # Archival stubs bypass near-duplicate detection entirely
        cursor.execute("""
            INSERT OR IGNORE INTO _near_winners
            SELECT cw.source_rowid
            FROM _content_winners cw
            JOIN pages pg ON pg.rowid = cw.source_rowid
            WHERE pg.parsed IS NULL OR pg.parsed = ''
        """)

        db.commit()


    def _close_content_scan(
        self,
        elements: list[dict],
        threshold: float | None = None,
        distance: int | None = None,
    ) -> np.ndarray:
        """Parallel Levenshtein window scan — shared by both deduplication paths.

        Called by :meth:`get_close_content` (list path) and
        :meth:`_elect_near_duplicates` (DB path) so that near-duplicate
        detection is byte-for-byte identical regardless of how data was loaded.

        The outer loop is inherently sequential: whether row *j* is still a live
        candidate when row *i* is processed depends on the results of earlier
        iterations.  The Levenshtein comparisons *within* each window are
        independent and are dispatched to a
        :class:`~concurrent.futures.ThreadPoolExecutor`.
        ``python-Levenshtein`` (``rapidfuzz`` backend) releases the GIL, so
        the threads are genuinely parallel here.  A ``ProcessPoolExecutor``
        would require pickling every content string per task, which is slower
        for large payloads.

        Args:
            elements:  Dicts each carrying at least ``parsed``, ``datetime``,
                       ``length``, and ``category`` keys, pre-sorted by URL for
                       near-duplicate locality.
            threshold: Levenshtein ratio above which two rows are considered
                       near-duplicates.  Defaults to :attr:`self.threshold`.
            distance:  How many positions ahead to scan from each row.
                       Defaults to :attr:`self.distance`.

        Returns:
            Boolean :class:`numpy.ndarray` of length ``len(elements)``; ``True``
            means the row survives.
        """
        threshold = threshold if threshold is not None else self.threshold
        distance  = distance  if distance  is not None else self.distance

        n    = len(elements)
        keep = np.ones(n, dtype=bool)

        max_workers = os.cpu_count() or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(n):
                if not keep[i]:
                    continue

                live_js = [j for j in range(i + 1, min(n, i + distance)) if keep[j]]
                if not live_js:
                    continue

                parsed_i = elements[i]["parsed"]

                # Dispatch ratio computations for the whole window in parallel.
                ratios = list(executor.map(
                    lambda j: Levenshtein.ratio(parsed_i, elements[j]["parsed"]),
                    live_js,
                ))

                near_js = [j for j, r in zip(live_js, ratios) if r > threshold]
                if not near_js:
                    continue

                group = [i] + near_js
                print(f"[dedup/near] index {i}: {len(near_js)} near-duplicate(s) found")

                best = self._elect_group(elements, group)
                for idx in group:
                    if idx != best:
                        keep[idx] = False

        return keep

    @staticmethod
    def _elect_group(elements: list[dict], indices: list[int]) -> int:
        """Return the index (into *elements*) of the best candidate in *indices*.

        Selects the winner by maximising a priority key that directly encodes
        the election rules, eliminating the fragility of the old sequential
        voting loop:

        Priority (highest first):

        1. **Non-external category** — ``"external"`` rows lose to any other
           category because they contain full ``<body>`` HTML with more noise.
        2. **Newer content datetime** — pre-computed UTC ISO string from
           ``batch_parse_web_page``; ``""`` (no date) sorts last.
        3. **Newer crawled datetime** — set by the crawler; ``""`` sorts last.
        4. **Longer content** — larger ``length`` value wins.
        5. **Lower index** — deterministic tiebreaker (lower index ↔ lower
           ``source_rowid`` for DB path, insertion order for list path).

        ``datetime`` and ``crawled`` values are normalised to ISO-8601 strings
        via :func:`_normalise_date` so the method works whether called from the
        list path (Python :class:`~datetime.datetime` objects) or the DB path
        (SQLite TEXT / empty string).

        Args:
            elements: Full list of row dicts.  Required keys: ``datetime``,
                      ``length``, ``category``.  Optional key: ``crawled``
                      (absent on the list path; treated as ``""``).
            indices:  Indices into *elements* that form the candidate group.

        Returns:
            The index (into *elements*) of the elected winner.
        """
        def _key(idx: int) -> tuple:
            e = elements[idx]
            return (
                # 1. non-external beats external (1 > 0)
                0 if (e.get("category") or "") == "external" else 1,
                # 2. newer content datetime (ISO string: later > earlier > "")
                _normalise_date(e.get("datetime")),
                # 3. newer crawled datetime
                _normalise_date(e.get("crawled")),
                # 4. longer content
                e.get("length") or 0,
                # 5. lower index (negate so max() picks the smallest)
                -idx,
            )

        return max(indices, key=_key)

    def _rebuild_pages(self, db: sqlite3.Connection, winners_table: str) -> None:
        """Replace ``pages`` with the winning rows, writing back canonicalised metadata.

        Columns substituted from ``_prepared`` rather than copied verbatim:

        ``url``
            Canonicalised URL (correct protocol, stripped ``www.``,
            no tracking params).

        ``domain``
            Domain extracted from the canonical URL.  Fixes rows that had their
            full URL stored as domain or a ``NULL`` domain — including archival
            stubs that were previously excluded from ``_prepared`` and therefore
            never had their domain corrected.

        ``wayback``
            Set to the original Wayback/archive wrapper URL when the canonical
            URL was unwrapped from an Internet Archive snapshot; otherwise the
            existing value in ``pages`` is preserved via ``COALESCE``.

        All other columns — including ``content_hash``, ``parsed``, ``length``,
        ``datetime``, ``crawled``, etc. — are copied verbatim from ``pages``;
        they were pre-computed upstream and must not be altered here.

        Args:
            db:            Open database connection.
            winners_table: Temp table whose ``source_rowid`` column identifies
                           the rows to keep.

        Raises:
            RuntimeError: If the ``pages`` table cannot be found in
                          ``sqlite_master``.
        """
        cursor = db.cursor()

        schema_row = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='pages'"
        ).fetchone()
        if schema_row is None:
            raise RuntimeError("pages table not found in sqlite_master")

        create_sql = schema_row[0].replace("CREATE TABLE pages", "CREATE TABLE pages_new")
        create_sql = create_sql.replace('CREATE TABLE "pages"', "CREATE TABLE pages_new")
        cursor.execute("DROP TABLE IF EXISTS pages_new")
        cursor.execute(create_sql)

        cols     = [row[1] for row in cursor.execute("PRAGMA table_info(pages)")]
        cols_set = set(cols)

        select_parts = []
        for col in cols:
            if col == "url":
                select_parts.append("pr.canonical_url")
            elif col == "domain":
                select_parts.append("pr.domain")
            elif col == "wayback" and "wayback" in cols_set:
                # Keep existing wayback value unless _prepared found a new one
                select_parts.append('COALESCE(pr.wayback, p."wayback")')
            else:
                select_parts.append(f'p."{col}"')

        col_list   = ", ".join(f'"{c}"' for c in cols)
        select_sql = ", ".join(select_parts)

        cursor.execute(f"""
            INSERT INTO pages_new ({col_list})
            SELECT {select_sql}
            FROM pages       p
            JOIN _prepared   pr ON pr.source_rowid = p.rowid
            WHERE p.rowid IN (SELECT source_rowid FROM {winners_table})
        """)
        db.commit()

        cursor.execute("BEGIN")
        cursor.execute("DROP TABLE pages")
        cursor.execute("ALTER TABLE pages_new RENAME TO pages")
        cursor.execute("COMMIT")

        for t in ("_prepared", "_url_winners", "_content_winners", "_near_winners"):
            cursor.execute(f"DROP TABLE IF EXISTS {t}")
        db.commit()

        database.compress_db(db)
        print("[dedup] pages table rebuilt.")


    @staticmethod
    def add_content_hash_column(db: sqlite3.Connection) -> None:
        """Add (or refresh) a ``content_hash`` column on the ``pages`` table.

        Computes a SHA-1 digest of each row's ``parsed`` field and stores it in
        ``content_hash``.  The column is created if it does not yet exist.  Rows
        with a NULL ``parsed`` value are skipped and left with a NULL hash.

        A covering index ``idx_pages_content_hash`` is created (or left in place)
        after the update so that subsequent deduplication queries are cheap.

        This method is a standalone maintenance utility.  The deduplication
        pipeline (:meth:`run_on_db`) computes hashes inline during Phase 1 and
        does **not** require this method to be called first.

        Assumption: ``parsed`` values fit in memory individually (they are fetched
        one batch at a time, not all at once).

        Args:
            db: Open ``sqlite3.Connection`` to the target database.
        """
        import hashlib

        cursor = db.cursor()

        columns = {row[1] for row in cursor.execute("PRAGMA table_info(pages)")}
        if "content_hash" not in columns:
            cursor.execute("ALTER TABLE pages ADD COLUMN content_hash TEXT")

        rows = cursor.execute("SELECT rowid, parsed FROM pages WHERE parsed IS NOT NULL")
        updates: list[tuple[str, int]] = []

        for rowid, content in rows:
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
            updates.append((digest, rowid))

            if len(updates) >= 1024:
                cursor.executemany(
                    "UPDATE pages SET content_hash = ? WHERE rowid = ?", updates
                )
                db.commit()
                updates.clear()

        if updates:
            cursor.executemany(
                "UPDATE pages SET content_hash = ? WHERE rowid = ?", updates
            )

        db.commit()

        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pages_content_hash ON pages (content_hash)"
        )
        db.commit()


    @staticmethod
    def get_unique_content_parallel(candidates):
        elected = candidates[0]
        date = datetime.fromtimestamp(0, tz=timezone(timedelta(0)))

        for candidate in candidates:
            cand_date = candidate.get("datetime", guess_date(candidate["date"]))
            if cand_date and cand_date > date:
                # Replace by more recent content if any
                date = cand_date
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


    def get_close_content(
        self,
        posts: list[web_page],
        threshold: float = 0.90,
        distance: int = 50,
    ) -> list[web_page]:
        """Find and remove near-duplicates using the Levenshtein ratio.

        Delegates the actual scan to :meth:`_close_content_scan`, which
        parallelises comparisons within each window via a
        :class:`~concurrent.futures.ThreadPoolExecutor`.  This method is the
        list-path counterpart to :meth:`_elect_near_duplicates`; both call the
        same shared scan implementation.

        The election among near-duplicate candidates honours the same priority
        rules as URL and content deduplication (non-external > newer > longer >
        shorter URL) via :meth:`_elect_group`.

        Args:
            posts:     List of :class:`core.types.web_page` dicts after URL and
                       exact-content deduplication.
            threshold: Minimum Levenshtein ratio for two pages to be considered
                       near-duplicates.  Defaults to :attr:`self.threshold`.
            distance:  Positions ahead to scan from each row after sorting by URL.
                       Defaults to :attr:`self.distance`.

        Returns:
            Filtered list with near-duplicates removed; one survivor per group.
        """
        # Sort by URL: near-duplicates are most likely on the same domain/path.
        # URL dedup upstream ensures keys are already unique here.
        elements = list(dict(sorted({p["url"]: p for p in posts}.items())).values())
        keep = self._close_content_scan(elements, threshold=threshold, distance=distance)
        return [elements[i] for i in range(len(elements)) if keep[i]]


    def run_on_list(self, posts: list[web_page]) -> list[web_page]:
        """Deduplicate an in-memory list of web pages, matching the full pipeline.

        This is the list-based counterpart to :meth:`run_on_db`.  The two methods
        are kept symmetrical: both run the same four phases (URL canonicalization,
        exact-URL deduplication, exact-content deduplication, optional
        near-duplicate removal) and honour the same election rules.

        Note:
            ``posts`` is consumed and partially destroyed during processing to
            avoid keeping two copies in memory simultaneously.

        Args:
            posts: Flat list of :class:`~core.types.web_page` dicts.  The list
                   is modified in-place; callers should not rely on its contents
                   after this call returns.

        Returns:
            Deduplicated list of sanitised :class:`~core.types.web_page` dicts,
            ready for downstream use.  Also writes a ``domains`` frequency file
            via [core.utils.get_models_folder][].
        """
        before = len(posts)
        print(f"[dedup] Phase 0  – initial records                      : {before}")

        # Phase 1+2: URL canonicalization + URL deduplication (combined in get_unique_urls)
        posts = self.get_unique_urls(posts)
        print(f"[dedup] Phase 1+2 – after URL canonicalization + dedup  : {len(posts)}")

        # Phase 3: Exact-content deduplication
        posts = self.get_unique_content(posts)
        print(f"[dedup] Phase 3  – after exact-content deduplication    : {len(posts)}")

        # Phase 4: Near-duplicate removal (optional)
        if self.threshold < 1.0:
            posts = self.get_close_content(posts, threshold=self.threshold, distance=self.distance)
            print(f"[dedup] Phase 4  – after near-duplicate removal         : {len(posts)}")
        else:
            print("[dedup] Phase 4  – near-duplicate removal skipped (threshold=1.0)")

        # Defensive guard: urls_to_ignore filtering happens in prepare_posts_parallel
        # on both the original and canonical URL, but apply a final pass here to
        # catch any edge case where canonicalization produced a newly-ignorable path.
        posts = [p for p in posts if not self.discard_post(p["url"], self.urls_to_ignore)]

        # List all unique domains with their frequency
        counts = Counter([post["domain"] for post in posts])
        print(f"[dedup] Done     – {before - len(posts)} removed, {len(posts)} remain. ({len(counts)} unique domains)")

        # Sort domains by frequency
        counts = dict(sorted(counts.items(), key=lambda item: item[1]))

        # Remove domains below page number threshold
        discard_list: list[str] = []
        if self.n_min > 0:
            discard_list = [domain for domain, count in counts.items() if count < self.n_min]
            posts = [item for item in posts if item["domain"] not in discard_list]

        with open(get_models_folder("domains"), "w", encoding="utf8") as f:
            for key, value in counts.items():
                if key not in discard_list:
                    f.write(f"{key}: {value}\n")

        return [sanitize_web_page(post) for post in posts]

    @timeit()
    def __call__(
        self,
        posts: list[web_page] | sqlite3.Connection,
        chunksize: int = 4096,
    ) -> list[web_page] | None:
        """Dispatch to :meth:`run_on_list` or :meth:`run_on_db` based on input type.

        This is the single public entry-point for deduplication.  Passing a list
        runs the pure-Python pipeline and returns the deduplicated list; passing
        an :class:`sqlite3.Connection` runs the SQL pipeline in-place and returns
        ``None``.

        Args:
            posts:     Either a flat list of :class:`core.types.web_page` dicts
                       **or** an open :class:`sqlite3.Connection` whose ``pages``
                       table should be deduplicated in-place.
            chunksize: Row-fetch batch size forwarded to :meth:`run_on_db` when
                       ``posts`` is a database connection.  Ignored for lists.

        Returns:
            Deduplicated list of :class:`core.types.web_page` when given a list;
            ``None`` when given a database connection (mutations are applied
            directly to the DB).

        Raises:
            TypeError: If ``posts`` is neither a list nor a
                       :class:`sqlite3.Connection`.
        """
        if isinstance(posts, sqlite3.Connection):
            self.run_on_db(posts, chunksize=chunksize)
            return None
        elif isinstance(posts, list):
            return self.run_on_list(posts)
        else:
            raise TypeError(
                f"Expected a list[web_page] or sqlite3.Connection, got {type(posts).__name__!r}"
            )


    def __init__(self, threshold: float = 0.9, distance: int = 50, discard_params: bool = True, n_min: int = 0, fix_urls: bool = True):
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
                [core.types.web_page][] list has been ordered alphabetically by URL, for performance, assuming near-duplicates
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