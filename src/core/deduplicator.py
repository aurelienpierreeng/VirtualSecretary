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
        if cls.discard_post(elem["url"], urls_to_ignore):
            return None
        
        input_url = elem["url"].rstrip("/")

        # Check if this is a Web archive URL
        canonical = patterns.wayback_extract_url(input_url)
        if canonical:
            elem["wayback"] = input_url
            input_url = canonical
        
        # Parse the actual URL
        url = patterns.split_url(input_url)
        if not url:
            return None
        
        protocol, domain, page, params, anchor = url

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
            test_url = protocol + domain.lstrip("www.") + page + params + anchor
            try:
                response = requests.head(test_url, timeout=2, allow_redirects=True, impersonate="chrome120")
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
        # with a text normalization implemented by user

        if "length" not in elem or elem["length"] is None or elem["length"] == 0:
            elem["length"] = len(elem["parsed"])

        # Get datetime for age comparison
        if "datetime" not in elem or elem["datetime"] is None:
            elem["datetime"] = guess_date(elem["date"])

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


    # ── Election priority (used by both URL and content SQL window functions) ──────────────
    #
    # The ordering mirrors the voting logic in `get_unique_urls_parallel`:
    #   1. Non-external category beats external  (internal crawl data has less noise)
    #   2. Newer datetime beats older             (primary quality signal)
    #   3. Longer content beats shorter           (secondary; overridden by age)
    #   4. Shorter canonical URL beats longer     (anchors / tracking params are noise)
    #   5. Lower source_rowid                     (deterministic tiebreaker)
    #
    # NULLS LAST on datetime: rows without a date are treated as oldest.
    # SQLite supports NULLS LAST since 3.30.0 (2019-10-04).
    _ELECTION_ORDER = """
        CASE WHEN category = 'external' THEN 1 ELSE 0 END ASC,
        datetime                                            DESC NULLS LAST,
        length                                              DESC,
        LENGTH(canonical_url)                               ASC,
        source_rowid                                        ASC
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
        print(f"[dedup] Phase 0 – initial rows : {before}")

        # ── Phase 1: URL canonicalization + hash computation ──────────────────
        self._fill_prepared(db, chunksize)
        after_prep = cursor.execute("SELECT COUNT(*) FROM _prepared").fetchone()[0]
        print(f"[dedup] Phase 1 – after URL filtering/canonicalization : {after_prep}")
        database.compress_db(db)

        # ── Phase 2: URL deduplication ────────────────────────────────────────
        self._elect_by_url(db)
        after_url = cursor.execute("SELECT COUNT(*) FROM _url_winners").fetchone()[0]
        print(f"[dedup] Phase 2 – after URL deduplication             : {after_url}")
        database.compress_db(db)

        # ── Phase 3: Exact content deduplication ──────────────────────────────
        self._elect_by_content(db)
        after_content = cursor.execute("SELECT COUNT(*) FROM _content_winners").fetchone()[0]
        print(f"[dedup] Phase 3 – after exact-content deduplication   : {after_content}")
        database.compress_db(db)

        # ── Phase 4: Near-duplicate removal (optional) ────────────────────────
        if self.threshold < 1.0:
            self._elect_near_duplicates(db)
            final_table = "_near_winners"
            after_near = cursor.execute(f"SELECT COUNT(*) FROM {final_table}").fetchone()[0]
            print(f"[dedup] Phase 4 – after near-duplicate removal        : {after_near}")
            database.compress_db(db)
        else:
            final_table = "_content_winners"
            print("[dedup] Phase 4 – near-duplicate removal skipped (threshold=1.0)")

        # ── Phase 5: Rebuild pages table with winners ─────────────────────────
        self._rebuild_pages(db, final_table)
        database.compress_db(db)

        final = cursor.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        print(f"[dedup] Done – removed {before - final} rows, {final} remain.")

    # ─────────────────────────────────────────────────────────────────────────────────────
    # Private DB pipeline helpers
    # ─────────────────────────────────────────────────────────────────────────────────────

    def _fill_prepared(self, db: sqlite3.Connection, chunksize: int) -> None:
        """Phase 1 – stream ``pages``, canonicalise URLs, compute content hashes.

        Rows that are filtered out by :attr:`urls_to_ignore` or that carry an
        empty ``parsed`` field are silently dropped; they will not appear in
        ``_prepared`` and will therefore be absent from the final table.

        The ``content_hash`` column is computed here (SHA-1 of ``parsed``) rather
        than in a separate pass so that we only read each row's content once.

        Threading note: :meth:`prepare_posts_parallel` is I/O-bound when
        ``fix_urls=True`` (it issues ``HEAD`` requests).  A
        :class:`~concurrent.futures.ThreadPoolExecutor` is therefore correct; a
        ``ProcessPoolExecutor`` would needlessly serialise/deserialise the rows.

        Args:
            db:        Open database connection.
            chunksize: Rows fetched per batch.
        """
        import hashlib

        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS _prepared")
        cursor.execute("""
            CREATE TABLE _prepared (
                source_rowid  INTEGER PRIMARY KEY,
                canonical_url TEXT    NOT NULL,
                domain        TEXT,
                parsed        TEXT    NOT NULL,
                content_hash  TEXT    NOT NULL,
                datetime      TEXT,
                length        INTEGER NOT NULL,
                category      TEXT
            )
        """)
        # Indexes created upfront so bulk inserts land with sorted access later.
        cursor.execute("CREATE INDEX idx_prep_url  ON _prepared (canonical_url)")
        cursor.execute("CREATE INDEX idx_prep_hash ON _prepared (content_hash)")
        db.commit()

        sel = db.execute(
            "SELECT rowid, url, title, content, date, datetime, parsed, category FROM pages"
        )

        max_workers = os.cpu_count() or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            while True:
                rows = sel.fetchmany(chunksize)
                if not rows:
                    break

                elems, ids = [], []
                for row in rows:
                    ids.append(row[0])
                    elems.append({
                        "url":      row[1],
                        "title":    row[2],
                        "content":  row[3],
                        "date":     row[4],
                        "datetime": row[5],
                        "parsed":   row[6],
                        "category": row[7],
                    })

                results = list(executor.map(
                    lambda e: self.prepare_posts_parallel(
                        e, self.discard_params, self.urls_to_ignore, self.fix_urls
                    ),
                    elems,
                ))

                inserts = []
                for rid, res in zip(ids, results):
                    if not res or not res.get("parsed"):
                        continue
                    parsed = res["parsed"]
                    inserts.append((
                        rid,
                        res["url"],
                        res.get("domain"),
                        parsed,
                        hashlib.sha1(parsed.encode("utf-8")).hexdigest(),
                        res.get("datetime"),
                        res.get("length") or len(parsed),
                        res.get("category"),
                    ))

                if inserts:
                    cursor.executemany(
                        """INSERT INTO _prepared
                               (source_rowid, canonical_url, domain, parsed,
                                content_hash, datetime, length, category)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        inserts,
                    )
                    db.commit()


    def _elect_by_url(self, db: sqlite3.Connection) -> None:
        """Phase 2 – keep one row per canonical URL using :attr:`_ELECTION_ORDER`.

        Creates the ``_url_winners`` temp table containing only the
        ``source_rowid`` values of the elected rows.
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
                           ORDER BY {self._ELECTION_ORDER}
                       ) AS rn
                FROM _prepared
            )
            WHERE rn = 1
        """)
        cursor.execute("CREATE INDEX idx_uw ON _url_winners (source_rowid)")
        db.commit()


    def _elect_by_content(self, db: sqlite3.Connection) -> None:
        """Phase 3 – among URL winners, keep one row per SHA-1 content hash.

        Operates exclusively on the ``_url_winners`` set so that Phase 2's work
        is not undone: two URLs that canonicalise differently but carry identical
        content will be collapsed here, and the elected row still follows
        :attr:`_ELECTION_ORDER`.

        Creates the ``_content_winners`` temp table.

        Assumption: two rows with different canonical URLs but identical ``parsed``
        text are true duplicates (e.g. HTTP vs HTTPS, trailing-slash variants that
        survived canonicalization, or pages whose content is query-parameter
        independent despite distinct URLs).
        """
        cursor = db.cursor()
        cursor.execute("DROP TABLE IF EXISTS _content_winners")
        cursor.execute(f"""
            CREATE TABLE _content_winners AS
            SELECT source_rowid
            FROM (
                SELECT p.source_rowid,
                       ROW_NUMBER() OVER (
                           PARTITION BY p.content_hash
                           ORDER BY {self._ELECTION_ORDER}
                       ) AS rn
                FROM _prepared      p
                JOIN _url_winners   u USING (source_rowid)
            )
            WHERE rn = 1
        """)
        cursor.execute("CREATE INDEX idx_cw ON _content_winners (source_rowid)")
        db.commit()


    def _elect_near_duplicates(self, db: sqlite3.Connection) -> None:
        """Phase 4 – Levenshtein near-duplicate detection on content winners.

        Algorithm mirrors :meth:`get_close_content`:

        * Load the surviving rows sorted by ``canonical_url`` (near-duplicates are
          more likely to share a URL prefix).
        * For each row *i* not yet eliminated, compare it against the next
          :attr:`distance` live rows using the Levenshtein ratio.
        * Matches above :attr:`threshold` form a group; :meth:`_elect_group` picks
          the winner; the rest are marked for removal.

        Parallelisation: the Levenshtein comparisons *within each window* are
        dispatched to a :class:`~concurrent.futures.ThreadPoolExecutor`.
        ``python-Levenshtein`` (backed by ``rapidfuzz``) releases the GIL for
        string comparison, making threads genuinely parallel here despite Python's
        GIL.  A ``ProcessPoolExecutor`` would require pickling every content string,
        which would be slower for large payloads.

        Creates the ``_near_winners`` temp table.

        Args:
            db: Open database connection.
        """
        cursor = db.cursor()

        rows = cursor.execute("""
            SELECT p.source_rowid, p.canonical_url, p.parsed,
                   p.datetime, p.length, p.category
            FROM _prepared       p
            JOIN _content_winners c USING (source_rowid)
            ORDER BY p.canonical_url
        """).fetchall()

        if not rows:
            cursor.execute("DROP TABLE IF EXISTS _near_winners")
            cursor.execute(
                "CREATE TABLE _near_winners AS SELECT source_rowid FROM _content_winners"
            )
            db.commit()
            return

        elements = [
            {
                "rowid":    r[0],
                "url":      r[1],
                "parsed":   r[2],
                "datetime": r[3] or "",   # empty string sorts before any ISO date
                "length":   r[4] or 0,
                "category": r[5] or "",
            }
            for r in rows
        ]

        n    = len(elements)
        keep = np.ones(n, dtype=bool)

        max_workers = os.cpu_count() or 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(n):
                if not keep[i]:
                    continue

                live_js = [j for j in range(i + 1, min(n, i + self.distance)) if keep[j]]
                if not live_js:
                    continue

                parsed_i = elements[i]["parsed"]

                # Compute Levenshtein ratios for the entire window in parallel.
                ratios = list(executor.map(
                    lambda j: Levenshtein.ratio(parsed_i, elements[j]["parsed"]),
                    live_js,
                ))

                near_js = [j for j, r in zip(live_js, ratios) if r > self.threshold]
                if not near_js:
                    continue

                group = [i] + near_js
                print(f"[dedup/near] index {i}: {len(near_js)} near-duplicate(s) found")

                best = self._elect_group(elements, group)
                for idx in group:
                    if idx != best:
                        keep[idx] = False

        winner_rowids = [elements[i]["rowid"] for i in range(n) if keep[i]]

        cursor.execute("DROP TABLE IF EXISTS _near_winners")
        cursor.execute("CREATE TABLE _near_winners (source_rowid INTEGER PRIMARY KEY)")
        cursor.executemany("INSERT INTO _near_winners VALUES (?)", [(r,) for r in winner_rowids])
        db.commit()


    @staticmethod
    def _elect_group(elements: list[dict], indices: list[int]) -> int:
        """Return the index (into *elements*) of the best candidate in *indices*.

        Applies the same sequential voting logic as :meth:`get_unique_urls_parallel`
        so that the near-duplicate pass honours identical priorities to the URL and
        content passes:

        * A candidate **votes** to become the new elected if it is longer than the
          current best, OR if it is more recent.
        * Its vote is **cancelled** if it is strictly older than the current best,
          regardless of length.
        * Its vote is **cancelled** if it carries the ``"external"`` category while
          the current best does not.

        ``datetime`` values are compared as ISO-8601 strings (lexicographic order
        is equivalent to chronological order for ISO dates).  An empty/null string
        is treated as the oldest possible date.

        Args:
            elements: Full list of row dicts (keys: rowid, url, parsed, datetime,
                      length, category).
            indices:  Indices into *elements* forming the near-duplicate group.

        Returns:
            The index (into *elements*) of the elected winner.
        """
        elected          = indices[0]
        elected_category = elements[elected]["category"]
        best_length      = elements[elected]["length"]
        best_date        = elements[elected]["datetime"]   # ISO string or ""

        for idx in indices[1:]:
            elem          = elements[idx]
            cand_date     = elem["datetime"]
            cand_length   = elem["length"]
            cand_category = elem["category"]
            vote          = False

            if cand_length > best_length:
                best_length = cand_length
                vote = True

            if cand_date and cand_date > best_date:
                best_date = cand_date
                vote = True
            elif cand_date and best_date and cand_date < best_date:
                vote = False   # strictly older: length advantage is irrelevant

            # External content loses to non-external regardless of other criteria.
            if cand_category == "external" and elected_category != "external":
                vote = False

            if vote:
                elected          = idx
                elected_category = cand_category

        return elected

    def _rebuild_pages(self, db: sqlite3.Connection, winners_table: str) -> None:
        """Replace ``pages`` with only the rows identified in *winners_table*.

        The original schema (including any custom columns) is preserved by
        extracting the ``CREATE TABLE`` SQL from ``sqlite_master`` and cloning it.
        The swap is performed inside a single ``BEGIN`` / ``COMMIT`` block to be
        atomic.  All pipeline temp tables are dropped on success.

        Args:
            db:            Open database connection.
            winners_table: Name of the temp table whose ``source_rowid`` column
                           identifies the rows to keep.

        Raises:
            RuntimeError: If the ``pages`` table cannot be found in
                          ``sqlite_master``.
        """
        cursor = db.cursor()

        row = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='pages'"
        ).fetchone()
        if row is None:
            raise RuntimeError("pages table not found in sqlite_master")

        # We need to cover both cases: pages and "pages"
        create_sql = row[0].replace("CREATE TABLE pages", "CREATE TABLE pages_new")
        create_sql = create_sql.replace("CREATE TABLE \"pages\"", "CREATE TABLE pages_new")

        cursor.execute("DROP TABLE IF EXISTS pages_new")

        cursor.execute(create_sql)
        cursor.execute(f"""
            INSERT INTO pages_new
            SELECT * FROM pages
            WHERE rowid IN (SELECT source_rowid FROM {winners_table})
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


    def get_close_content(self, posts: list[web_page], threshold: float = 0.90, distance: float = 50) -> list[web_page]:
        """Find near-duplicate by computing the Levenshtein distance between pages contents.

        Params:
            posts: 
                dictionnary mapping an unused key to a liste of [core.types.web_page][]

            threshold: 
                the minimum distance ratio of Lenvenshtein metric for 2 contents to be assumed duplicates

            distance: 
                for efficiency, the list of web_page is first sorted alphabetically by URL, assuming duplicates
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

                        if elements[idx]["datetime"] and elements[idx]["datetime"] > date:
                            date = elements[idx]["datetime"]
                            vote = True
                        elif elements[idx]["datetime"] and elements[idx]["datetime"] < date:
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
            posts: Flat list of :class:`core.types.web_page` dicts.  The list
                   is modified in-place; callers should not rely on its contents
                   after this call returns.

        Returns:
            Deduplicated list of sanitised :class:`core.types.web_page` dicts,
            ready for downstream use (``to_db=False``).  Also writes a
            ``domains`` frequency file via :func:`~core.utils.get_models_folder`.
        """
        print("[dedup] Phase 0 – initial posts :", len(posts))

        # Phase 1 + 2: URL canonicalization + URL deduplication
        posts = self.get_unique_urls(posts)
        print("[dedup] Phase 2 – after URL deduplication             :", len(posts))

        # Phase 3: Exact-content deduplication
        posts = self.get_unique_content(posts)
        print("[dedup] Phase 3 – after exact-content deduplication   :", len(posts))

        # Phase 4: Near-duplicate removal (optional)
        if self.threshold < 1.0:
            posts = self.get_close_content(posts, threshold=self.threshold, distance=self.distance)
            print("[dedup] Phase 4 – after near-duplicate removal        :", len(posts))
        else:
            print("[dedup] Phase 4 – near-duplicate removal skipped (threshold=1.0)")

        # List all unique domains with their frequency
        counts = Counter([post["domain"] for post in posts])
        print(f"[dedup] Done – {len(counts)} unique domains, {len(posts)} posts remain.")

        # Sort domains by frequency
        counts = dict(sorted(counts.items(), key=lambda counts: counts[1]))

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
