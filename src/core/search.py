from scipy.signal import convolve2d
from enum import IntEnum

import sqlite3
import joblib
import pickle
import numpy as np
import regex as re
import os

from rank_bm25 import BM25Plus
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import csr_matrix


from .utils import get_models_folder, ensure_decompressed, timeit
from .nlp import Lexicon, Word2Vec, WordEmbedding
from . import database
    
class BM25PlusCSR:
    """
    BM25+ with CSR inverted index:
    - doc_ids / tfs stored in contiguous arrays
    - indptr for token → posting list slicing
    - fully vectorized scoring
    """

    __slots__ = (
        "k1",
        "b",
        "delta",
        "corpus_size",
        "avgdl",
        "doc_lens",
        "denom_const",
        "idf",
        "doc_ids",
        "tfs",
        "indptr",
    )

    def __init__(
        self,
        corpus: list[list[int]],
        word2vec,
        k1: float = 1.7,
        b: float = 0.3,
        delta: float = 0.65,
    ):
        self.k1 = np.float32(k1)
        self.b = np.float32(b)
        self.delta = np.float32(delta)

        self.corpus_size = len(corpus)
        vocab_size = len(word2vec.wv)

        # 1. Document lengths
        flat_docs = []
        doc_lens = np.zeros(self.corpus_size, dtype=np.int32)

        for i, doc in enumerate(corpus):
            flat_docs.append(doc)
            doc_lens[i] = len(doc)

        self.doc_lens = doc_lens
        self.avgdl = np.float32(doc_lens.mean() if self.corpus_size else 1.0)

        self.denom_const = (
            self.k1 * (1.0 - self.b + self.b * (doc_lens / self.avgdl))
        ).astype(np.float32)

        # 2. Build raw postings (doc_id, token_id, tf)
        self.doc_ids = []
        token_ids = []
        self.tfs = []

        for d, doc in enumerate(flat_docs):
            uniq, df = np.unique(doc, return_counts=True)
            for t, c in zip(uniq, df):
                self.doc_ids.append(d)
                token_ids.append(t)
                self.tfs.append(c)

        self.doc_ids = np.asarray(self.doc_ids, dtype=np.int32)
        token_ids = np.asarray(token_ids, dtype=np.int32)
        self.tfs = np.asarray(self.tfs, dtype=np.uint16)

        # 3. Sort by token_id (critical for CSR)
        order = np.argsort(token_ids, kind="mergesort")

        self.doc_ids = self.doc_ids[order]
        token_ids = token_ids[order]
        self.tfs = self.tfs[order]

        # 4. Build CSR index (indptr)
        self.indptr = np.zeros(vocab_size + 1, dtype=np.int32)
        df = np.bincount(token_ids, minlength=vocab_size)
        self.indptr[1:] = np.cumsum(df)

        # 5. Compute IDF (BM25+ log-smoothed)
        self.idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5)).astype(np.float32)

    @classmethod
    def from_cache(
        cls,
        k1: float,
        b: float,
        delta: float,
        corpus_size: int,
        avgdl: float,
        doc_lens: np.ndarray,
        denom_const: np.ndarray,
        idf: np.ndarray,
        doc_ids: np.ndarray,
        tfs: np.ndarray,
        indptr: np.ndarray,
    ):
        ranker = cls.__new__(cls)
        ranker.k1 = np.float32(k1)
        ranker.b = np.float32(b)
        ranker.delta = np.float32(delta)
        ranker.corpus_size = corpus_size
        ranker.avgdl = np.float32(avgdl)
        ranker.doc_lens = doc_lens
        ranker.denom_const = denom_const
        ranker.idf = idf
        ranker.doc_ids = doc_ids
        ranker.tfs = tfs
        ranker.indptr = indptr

        return ranker

    def add_documents(self, new_corpus: list[list[int]]):
        """Append documents to the CSR index IN PLACE, reusing the existing postings — the old
        corpus is never re-read. New document ids continue from ``corpus_size``. Powers the
        incremental (diff-only) index update: the existing token→posting arrays are merged with the
        new documents' postings token-by-token (both already grouped by token), so the cost scales
        with the delta, not the whole corpus. IDF and the length-normalisation constants are then
        recomputed from the updated per-token document frequencies (cheap, fully vectorised)."""
        M = len(new_corpus)
        if M == 0:
            return

        vocab_size = len(self.indptr) - 1
        base = self.corpus_size

        new_doc_lens = np.fromiter((len(d) for d in new_corpus), dtype=np.int32, count=M)

        # Raw postings for the new documents: (doc_id, token_id, tf), grouped per doc.
        nd, nt, nf = [], [], []
        for j, doc in enumerate(new_corpus):
            if not doc:
                continue
            uniq, cnt = np.unique(np.asarray(doc, dtype=np.int32), return_counts=True)
            nd.append(np.full(uniq.shape, base + j, dtype=np.int32))
            nt.append(uniq.astype(np.int32))
            nf.append(cnt.astype(np.uint16))
        if nt:
            new_doc_ids = np.concatenate(nd)
            new_token_ids = np.concatenate(nt)
            new_tfs = np.concatenate(nf)
        else:
            new_doc_ids = np.empty(0, np.int32)
            new_token_ids = np.empty(0, np.int32)
            new_tfs = np.empty(0, np.uint16)

        old_df = np.diff(self.indptr).astype(np.int64)
        new_df = np.bincount(new_token_ids, minlength=vocab_size).astype(np.int64)
        comb_df = old_df + new_df

        new_indptr = np.zeros(vocab_size + 1, dtype=np.int32)
        new_indptr[1:] = np.cumsum(comb_df)
        total = int(new_indptr[-1])

        merged_doc_ids = np.empty(total, dtype=np.int32)
        merged_tfs = np.empty(total, dtype=np.uint16)

        # Scatter the OLD postings to their new home: for token t, its block moves from
        # indptr[t] to new_indptr[t] (same intra-token order preserved).
        if self.doc_ids.size:
            old_tokens = np.repeat(np.arange(vocab_size, dtype=np.int64), old_df)
            old_local = np.arange(self.doc_ids.size, dtype=np.int64) - self.indptr[:-1].astype(np.int64)[old_tokens]
            dest_old = new_indptr[old_tokens] + old_local
            merged_doc_ids[dest_old] = self.doc_ids
            merged_tfs[dest_old] = self.tfs

        # Scatter the NEW postings right after each token's old block.
        if new_token_ids.size:
            order = np.argsort(new_token_ids, kind="mergesort")
            s_tok = new_token_ids[order].astype(np.int64)
            s_doc = new_doc_ids[order]
            s_tf = new_tfs[order]
            new_ptr = np.zeros(vocab_size + 1, dtype=np.int64)
            new_ptr[1:] = np.cumsum(new_df)
            new_local = np.arange(s_tok.size, dtype=np.int64) - new_ptr[s_tok]
            dest_new = new_indptr[s_tok] + old_df[s_tok] + new_local
            merged_doc_ids[dest_new] = s_doc
            merged_tfs[dest_new] = s_tf

        self.corpus_size = base + M
        self.doc_lens = np.concatenate([self.doc_lens, new_doc_lens])
        self.avgdl = np.float32(self.doc_lens.mean() if self.corpus_size else 1.0)
        self.denom_const = (
            self.k1 * (1.0 - self.b + self.b * (self.doc_lens / self.avgdl))
        ).astype(np.float32)
        self.indptr = new_indptr
        self.doc_ids = merged_doc_ids
        self.tfs = merged_tfs
        self.idf = np.log((self.corpus_size - comb_df + 0.5) / (comb_df + 0.5)).astype(np.float32)

    def get_scores(self, tokens: list[int]) -> np.ndarray:
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        if not tokens:
            return scores

        for t in set(tokens):

            i0 = self.indptr[t]
            i1 = self.indptr[t + 1]

            if i0 == i1:
                continue

            docs = self.doc_ids[i0:i1]
            freq = self.tfs[i0:i1]
            denom = freq + self.denom_const[docs]
            scores[docs] += self.idf[t] * ((freq * self.k1 + 1.0) / denom + self.delta)

        return scores


class search_methods(IntEnum):
    """Search methods available"""
    AI = 1
    """Vector-based similarity on document centroid in embedding space"""
    
    FUZZY = 2
    """BM25+ keywords statistics on normalized and stemmed content"""
    
    MIXED = 3
    """Combination of `AI` and `FUZZY` aggregated by Reciprocal Rank Fusion."""

def _subset_clause(db: sqlite3.Connection) -> str:
    """SQL boolean selecting the *searchable* rows of the ``pages`` table.

    Returns ``"in_index = 1"`` when the table carries the ``in_index`` flag — the
    monolithic canonical, where only a curated subset is meant for search and the
    rest is LM-only / archived material — else ``"1"`` (a DB that is already a pure
    search subset: the legacy ``chantal.db`` or the slimmed deploy copy). Composes
    into any query as ``WHERE {clause}`` or ``... AND {clause}``.

    Kept as a one-line ``PRAGMA`` lookup (cheap) so every corpus read can restrict
    itself consistently without threading a flag through the whole class.
    """
    cols = {row[1] for row in db.execute("PRAGMA table_info(pages)")}
    return "in_index = 1" if "in_index" in cols else "1"


class Indexer():
    @timeit()
    def __init__(self,
                 db: sqlite3.Connection,
                 name: str,
                 word2vec: WordEmbedding,
                 strip_collocations: bool = False,
                 principal_components: int = 1):
        """Search engine based on word similarity.

        Arguments:
            db:
                Opened SQLite database containing at least a `pages` table of [core.types.web_page][]
                items saved as database.
            
            name: 
                name under which the model will be saved for la ter reuse.

            word2vec: 
                the instance of word embedding model.

            strip_collocations: 
                remove the matrix of collocations in documents, which is the list of word tokens represented by their index in the
                word2vec dictionnary. It is used for [core.search.Indexer.find_query_pattern][], which is optional and significatively slower
                (but not significatively better), so if you don't plan on using it, removing collocations saves some RAM and I/O.

            principal_components: 
                number of principal components to compute and remove from the index dataset. 
                This helps to make queries more selective and specific in the presence of boilerplate text and formatting language in the sampling.

        NOTE:
            The class is optimized to run online, on server: load fast when spawning a new server-side worker,
            use RAM sparingly.
            
        """

        self.sql: str = ""
        """Cache the previous SQL filtering conditions"""

        self.word2vec: WordEmbedding = word2vec
        """Word embedding model (Word2Vec or FastText), via the WordEmbedding interface"""
        
        self.init_stats_table(db)
        self.init_categories_table(db)
        self.init_pages_search_indexes(db)

        # TODO
        self.collocations: np.ndarray | None = None # if strip_collocations else [doc[2] for doc in docs]
        """Store the list of document tokens encoded by their index number in the
        Word2Vec vocabulary. Unknown tokens are discarded. This gives a symbolic
        and more compact representation of tokens collocations in documents (32 bits/token).

        Documents are on the first axis.
        """

        ##################################################################
        # 1. Precompute the BM25+ document-wise constants (stats & counts)
        ##################################################################

        # TODO: replace by database.SQLitePageCorpu
        # ORDER BY rowid (not url): rowid is always a unique, stable total order, so the
        # three arrays (this BM25 corpus, self.vectors, and search_rowid) align row-for-row
        # even when the pages table allows duplicate URLs. ORDER BY url is only a total order
        # when url is unique (PRIMARY KEY); relying on it would silently misalign the numpy
        # arrays with search_rowid once the URL primary key is dropped.
        #
        # {subset}: on the monolithic canonical only `in_index = 1` rows are searchable;
        # this WHERE keeps the BM25 corpus, self.vectors and search_rowid (all read the
        # same `WHERE {subset} ORDER BY rowid`) covering exactly that subset, in lockstep.
        subset = _subset_clause(db)

        # To spare some memory, build a symbolic corpus representation using
        # word indices in the Word2Vec vocabulary, then construct a local
        # BM25Plus reimplementation that precomputes freqs/lengths/inverted index.
        #
        # STREAM the cursor (don't .fetchall() first): materializing all N stemmed
        # token-lists AND the derived index-list at once doubles peak RAM and OOMs on
        # a large corpus. Iterating consumes rows in small batches, so only the compact
        # int-index corpus is held. key_to_index is hoisted out of the inner loop.
        key_to_index = self.word2vec.wv.key_to_index
        corpus_token_indices = [
            [
                key_to_index[word]
                for sentence in stemmed
                for word in sentence
                if word in key_to_index
            ]
            for (stemmed,) in db.execute(f"SELECT stemmed FROM pages WHERE {subset} ORDER BY rowid")
        ]

        self.ranker: BM25PlusCSR = BM25PlusCSR(corpus_token_indices, self.word2vec, k1=1.8, b=0.4, delta=0.8)
        """BM25+ CSR ranker (TF-IDF)."""

        # Use our own implementation of BM25+, which gives similar ranking albeit with different coeffs
        # but runs 7 times faster. Otherwise:
        # self.ranker = BM25Plus(corpus_token_indices, k1=1.7, b=0.3, delta=0.65)
        # BM25+ values from https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

        #######################################################################################
        # 2. Compute the embedded corpus principal component(s) and remove them from embeddings
        #######################################################################################

        # PC encode boilerplate text, stopwords and non-specific language structures that hinder
        # discrimination between relevant and irrelevant documents. You can see them as the "common glue"
        # between all documents in the corpus, which is the opposite of what we are looking for to retrieve information.

        # ORDER BY rowid + the same {subset} to stay aligned with the BM25 corpus and search_rowid.
        # Preallocate and fill row-by-row from a streamed cursor: the old
        # np.array([... for ... in cursor.fetchall()]) held a Python list of N (300,) arrays
        # AND the final matrix at the same time — a needless multi-GB transient on a big corpus.
        n_docs = len(corpus_token_indices)
        cursor = db.execute(f"SELECT vectorized FROM pages WHERE {subset} ORDER BY rowid")
        first = cursor.fetchone()
        dim = first[0].shape[0]
        self.vectors = np.empty((n_docs, dim), dtype=np.float32)
        self.vectors[0] = first[0]
        for i, (v,) in enumerate(cursor, start=1):
            self.vectors[i] = v
        """Store the list of document-wise vector embeddings, where the vector represents
        the normalized centroid of tokens vectors contained the document.
        Documents are on the first axis.
        """

        pca = PCA(n_components=principal_components)
        pca.fit(self.vectors)
        self.pc: np.ndarray = pca.components_
        """Principal component(s) of the dataset vectors (normalized)"""

        # Remove PC from embedding vectors. 
        # Note: DB document embeddings are left unchanged.
        self.vectors = self.normalize_pc(self.vectors)

        # Assign a stable 0-based search_rowid to every page in URL order,
        # replacing the in-memory self.index / self.url_to_index / self._index_arr LUTs.
        # The column persists in the DB, so it survives process restarts and VACUUM.
        self._build_search_rowids(db)

        ###############
        # 4. Misc stats
        ###############
        self.stats = self.build_stats(db)
        self.words = self.stats["words"]
        self.pages = self.stats["pages"]

        self.save_search_stats(db, self.stats)
        self.save_categories_index(db, self.stats["category_counts"])

        # Clusterize/quantize pages for performance and "topic" extraction
        self.get_clusters(db)

        # 5. Save the pickled object to disk for reuse
        self.save(name)

        # 6. Compress DB just in case
        database.compress_db(db)


    def init_stats_table(self, db: sqlite3.Connection):
        """
        Create or migrate the plain-SQLite stats table.

        Scalar stats use `item = ''`. Grouped stats use `name` for the metric
        family and `item` for the domain/category/language key.
        """
        cursor = db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stats (
                name TEXT NOT NULL,
                item TEXT NOT NULL DEFAULT '',
                value_integer INTEGER,
                value_real REAL,
                value_text TEXT,
                PRIMARY KEY (name, item)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_stats_name
            ON stats(name)
        """)

        db.commit()


    def init_categories_table(self, db: sqlite3.Connection):
        """
        Create a queryable catalog of all page categories.
        """
        cursor = db.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category TEXT PRIMARY KEY,
                pages INTEGER NOT NULL
            )
        """)

        db.commit()


    def init_pages_search_indexes(self, db: sqlite3.Connection):
        """
        Create or migrate all persistent indexes needed by the search layer.

        ``search_rowid`` is an explicit INTEGER column we assign (0, 1, 2 …)
        to every page in ``ORDER BY url`` order at build time.  Because it is
        a real column value, ``VACUUM`` cannot renumber it — unlike SQLite's
        implicit rowid for tables with a TEXT primary key.

        The **covering index** ``idx_pages_search_rowid_category`` is the key
        performance fix for ``filter_contents``: SQLite can answer the entire
        candidate-filter query from the compact index without ever touching the
        main ``pages`` rows (which are large due to stored embeddings / content).
        Benchmark: 500-candidate category filter 439 ms → < 2 ms.
        """
        cursor = db.cursor()

        # Migration-safe: ignore the error if the column already exists.
        try:
            cursor.execute("ALTER TABLE pages ADD COLUMN search_rowid INTEGER")
        except sqlite3.OperationalError:
            pass  # column already present

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_search_rowid
            ON pages(search_rowid)
        """)

        # Covering index: (search_rowid, category) lets SQLite evaluate
        # WHERE category … filters on candidate sets with zero main-table I/O.
        # SQLite secondary indexes for TEXT-PK tables implicitly include `url`
        # (the PK) so url-based WHERE conditions are also covered for free.
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_search_rowid_category
            ON pages(search_rowid, category)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_category_url
            ON pages(category, url)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_category_coalesce_url
            ON pages(COALESCE(category, ''), url)
        """)

        db.commit()


    def save_search_stats(self, db: sqlite3.Connection, stats: dict):
        """
        Store cheap display and diagnostic metadata in plain SQLite rows.
        """
        rows = []

        def append(name: str, value, item: str = ""):
            if isinstance(value, bool):
                rows.append((name, item, int(value), None, None))
            elif isinstance(value, int):
                rows.append((name, item, value, None, None))
            elif isinstance(value, float):
                rows.append((name, item, None, value, None))
            elif value is None:
                rows.append((name, item, None, None, None))
            else:
                rows.append((name, item, None, None, str(value)))

        scalar_keys = (
            "words",
            "pages",
            "domains",
            "categories",
            "most_recent_datetime",
            "oldest_datetime",
            "total_content_length",
            "average_content_length",
            "max_content_length",
        )

        for key in scalar_keys:
            append(key, stats[key])

        for domain, pages in stats["domain_counts"].items():
            append("domain_pages", pages, domain)

        for category, pages in stats["category_counts"].items():
            append("category_pages", pages, category)

        for lang, pages in stats["language_counts"].items():
            append("language_pages", pages, lang)

        cursor = db.cursor()
        cursor.execute("DELETE FROM stats")
        cursor.executemany("""
            INSERT INTO stats (name, item, value_integer, value_real, value_text)
            VALUES (?, ?, ?, ?, ?)
        """, rows)
        db.commit()


    def save_categories_index(self, db: sqlite3.Connection, category_counts: dict[str, int]):
        """
        Store all existing non-empty categories and their page counts.
        """
        rows = [
            (category, pages)
            for category, pages in category_counts.items()
            if category != "(none)"
        ]

        cursor = db.cursor()
        cursor.execute("DELETE FROM categories")
        cursor.executemany("""
            INSERT INTO categories (category, pages)
            VALUES (?, ?)
        """, rows)
        db.commit()


    def build_stats(self, db: sqlite3.Connection) -> dict:
        """
        Compute index metadata while the database is writable/prepared.
        """
        cursor = db.cursor()

        # All dashboard stats describe the SEARCHABLE corpus, so every aggregate is
        # scoped to the same subset the index is built from (in_index on the canonical).
        sub = _subset_clause(db)

        pages = cursor.execute(f"SELECT COUNT(*) FROM pages WHERE {sub}").fetchone()[0]
        domains = cursor.execute(f"""
            SELECT COUNT(DISTINCT COALESCE(NULLIF(domain, ''), url))
            FROM pages
            WHERE {sub}
        """).fetchone()[0]
        categories = cursor.execute(f"""
            SELECT COUNT(DISTINCT category)
            FROM pages
            WHERE {sub}
              AND category IS NOT NULL
              AND category != ''
        """).fetchone()[0]
        most_recent_datetime = cursor.execute(f"""
            SELECT MAX(datetime)
            FROM pages
            WHERE {sub}
              AND datetime IS NOT NULL
        """).fetchone()[0]
        oldest_datetime = cursor.execute(f"""
            SELECT MIN(datetime)
            FROM pages
            WHERE {sub}
              AND datetime IS NOT NULL
        """).fetchone()[0]
        length_stats = cursor.execute(f"""
            SELECT COALESCE(SUM(length), 0),
                   COALESCE(AVG(length), 0),
                   COALESCE(MAX(length), 0)
            FROM pages
            WHERE {sub}
              AND length IS NOT NULL
        """).fetchone()
        domain_counts = cursor.execute(f"""
            SELECT COALESCE(NULLIF(domain, ''), url) AS domain,
                   COUNT(*) AS pages
            FROM pages
            WHERE {sub}
            GROUP BY COALESCE(NULLIF(domain, ''), url)
            ORDER BY pages DESC, domain ASC
        """).fetchall()
        category_counts = cursor.execute(f"""
            SELECT COALESCE(NULLIF(category, ''), '(none)') AS category,
                   COUNT(*) AS pages
            FROM pages
            WHERE {sub}
            GROUP BY COALESCE(NULLIF(category, ''), '(none)')
            ORDER BY pages DESC, category ASC
        """).fetchall()
        language_counts = cursor.execute(f"""
            SELECT COALESCE(NULLIF(lang, ''), '(unknown)') AS lang,
                   COUNT(*) AS pages
            FROM pages
            WHERE {sub}
            GROUP BY COALESCE(NULLIF(lang, ''), '(unknown)')
            ORDER BY pages DESC, lang ASC
        """).fetchall()

        return {
            "words": len(self.word2vec.wv),
            "pages": pages,
            "domains": domains,
            "categories": categories,
            "most_recent_datetime": most_recent_datetime,
            "oldest_datetime": oldest_datetime,
            "total_content_length": length_stats[0],
            "average_content_length": length_stats[1],
            "max_content_length": length_stats[2],
            "domain_counts": dict(domain_counts),
            "category_counts": dict(category_counts),
            "language_counts": dict(language_counts),
        }


    @staticmethod
    def array_to_raw(array: np.ndarray) -> sqlite3.Binary:
        """
        Store arrays without the `.npy` wrapper used by SQLite converters.
        Raw contiguous blobs hydrate faster with `np.frombuffer()` at runtime.
        """
        return sqlite3.Binary(np.ascontiguousarray(array).tobytes())


    @staticmethod
    def raw_to_array(blob: bytes, dtype: np.dtype, shape: tuple[int, ...] | None = None) -> np.ndarray:
        array = np.frombuffer(blob, dtype=dtype)

        if shape is not None:
            array = array.reshape(shape)

        return array


    def _build_search_rowids(self, db: sqlite3.Connection):
        """Assign ``search_rowid = 0, 1, 2, …`` to every page in ``ORDER BY rowid``.

        This is the single source of truth that glues the DB to the in-RAM
        numpy arrays (``self.vectors``, ``self.ranker``).  Both are built by
        reading pages ``ORDER BY rowid``, so position 0 in every array
        corresponds to ``search_rowid = 0``, and so on.

        ``rowid`` (not ``url``) is used as the assignment order because it is always a
        unique, stable total order — the mapping stays a clean bijection even if the
        pages table permits duplicate URLs. The previous ``ORDER BY url`` + ``WHERE url = ?``
        implementation required ``url`` to be a UNIQUE/PRIMARY KEY; this one does not, which
        is what lets the URL primary key be dropped.

        Run once per Indexer build; the values survive ``VACUUM`` because
        they live in a real column, not in SQLite's internal b-tree position.

        Also stores a two-point fingerprint (page count + boundary URL hash)
        in ``self.index_fingerprint`` so ``verify_db_integrity()`` can detect
        invalidating changes in O(1) at load time.
        """
        # Only the searchable subset gets a search_rowid; everything else is cleared to NULL
        # so it can never surface as a candidate (rank/filter_contents key off search_rowid).
        # Guard on `search_rowid IS NOT NULL` so a fresh build (column all-NULL after the ALTER)
        # touches nothing — an unguarded full-table UPDATE writes a rollback journal the size of
        # the whole DB and can exhaust the disk on a large monolithic canonical.
        subset = _subset_clause(db)
        db.execute(
            f"UPDATE pages SET search_rowid = NULL WHERE search_rowid IS NOT NULL AND NOT ({subset})"
        )
        db.commit()
        rowids = [row[0] for row in db.execute(f"SELECT rowid FROM pages WHERE {subset} ORDER BY rowid")]

        # Assign in committed chunks: each commit truncates the (TRUNCATE-mode) rollback journal,
        # so a 400k-row assignment on a multi-GB DB never accumulates a journal larger than one
        # chunk's touched pages — the difference between completing and filling the disk.
        CHUNK = 50000
        for start in range(0, len(rowids), CHUNK):
            db.executemany(
                "UPDATE pages SET search_rowid = ? WHERE rowid = ?",
                ((start + j, rid) for j, rid in enumerate(rowids[start:start + CHUNK])),
            )
            db.commit()

        # Cheap two-point fingerprint: count + hash of the boundary URLs (see _compute_fingerprint).
        self.index_fingerprint = self._compute_fingerprint(db)


    def _compute_fingerprint(self, db: sqlite3.Connection) -> tuple:
        """(searchable-row count, sha256 of first+last boundary URLs in search_rowid order).
        The two-point signature ``verify_db_integrity`` checks at load time; recomputed here after
        BOTH a full build and an incremental append so the artifact stays loadable against its DB."""
        import hashlib
        count = db.execute("SELECT COUNT(*) FROM pages WHERE search_rowid IS NOT NULL").fetchone()[0]
        first = db.execute("SELECT url FROM pages WHERE search_rowid IS NOT NULL ORDER BY search_rowid ASC  LIMIT 1").fetchone()
        last  = db.execute("SELECT url FROM pages WHERE search_rowid IS NOT NULL ORDER BY search_rowid DESC LIMIT 1").fetchone()
        boundary = "".join([
            (first[0] if first and first[0] else ""),
            (last[0]  if last  and last[0]  else ""),
        ]).encode()
        return (count, hashlib.sha256(boundary).hexdigest())


    def verify_db_integrity(self, db: sqlite3.Connection, full: bool = False):
        """Raise ``RuntimeError`` if the DB has changed since this Indexer was built.

        Arguments:
            full:
                ``False`` (default) — page count + boundary-URL hash, three
                O(log N) index seeks, effectively O(1).  Catches all
                insertions, deletions, and boundary-URL edits.
                ``True`` — hashes every URL in rowid order, O(N), detects any
                mid-corpus mutation.

        Called automatically by :meth:`load`.

        Bounds model (append-only + holes): the pickled arrays hold ``len(self.vectors)`` rows at
        positions 0…n-1, assigned once at build/append time and never renumbered (except a full
        rebuild). So the only hard invariant is that every ``search_rowid`` in the DB indexes a real
        array position — i.e. ``MAX(search_rowid) < len(self.vectors)`` and the number of assigned
        rowids does not exceed the array length. Rows deleted/demoted since the build are holes
        (fewer assigned rowids than vectors) and are fine — ``rank()`` skips them. A count that
        *exceeds* the array, or an out-of-range ``search_rowid``, means the wrong artifact for this
        DB (or a missed rebuild) and would index out of bounds, so it is rejected.
        """
        n = int(self.vectors.shape[0])
        max_sr = db.execute("SELECT MAX(search_rowid) FROM pages").fetchone()[0]
        if max_sr is not None and max_sr >= n:
            raise RuntimeError(
                f"DB/index mismatch: found search_rowid {max_sr} but the index has only {n} "
                f"vectors. Wrong engine for this DB, or the index needs rebuilding."
            )
        assigned = db.execute(
            "SELECT COUNT(*) FROM pages WHERE search_rowid IS NOT NULL"
        ).fetchone()[0]
        if assigned > n:
            raise RuntimeError(
                f"DB/index mismatch: {assigned} assigned search_rowids exceed {n} indexed vectors. "
                f"Rebuild the index (chantal-05 without --incremental)."
            )




    def normalize_pc(self, vector: np.ndarray) -> np.ndarray:
        """Remove the principal component of the dataset to the vector.
        This helps removing stopwords, webpage boilerplates (menu, sidebars),
        formatting language and SEO junk, and makes cosine similarity between
        query and documents more specific.

        Taken from _A simple but tough-to-beat baseline for sentence embeddings_,
        Sanjeev Arora, Yingyu Liang, Tengyu Ma. https://openreview.net/pdf?id=SyK00v5xx

        Arguments:
            vector: 
                can be a single vector (1D) or a document-wise stack of vectors (2D).
                We always consider the embedding vector to be on the last axis, document-wise
                vectors should be vertically stacked.
        Returns:
            normalized vector
        """
        vector = vector - np.matmul(np.matmul(vector, self.pc.T), self.pc)

        return vector / (np.linalg.norm(vector, axis=-1, keepdims=True) + 1e-8)


    @timeit()
    def filter_contents(self,
                        db: sqlite3.Connection,
                        sql_query: str = "",
                        sql_params: list[str] | None = None,
                        candidate_indices: np.ndarray | list[int] | None = None) -> list[int]:
        """Filter pages by an arbitrary SQL predicate, returning ``search_rowid`` integers.

        With ``candidate_indices``, the predicate is evaluated only over that
        small set via ``WHERE search_rowid IN (...)``.  This avoids a full table
        scan and, combined with the covering index
        ``idx_pages_search_rowid_category``, avoids touching the main
        ``pages`` rows entirely for category/url filters — the most common case.

        Without ``candidate_indices``, the full table is scanned (used for
        building candidate sets outside of :meth:`rank`).
        """
        if sql_params is None:
            sql_params = []

        if candidate_indices is not None:
            # Leave a safe margin below the 999-variable limit present in older
            # SQLite builds; sql_params count against the same limit.
            max_chunk = max(1, 900 - len(sql_params))
            matched: list[int] = []
            indices_list = [int(i) for i in candidate_indices]

            for start in range(0, len(indices_list), max_chunk):
                chunk = indices_list[start : start + max_chunk]
                ph = ",".join("?" * len(chunk))

                # ``WHERE search_rowid IN (…)`` lets SQLite use
                # idx_pages_search_rowid_category as a covering index:
                # category is read directly from the compact index without
                # touching the large main-table rows (embeddings / content).
                # sql_query starts with WHERE; rewrite it as AND so it composes
                # with the IN predicate we already own.
                extra = (
                    " AND " + sql_query[sql_query.upper().index("WHERE") + 5:].strip()
                    if sql_query else ""
                )
                query = f"SELECT search_rowid FROM pages WHERE search_rowid IN ({ph}){extra}"
                matched.extend(
                    row[0] for row in db.execute(query, [*chunk, *sql_params])
                )

            return matched

        # No candidate set: scan the whole table.
        query = f"SELECT search_rowid FROM pages {sql_query} ORDER BY search_rowid"
        return [row[0] for row in db.execute(query, sql_params)]


    def save(self, name: str):
        # Save the model to a reusable object.
        #
        # FastText's `wv.buckets_word` is a recomputable cache — one tiny array
        # per vocabulary word, mapping it to its character-n-gram bucket ids.
        # Pickling it explodes the artifact into ~one-array-per-word objects, and
        # reconstructing those millions of small objects (not array I/O) is what
        # dominates engine load time. The query path never reads it (in-vocab
        # lookups use `wv.vectors`; OOV reconstruction hashes n-grams on the
        # fly), so we drop it from the artifact and rebuild it in O(vocab) on
        # load (see `load`). `wv.vectors_ngrams`, which OOV reconstruction *does*
        # need, is kept. Word2Vec models have no `buckets_word` and are unaffected.
        wv = getattr(self.word2vec, "wv", None)
        cached_buckets = getattr(wv, "buckets_word", None) if wv is not None else None
        try:
            if cached_buckets is not None:
                wv.buckets_word = None
            joblib.dump(self, get_models_folder(name + ".joblib"), compress=0, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            # Keep the live, in-memory model fully usable after saving.
            if cached_buckets is not None:
                wv.buckets_word = cached_buckets


    @classmethod
    @timeit()
    def load(cls, name: str, db: sqlite3.Connection):
        """Load an existing trained model by its name from the `../models` folder."""
        try:
            model = joblib.load(ensure_decompressed(get_models_folder(name) + ".joblib"))
        except FileNotFoundError:
            model = joblib.load(get_models_folder(name) + ".joblib.bz2")

        if not isinstance(model, Indexer):
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))

        # Rebuild the FastText `buckets_word` cache that `save` dropped from the
        # artifact. This is O(vocab) and far cheaper than unpickling one array
        # per word. No-op for Word2Vec (no such cache) and for older artifacts
        # that still carry buckets_word.
        wv = getattr(model.word2vec, "wv", None)
        if (wv is not None
                and getattr(wv, "buckets_word", None) is None
                and hasattr(wv, "recalc_char_ngram_buckets")):
            wv.recalc_char_ngram_buckets()

        # Ensure indexes introduced in later versions exist in the live DB.
        # CREATE INDEX IF NOT EXISTS and ALTER TABLE … ADD COLUMN are both
        # idempotent, so this is safe to call on every load.
        model.init_pages_search_indexes(db)

        # Guard against a DB modified since the Indexer was built.
        model.verify_db_integrity(db)

        return model


    def tokenize_query(self, query:str, language: str | None = None, meta_tokens: bool = True, n_grams: bool = True) -> list[str]:
        """Tokenize a query string, returning only tokens known to our vocabulary."""
        query = self.word2vec.tokenizer.normalize_text(query)

        if n_grams:
            # Use both variants with n-grams and without to maximize coverage
            without_ngrams = self.word2vec.tokenizer.tokenize_document_flat(query, language=language, meta_tokens=meta_tokens, n_grams=False)
            with_ngrams = self.word2vec.tokenizer.tokenize_document_flat(query, language=language, meta_tokens=meta_tokens, n_grams=True)
            tokens = (set(without_ngrams) | set(with_ngrams))
        else:
            tokens = set(self.word2vec.tokenizer.tokenize_document_flat(query, language=language, meta_tokens=meta_tokens, n_grams=False))

        # Filter out unknown tokens
        return [token for token in tokens if self.word2vec.get_word(token) is not None]


    def vectorize_query(self, tokenized_query: list[str], use_sif: bool = True, sif_smoothing: float = 1e-3) -> np.ndarray:
        """Prepare a text search query: cleanup, tokenize and get the centroid vector.

        Returns:
            tuple[vector, norm, tokens]
        """

        # Get the centroid of the word embedding vector.
        #
        # NOTE: we deliberately do NOT remove the document-set principal
        # component here. `self.pc` was fit on OUT-space document centroids,
        # whereas the query is built in IN-space (dual-embedding space model).
        # Subtracting an OUT-space direction from an IN-space vector is
        # geometrically inconsistent and mismatches the two spaces. PC removal
        # stays on the document side only (build time). `get_features` already
        # returns an L2-normalized centroid.
        return self.word2vec.get_features(tokenized_query, embed="IN", use_sif=use_sif, sif_smoothing=sif_smoothing)


    @timeit()
    def find_query_pattern(self,
                           indexed_query: np.ndarray[np.int32],
                           documents: list[tuple[int, str, float]],
                           fast: bool = False) -> list[tuple[int, str, float]]:
        """The rankers methods treat documents as continuous bag of words (CBOW).
        As such, they are good for topic extraction (aboutness), but they do not care about words colocations
        and ordering, therefore they loose syntactical meaning.

        This method adds an additional layer of detection using convolution filters that
        will detect word sequences, direct or reversed, and correct the similarity factor set by the
        other ranking methods using that collocation factor.

        Its major drawback is to be 100 to 500 times slower than the other rankers, due to 2D convolutions,
        which means it needs to run on a subset of the search index, after previous methods were tried,
        to refine a previous ranking.

        Parameters:
            indexed_query: 
                the search query tokens translated into their integer indices in the Word2Vec vocabulary.
                Use [core.nlp.Word2Vec.tokens_to_indices][] to convert the tokenized query.

            documents: 
                a symbolic list of documents, as a `(index, url, similarity)` tuple.

            fast: 
                if `True`, uses a simplified variant that is 6 times faster and only uses local averages.
                Results from this method are rather inaccurate, for example, for a request like `token_1 token_2`,
                sentences repeating `token_1` twice will score as much as sentences containing the desired sequence
                `token_1 token_2`. If `False`, use the convolutional filter.

        References:
            Text Matching as Image Recognition, Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, and Xueqi Cheng. (2016).
            https://arxiv.org/pdf/1602.06359.pdf

        """
        if self.collocations is None:
            raise ValueError("Collocations have not been precomputed for this indexer.")
        
        if not fast:
            kernel_direct = np.eye(indexed_query.size, dtype=np.float32) / indexed_query.size
            kernel_reverse = np.rot90(np.eye(3, dtype=np.float32)) / 3.

        kernel_query = np.ones(indexed_query.shape, dtype=np.float32) / indexed_query.size

        results = []

        for doc in documents:
            index = doc[0]
            url = doc[1]
            similarity = doc[2]

            collocations = self.collocations[index]
            if collocations.size > indexed_query.size:
                if fast:
                    # Fast variant of the following method. (6 times faster)
                    # Looses info about tokens order and yields disputable results regarding relevance.
                    interaction = (collocations[:, np.newaxis] == indexed_query).any(axis=1) / indexed_query.size
                else:
                    # Build the interaction matrix: True where doc[i] == indexed_query[j]
                    # Loosely inspired by https://arxiv.org/pdf/1610.08136.pdf
                    interaction = np.equal(collocations[:, np.newaxis], indexed_query)

                    # Find permutations of tokens, by packs of 3.
                    # Inspired by https://arxiv.org/pdf/1602.06359.pdf
                    direct = convolve2d(interaction, kernel_direct, mode='same', boundary='circular', fillvalue=0)
                    reverse = convolve2d(interaction, kernel_reverse, mode='same', boundary='circular', fillvalue=0)

                    # Sum both filters output and then average over the query direction
                    interaction = (direct + reverse).sum(axis=1) / (2. * indexed_query.size)

                # Moving average along the doc direction
                scores = np.convolve(interaction, kernel_query, mode="same")
                max_score = np.max(scores)

                results.append((index, url, similarity + max_score))

            else:
                results.append(doc)

        return results

    @timeit()
    def rank_fuzzy(self, tokens: list[str]) -> np.ndarray:
        symbolic_tokens = [self.word2vec.wv.key_to_index[word]
                           for word in tokens
                           if word in self.word2vec.wv.key_to_index]

        return self.ranker.get_scores(symbolic_tokens)

    @timeit()
    def rank_ai(self, tokens: list[str], fast: bool = False, clip: bool = False, coverage: float = 0.2,
                use_sif: bool = True, sif_smoothing: float = 1e-3) -> np.ndarray:
        """Cosine-similarity ranking against document centroid vectors.

        Arguments:
            tokens:     tokenised query (output of ``tokenize_query``).
            fast:       use a single dot-product against the aggregate query
                        vector instead of the per-token dual-embedding loop.
            clip:       clamp scores to [0, 1]. Off by default: clipping
                        saturates the SIF-weighted cosine and destroys rank
                        resolution, which is fatal for the rank-based RRF
                        fusion. Only enable when blending raw scores.
            coverage:   if cluster data has been loaded, restrict the matmul to
                        the nearest clusters that together cover at least this
                        fraction of the corpus. Documents outside the selected
                        clusters keep a score of 0; RRF fusion with BM25+ in
                        ``rank()`` still surfaces them. A small fixed handful of
                        clusters (the old behaviour) was far too aggressive for
                        broad queries — the right cluster was easily missed.
        """

        if fast:
            # The following seems very close to the next in terms of results.
            # Experimentally, I saw very little difference in rankings, at least not in the first results.
            # The by-the-book is perhaps more immune to keywords stuffing and more sensitive to structure.
            # Differences appear in the tail of the ranking, mostly.
            # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
            query_vec = self.vectorize_query(tokens, use_sif=use_sif, sif_smoothing=sif_smoothing)

            if self.cluster_centroids is not None:
                candidate_indices = self._cluster_candidate_indices(query_vec, coverage=coverage)
                if candidate_indices is not None and candidate_indices.size:
                    aggregate = np.zeros(self.vectors.shape[0], dtype=np.float32)
                    aggregate[candidate_indices] = np.nan_to_num(
                        self.vectors[candidate_indices] @ query_vec
                    )
                    return aggregate

            return np.nan_to_num(np.dot(self.vectors, query_vec))

        else:

            # This is the by-the-book dual embedding space as defined in
            # https://arxiv.org/pdf/1602.01137.pdf
            aggregate = np.zeros(self.vectors.shape[0], dtype=np.float32)
            weights = 0.

            # When clustering is active, build one candidate mask upfront from
            # the mean query direction so we don't recompute it per token.
            candidate_indices: np.ndarray | None = None
            if self.cluster_centroids is not None:
                mean_vec = self.vectorize_query(tokens, use_sif=use_sif, sif_smoothing=sif_smoothing)
                candidate_indices = self._cluster_candidate_indices(mean_vec, coverage=coverage)

            for token in tokens:
                # Compute the cosine similarity of centroids between query and documents,
                # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
                vector = self.word2vec.get_wordvec(token, embed="IN", normalize=True)
                if vector is not None:
                    # No normalize_pc() here: the per-token query vectors live in
                    # IN-space; self.pc is an OUT-space direction. See
                    # vectorize_query() for the rationale. Vector is already
                    # L2-normalized (normalize=True).
                    # SIF-weight each term's cosine so rare, discriminative
                    # tokens ("darktable") dominate common ones ("install").
                    weight = self.word2vec.SIF(token, a=sif_smoothing) if use_sif else 1.
                    if candidate_indices is not None:
                        aggregate[candidate_indices] += weight * np.nan_to_num(
                            self.vectors[candidate_indices] @ vector
                        )
                    else:
                        aggregate += weight * np.nan_to_num(self.vectors @ vector)

                    weights += weight
                else:
                    print(f"token {token} was not found in embedding vocabulary")

            # SIF-weighted average cosine, bounded in [-1, 1]. We must NOT divide
            # an unweighted cosine sum by sum(SIF): with small SIF weights that
            # blows past 1 and `clip` then flattens nearly every document to
            # exactly 1.0, collapsing the ranking into a giant tie that argsort
            # breaks by search_rowid — i.e. URL alphabetical order, which is how
            # irrelevant `a…`/`b…`/`c…` pages were surfacing at the top.
            scores = aggregate / weights if weights > 0. else aggregate

            # 0 means orthogonal/unrelated, negative means opposite direction.
            # For RRF only the order matters, so clipping is off by default;
            # enable it for callers that blend raw scores.
            return np.clip(scores, 0., 1.) if clip else scores

    @timeit()
    def _cluster_candidate_indices(self, query_vec: np.ndarray, coverage: float = 0.2, min_clusters: int = 8) -> np.ndarray | None:
        """Return the row indices of the documents in the nearest clusters to
        ``query_vec``, taking enough clusters (nearest first) to cover at least
        ``coverage`` of the corpus, with a floor of ``min_clusters``.

        Picking a fixed handful of clusters out of the ``pages / 200`` total is
        far too aggressive: for a broad query the right cluster is easily
        missed and all its documents get a hard 0 from the AI path. Covering a
        fraction of the corpus keeps the per-document matmul pruned for speed
        while making the gate robust to broad, ambiguous query directions.

        The centroid comparison is a (K x D) matmul over the ~``pages / 200``
        cluster centroids — negligible next to the document matmul it prunes.

        Arguments:
            query_vec:    normalised query vector, shape (D,).
            coverage:     minimum fraction of the corpus to include.
            min_clusters: lower bound on the number of clusters selected.
        """
        # Score every cluster centroid against the query.
        # Centroids are unit-normalised (mean of normalised vectors, then
        # re-normalised in get_clusters), so dot product == cosine similarity.
        cluster_similarity = self.cluster_centroids @ query_vec

        # Nearest clusters first.
        order = np.argsort(cluster_similarity)[::-1]

        # cluster_doc_indices keys are ordered identically to centroid rows
        # (both built from the same np.unique(labels) pass).
        labels_ordered = list(self.cluster_doc_indices.keys())

        target = max(1, int(coverage * self.vectors.shape[0]))
        floor = min(min_clusters, order.size)

        picked: list[np.ndarray] = []
        covered = 0
        for taken, pos in enumerate(order, start=1):
            idx = self.cluster_doc_indices[labels_ordered[pos]]
            picked.append(idx)
            covered += idx.size
            if taken >= floor and covered >= target:
                break

        return np.concatenate(picked)


    @staticmethod
    def _scores_to_ranks(scores: np.ndarray) -> np.ndarray:
        """Convert a score vector into 0-based ranks (rank 0 = highest score),
        as expected by [core.search.Indexer.rrf][]. Ties are broken by index.
        """
        order = np.argsort(scores)[::-1]
        ranks = np.empty(scores.shape[0], dtype=np.int32)
        ranks[order] = np.arange(scores.shape[0], dtype=np.int32)
        return ranks


    def rrf(self, ranks_1: np.ndarray, ranks_2: np.ndarray, coeff: float = 60,
            weight_1: float = 1.0, weight_2: float = 1.0) -> np.ndarray:
        """Reciprocal Rank Fusion

        Aggregate 2 sets of page rankings obtained from different semantic geometries and weighted differently.

        Reference:
            _Reciprocal rank fusion outperforms condorcet and individual rank learning methods_,
            Gordon V. Cormack, Charles L A Clarke, Stefan Buettcher.
            https://dl.acm.org/doi/10.1145/1571941.1572114

        Arguments:
            ranks_1:
                0-based ranks from the first ranker (best = 0).
            ranks_2:
                0-based ranks from the second ranker (best = 0).
            coeff:
                RRF smoothing constant ``k``; larger flattens the contribution
                of top ranks.
            weight_1:
                vote weight of ``ranks_1``.
            weight_2:
                vote weight of ``ranks_2``. Plain RRF (both weights ``1.0``)
                gives each ranker an equal say, which is only sound when *both*
                input rankings are individually trustworthy. Here the AI centroid
                ranker is not: on this small, domain-specific corpus it
                confidently places off-topic documents in its own top-10
                whenever a query word is semantically generic (e.g. "waterfall"
                pulling in paintings/3D-renders, "backup" pulling in a generic
                encyclopedia article), and it simultaneously *buries* canonical
                but short/link-heavy pages whose centroid is diluted toward the
                corpus mean. Down-weighting its vote lets BM25 — the
                higher-precision lexical signal for keyword queries — own the
                top of the ranking while the AI ranker re-orders within the
                lexically-supported set. See [core.search.Indexer.rank][].
        """
        return weight_1 / (coeff + ranks_1) + weight_2 / (coeff + ranks_2)


    @timeit()
    def rank(self, db: sqlite3.Connection, tokens: list[str], 
             method: search_methods,
             n_results: int = 500, fine_search: bool = False,
             sql_query: str = "", sql_params: list[str] = [],
             ai_weight: float = 0.33) -> list[tuple[int, str, float]]:
        """Apply a label on a post based on the trained model.

        Arguments:
            db: 
                the SQLite database holding the indexed set of document. This database must absolutely be up-to-date
                with the one used to instanciate this class, regarding row ordering of documents,
                otherwise rowid mismatches are to be expected between fuzzy, AI and regex searches.
                tokens: the tokenized query.

            method: 
                `ai`, `fuzzy` or `grep`:
                    - `ai` use word embedding and meta-tokens with dual-embedding space, 
                    - `fuzzy` uses meta-tokens with BM25Okapi stats model, 
                    - `mixed` use a combination of `ai` and `fuzzy` merged by
                      Reciprocal Rank Fusion, using the `ai_weight` factor.

            n_results: 
                number of results to retain

            fine_search: 
                optionally refine the search using a 2D interaction matrix. See [1]

            sql_query: 
                SQL query to narrow-down the search, for example `WHERE field = value`. Supports PCRE regex with `WHERE field REGEXP 'pattern'`.
            
            sql_params: 
                the SQL parameters such that:
                ```python
                    cursor = db.execute(
                    f"SELECT url FROM pages {sql_query}",
                    sql_params
                )
                ```
                where each `sql_params` item is matched in the `sql_query` by a `?`. For example:
                ```sql
                    SELECT url              // imposed by the search API
                    FROM pages              // imposed by the search API
                    WHERE instr(url, ?) > 0 // implementation-side `sql_query`
                    ORDER BY url            // imposed by the search API
                ```
                and `sql_params = ['google.com']` will filter all URLs from Google.

            ai_weight:
                vote weight of the AI (embedding) ranker in the weighted RRF
                fusion with BM25+ (which keeps weight 1.0), for `method=MIXED`.
                - ``0.0`` effectively disables the AI part and is equivalent to `method=FUZZY`.
                - ``< 0.5`` makes BM25 the primary signal and lets the noisier
                centroid ranker only re-order within lexically-supported
                candidates.
                - ``0.33`` is the tuned default (drives top-10 junk to
                zero); 
                - ``0.5`` uses plain symmetric RRF: AI and FUZZY contribute as much
                - ``1.0` effectively disables the FUZZY part and is equivalent to `method=AI`.

        Note:
            Both SQL search into the database and Python filtering into the index are supported,
            and can be combined. The local index is a partial copy of the database and is already
            a Python object, so it will be faster to filter if you only need to parse the copied data
            to filter in/out.

        Returns:
            list: the list of best-matching results as (rank, url, similarity) tuples.

        [1]: https://eng.aurelienpierre.com/2024/03/designing-an-ai-search-engine-from-scratch-in-the-2020s/#accounting-for-words-patterns
        """
        
        # Weighting the AI to 0 effectively removes them from ranking,
        # in this case, spare the matrix product.
        if ai_weight == 0 and search_methods.MIXED:
            method = search_methods.FUZZY
        elif ai_weight == 1 and search_methods.MIXED:
            method = search_methods.AI

        # Note: match needs at least Python 3.10
        match method:
            case search_methods.MIXED:
                # Hybrid retrieval: fuse the dual-embedding cosine ranking with
                # BM25+ via *weighted* Reciprocal Rank Fusion. RRF is rank-based,
                # so a document BM25 ranks highly surfaces even when the AI path
                # misses it (diluted long-doc centroid, or cluster-gated out).
                #
                # The AI vote is down-weighted (ai_weight < 1) on purpose: the
                # centroid ranker, on this small domain corpus, confidently puts
                # off-topic documents in its own top-10 for semantically generic
                # query words, and buries canonical short/link-heavy pages whose
                # centroid is diluted toward the corpus mean. Equal-weight RRF
                # therefore let lexically-unsupported junk ride to the top while
                # pushing the right pages down. BM25 is the higher-precision
                # signal for keyword queries, so it owns the top of the ranking
                # and the AI ranker re-orders within the lexically-supported set.
                # Empirically, ai_weight=0.5 drives top-10 junk (results with no
                # BM25 support) to zero without collapsing into pure BM25.
                ai_ranks = self._scores_to_ranks(self.rank_ai(tokens))
                bm_ranks = self._scores_to_ranks(self.rank_fuzzy(tokens))
                aggregates = self.rrf(ai_ranks, bm_ranks, weight_1=ai_weight, weight_2=1.0 - ai_weight)
            case search_methods.AI:
                aggregates = self.rank_ai(tokens)
            case search_methods.FUZZY:
                aggregates = self.rank_fuzzy(tokens)
            case _:
                raise ValueError("Unknown ranking method (%s)" % method)
            
        # Normalize to [0, 1] for stable, legible display scores.
        peak = aggregates.max()
        if peak > 0:
            aggregates = aggregates / peak

        # O(n) partition to isolate the top-n_results candidates, then O(k log k) sort on
        # just that small slice — much cheaper than a full O(n log n) argsort.
        n_results = min(n_results, aggregates.size - 1)
        best_indices = np.argpartition(aggregates, -n_results)[-n_results:]
        best_indices = best_indices[np.argsort(aggregates[best_indices])[::-1]]
        # best_indices is now sorted descending by relevance score.
    
        if sql_query != "":
            sql_hits = self.filter_contents(
                db, sql_query, sql_params, candidate_indices=best_indices
            )
            # assume_unique=True: argpartition over a flat array guarantees unique indices,
            # so NumPy can skip an internal hash/sort pass — roughly halves np.isin cost.
            best_indices = best_indices[np.isin(best_indices, sql_hits, assume_unique=True)]

        # Fetch URLs for the top-k results in one SQL round-trip.
        # O(k · log N) with idx_pages_search_rowid — far cheaper than
        # keeping all N URLs in RAM.  Chunked to respect the variable limit.
        # Restrict to the live searchable subset: after an INCREMENTAL update, positions whose row
        # was deleted or demoted out of the index (in_index 1→0) since the last full build linger in
        # the arrays as "holes". They resolve to no searchable URL here and are simply skipped, so a
        # stale vector can never surface a result. (A periodic full rebuild reclaims the holes.)
        subset = _subset_clause(db)
        best_indices_list = best_indices.tolist()
        rowid_to_url: dict[int, str] = {}
        for start in range(0, len(best_indices_list), 900):
            chunk = best_indices_list[start : start + 900]
            ph = ",".join("?" * len(chunk))
            rowid_to_url.update(db.execute(
                f"SELECT search_rowid, url FROM pages WHERE search_rowid IN ({ph}) AND {subset}",
                chunk,
            ).fetchall())

        best_indices_list = [i for i in best_indices_list if i in rowid_to_url]
        best_elems  = [rowid_to_url[i] for i in best_indices_list]
        best_scores = aggregates[best_indices_list] if best_indices_list else aggregates[:0]

        if self.collocations and len(tokens) > 2 and fine_search:
            indexed_query = self.word2vec.tokens_to_indices(tokens)
            ranked = self.find_query_pattern(
                indexed_query,
                zip(best_indices_list, best_elems, best_scores.tolist()),
            )
            return sorted(ranked, key=lambda x: x[2], reverse=True)

        # Already sorted descending from argsort above — no re-sort needed.
        return list(zip(best_indices_list, best_elems, best_scores.tolist()))


    def get_related(self, tokens: list[str], n: int = 15, k: int = 5, use_sif: bool = True, sif_smoothing: float = 1e-3) -> list:
        """Get the n closest keywords from the query."""

        vector = self.word2vec.get_features(tokens, use_sif=use_sif, sif_smoothing=sif_smoothing)

        # wv.similar_by_vector returns a list of (word, distance) tuples
        from_query = [elem for elem in self.word2vec.wv.similar_by_vector(vector, topn=n)]
        from_tokens = [elem for token in tokens for elem in self.word2vec.wv.most_similar(token, topn=k)]

        # sort by relevance
        related = sorted(from_query + from_tokens, key=lambda x:x[1], reverse=True)

        return list(set([elem[0] for elem in related if elem[0] not in tokens]))


    @timeit()
    def get_clusters(self, db: sqlite3.Connection):
        """Find document latent topics modelled as clusters of document centroids.

        Writes to the database:
            - `clusters` table   — one row per cluster: label (PK), human-legible
                                   keyword labels, centroid BLOB, and max cosine
                                   radius so callers can gauge cluster tightness.
            - `pages.cluster`    — integer FK into `clusters.label` for each page.
            - `search` table     — three new BLOB columns (`cluster_labels_raw`,
                                   `cluster_centroids_raw`, `cluster_centroids_shape`)
                                   that mirror the pattern used for `vectors_raw` so
                                   the data loads at the same speed on startup without
                                   touching `pages` at all.

        Sets on self (immediately usable without reloading):
            self.cluster_centroids:   (K, D) float32 — one centroid per cluster.
            self.cluster_doc_indices: dict[int, np.ndarray[int32]] — maps each
                                      cluster label to its member row indices in
                                      `self.vectors` / `self.ranker` (positions = search_rowid).
        """

        # 1. Cluster document vectors.
        #
        # Stability measures (clusters must stay coherent run-to-run and across
        # corpus updates):
        #   - cluster on a PCA-denoised projection. The trailing low-variance
        #     dimensions are mostly noise and are the main source of assignment
        #     jitter; dropping them makes k-means far more reproducible.
        #   - run several inits and keep the best inertia (n_init).
        # Centroids are then recomputed in the FULL embedding space from the
        # labels (not taken from the reduced-space k-means centers), so
        # _cluster_candidate_indices() can keep dotting them against full-size
        # query vectors at search time.
        num_cpu = os.cpu_count() or 1
        n_clusters = max(2, int(self.pages / 200))

        n_pca = min(64, self.vectors.shape[1], self.vectors.shape[0])
        reducer = PCA(n_components=n_pca, random_state=0)
        reduced = reducer.fit_transform(self.vectors)

        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=512 * num_cpu,
            n_init=10,
            max_iter=300,
            random_state=0,
        )
        labels = kmeans.fit_predict(reduced)           # shape (N,)
        unique_labels = np.unique(labels)              # present (non-empty) labels

        # 2. Associate docs with their clusters now, speed things up later
        self.cluster_doc_indices: dict[int, np.ndarray] = {
            int(l): np.where(labels == l)[0].astype(np.int32)
            for l in unique_labels
        }

        # 3. Per-cluster centroids in the ORIGINAL embedding space: the
        # normalised mean of member vectors (which are already L2-normalised).
        # Built positionally from cluster_doc_indices so row i of
        # cluster_centroids matches the i-th key of cluster_doc_indices, which
        # is the ordering _cluster_candidate_indices() relies on.
        self.cluster_centroids = np.array([
            self.vectors[idx].mean(axis=0)
            for idx in self.cluster_doc_indices.values()
        ], dtype=np.float32)
        self.cluster_centroids /= (
            np.linalg.norm(self.cluster_centroids, axis=1, keepdims=True) + 1e-8
        )

        # Human-legible keywords: the 5 vocabulary tokens whose input embedding
        # is most similar to the cluster centroid direction.
        #for i, c in enumerate(self.cluster_centroids):
        #    print(f"cluster {i}/{n_clusters} :", [word for word, _ in self.word2vec.wv.similar_by_vector(c, topn=5)])


    @timeit()
    def update_incremental(self, db: sqlite3.Connection, name: str) -> int:
        """Append newly-added documents to an EXISTING index without recomputing everything —
        the daily/server-side counterpart to the full ``Indexer(db, …)`` build.

        The "diff" is exactly the searchable rows that do not yet have a ``search_rowid`` (freshly
        crawled/merged pages; a full build assigns 0…N-1, so anything NULL is new). For those rows
        only, this:
          * appends their postings to the BM25 CSR (``ranker.add_documents`` — no full re-read);
          * projects their vectors with the STORED principal component(s) (``self.pc`` reused, PCA
            NOT refit) and appends them to ``self.vectors``;
          * assigns ``search_rowid = N, N+1, …`` (append, never renumbers the existing rows, so the
            pickled arrays stay aligned);
          * assigns each new doc to the NEAREST EXISTING K-means centroid (``self.cluster_centroids``
            reused, K-means NOT refit) and extends ``cluster_doc_indices``;
          * refreshes the dashboard stats + the integrity fingerprint, and re-saves the artifact.

        Cost scales with the delta, not the corpus. Documents that LEFT the subset (in_index 1→0),
        were deleted, or were replaced by a re-crawl (a new row is appended; the old one is dropped by
        dedup) leave "holes": stale positions that stay in the arrays until the next full rebuild.
        They are harmless — ``rank()`` restricts its URL fetch to the live searchable subset, so a
        hole resolves to nothing and is skipped — they only bloat the index until a periodic full
        rebuild reclaims them. Returns the number of documents appended.
        """
        subset = _subset_clause(db)

        # The pickled vectors occupy positions 0..base-1, so new rows must append at `base`. Rows
        # deleted or demoted since the last build leave holes (positions still in the arrays, no
        # live row) — those are tolerated: rank() skips them. The one thing that must hold is that
        # NO existing search_rowid is >= base, otherwise appending at `base` would collide.
        base = int(self.vectors.shape[0])
        max_sr = db.execute("SELECT MAX(search_rowid) FROM pages").fetchone()[0]
        if max_sr is not None and max_sr >= base:
            raise RuntimeError(
                f"Index/DB misaligned: found search_rowid {max_sr} >= {base} pickled vectors. "
                f"Run a FULL rebuild (chantal-05 without --incremental)."
            )

        rows = db.execute(
            f"SELECT rowid, stemmed, vectorized FROM pages "
            f"WHERE {subset} AND search_rowid IS NULL ORDER BY rowid"
        ).fetchall()
        if not rows:
            print("Incremental index: no new documents — nothing to do.")
            return 0

        key_to_index = self.word2vec.wv.key_to_index
        new_corpus = [
            [key_to_index[w] for sentence in stemmed for w in sentence if w in key_to_index]
            for (_, stemmed, _) in rows
        ]
        new_vecs = np.array([v for (_, _, v) in rows], dtype=np.float32)

        # 1. BM25 append (reuses existing postings; IDF/avgdl recomputed from updated frequencies).
        self.ranker.add_documents(new_corpus)

        # 2. Vectors: project with the EXISTING principal component(s), append (PCA not refit).
        self.vectors = np.vstack([self.vectors, self.normalize_pc(new_vecs)])

        # 3. Assign search_rowid = base, base+1, … (append; existing rows keep their ids).
        db.executemany(
            "UPDATE pages SET search_rowid = ? WHERE rowid = ?",
            [(base + j, rows[j][0]) for j in range(len(rows))],
        )
        db.commit()

        # 4. Assign each new doc to the nearest EXISTING cluster centroid (K-means not refit).
        centroids = getattr(self, "cluster_centroids", None)
        if centroids is not None and len(centroids):
            keys = list(self.cluster_doc_indices.keys())
            nearest = (self.vectors[base:] @ centroids.T).argmax(axis=1)
            for j, pos in enumerate(nearest):
                k = keys[int(pos)]
                self.cluster_doc_indices[k] = np.append(
                    self.cluster_doc_indices[k], np.int32(base + j)
                )

        # 5. Refresh stats + integrity fingerprint, then re-save the artifact.
        self.stats = self.build_stats(db)
        self.pages = self.stats["pages"]
        self.words = self.stats["words"]
        self.save_search_stats(db, self.stats)
        self.save_categories_index(db, self.stats["category_counts"])
        self.index_fingerprint = self._compute_fingerprint(db)
        self.save(name)

        print(f"Incremental index: appended {len(rows)} documents ({base} → {self.vectors.shape[0]}).")
        return len(rows)


    def compute_ctfidf_labels(self, labels: np.ndarray, top_n: int = 10) -> dict[int, list[str]]:
        """
        Compute c-TF-IDF topic keywords for each cluster using the existing BM25+ ranker.

        Returns a dict mapping cluster label → list of top_n discriminative keywords.
        """
        unique_labels = [l for l in np.unique(labels) if l != -1]   # skip noise
        n_clusters = len(unique_labels)
        label_to_pos = {l: i for i, l in enumerate(unique_labels)}
        vocab_size = len(self.ranker.indptr) - 1

        # Map each document to its cluster position (-1 = noise, excluded)
        doc_to_pos = np.full(self.ranker.corpus_size, -1, dtype=np.int32)
        for l in unique_labels:
            doc_to_pos[labels == l] = label_to_pos[l]

        # Reconstruct token_id for every posting from the CSR indptr
        # indptr[t+1] - indptr[t] = number of postings for token t
        token_ids = np.repeat(
            np.arange(vocab_size, dtype=np.int32),
            np.diff(self.ranker.indptr)
        )                                                    # (n_postings,)

        # Assign each posting to a cluster position
        posting_cluster_pos = doc_to_pos[self.ranker.doc_ids]   # (n_postings,)

        # Keep only postings that belong to a real cluster (not noise)
        valid = posting_cluster_pos >= 0

        # Build sparse (vocab_size × n_clusters) TF matrix
        tf_matrix = csr_matrix(
            (
                self.ranker.tfs[valid].astype(np.float32),
                (token_ids[valid], posting_cluster_pos[valid]),
            ),
            shape=(vocab_size, n_clusters),
        )

        # Normalise each cluster column by its document count
        cluster_sizes = np.array(
            [(labels == l).sum() for l in unique_labels], dtype=np.float32
        )
        tf_norm = tf_matrix.multiply(1.0 / cluster_sizes[np.newaxis, :])

        # IDF across clusters: how many clusters contain this token at all
        cluster_df = np.diff(tf_matrix.indptr) if tf_matrix.format == "csc" \
                    else (tf_matrix > 0).sum(axis=1).A1      # (vocab_size,)
        idf = np.log(1.0 + n_clusters / (cluster_df + 1.0)).astype(np.float32)

        # c-TF-IDF = normalised TF × IDF
        ctfidf = tf_norm.multiply(idf[:, np.newaxis])         # sparse broadcast

        # Extract top_n tokens per cluster
        wv = self.word2vec.wv
        topic_labels = {}
        for i, l in enumerate(unique_labels):
            col = ctfidf.getcol(i).toarray().ravel()          # (vocab_size,)
            top_token_ids = np.argpartition(col, -top_n)[-top_n:]
            top_token_ids = top_token_ids[np.argsort(col[top_token_ids])[::-1]]
            topic_labels[int(l)] = [
                wv.index_to_key[t]
                for t in top_token_ids
                if t < len(wv.index_to_key)
            ]

        return topic_labels