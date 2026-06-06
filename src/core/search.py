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


from .utils import get_models_folder, timeit
from .nlp import Lexicon, Word2Vec
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
    FUZZY = 2

class Indexer():
    @timeit()
    def __init__(self,
                 db: sqlite3.Connection,
                 name: str,
                 word2vec: Word2Vec,
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

        self.word2vec: Word2Vec = word2vec
        """Word2Vec embedding language model"""
        
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
        cursor = db.execute("SELECT stemmed FROM pages ORDER BY url")
        rows = cursor.fetchall()

        # To spare some memory, build a symbolic corpus representation using
        # word indices in the Word2Vec vocabulary, then construct a local
        # BM25Plus reimplementation that precomputes freqs/lengths/inverted index.
        corpus_token_indices = [
            [
                self.word2vec.wv.key_to_index[word]
                for sentence in doc[0]
                for word in sentence
                if word in self.word2vec.wv.key_to_index
            ]
            for doc in rows
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

        cursor = db.execute("SELECT vectorized FROM pages ORDER BY url")

        self.vectors = np.array([item[0] for item in cursor.fetchall()], dtype=np.float32)
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
        Add persistent indexes for the user-facing search filters, and
        ensure the ``search_rowid`` column exists.

        ``search_rowid`` is an explicit INTEGER column that we assign to the
        0-based position of each page in ``ORDER BY url`` order when the
        Indexer is (re)built.  Because it is a real column value, ``VACUUM``
        cannot renumber it — unlike SQLite's implicit rowid on tables with a
        TEXT primary key.  This column replaces the in-memory
        ``self.index`` / ``self.url_to_index`` / ``self._index_arr`` LUTs.
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

        pages = cursor.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        domains = cursor.execute("""
            SELECT COUNT(DISTINCT COALESCE(NULLIF(domain, ''), url))
            FROM pages
        """).fetchone()[0]
        categories = cursor.execute("""
            SELECT COUNT(DISTINCT category)
            FROM pages
            WHERE category IS NOT NULL
              AND category != ''
        """).fetchone()[0]
        most_recent_datetime = cursor.execute("""
            SELECT MAX(datetime)
            FROM pages
            WHERE datetime IS NOT NULL
        """).fetchone()[0]
        oldest_datetime = cursor.execute("""
            SELECT MIN(datetime)
            FROM pages
            WHERE datetime IS NOT NULL
        """).fetchone()[0]
        length_stats = cursor.execute("""
            SELECT COALESCE(SUM(length), 0),
                   COALESCE(AVG(length), 0),
                   COALESCE(MAX(length), 0)
            FROM pages
            WHERE length IS NOT NULL
        """).fetchone()
        domain_counts = cursor.execute("""
            SELECT COALESCE(NULLIF(domain, ''), url) AS domain,
                   COUNT(*) AS pages
            FROM pages
            GROUP BY COALESCE(NULLIF(domain, ''), url)
            ORDER BY pages DESC, domain ASC
        """).fetchall()
        category_counts = cursor.execute("""
            SELECT COALESCE(NULLIF(category, ''), '(none)') AS category,
                   COUNT(*) AS pages
            FROM pages
            GROUP BY COALESCE(NULLIF(category, ''), '(none)')
            ORDER BY pages DESC, category ASC
        """).fetchall()
        language_counts = cursor.execute("""
            SELECT COALESCE(NULLIF(lang, ''), '(unknown)') AS lang,
                   COUNT(*) AS pages
            FROM pages
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
        """Assign ``search_rowid = 0, 1, 2, …`` to every page in ``ORDER BY url``.

        This is the single source of truth that glues the DB to the in-RAM
        numpy arrays (``self.vectors``, ``self.ranker``).  Both are built by
        reading pages ``ORDER BY url``, so position 0 in every array
        corresponds to ``search_rowid = 0``, and so on.

        Run once per Indexer build; the values survive ``VACUUM`` because
        they live in a real column, not in SQLite's internal b-tree position.

        Also stores a two-point fingerprint (page count + boundary URL hash)
        in ``self.index_fingerprint`` so ``verify_db_integrity()`` can detect
        invalidating changes in O(1) at load time.
        """
        cursor = db.execute("SELECT url FROM pages ORDER BY url")
        urls = [row[0] for row in cursor.fetchall()]

        db.executemany(
            "UPDATE pages SET search_rowid = ? WHERE url = ?",
            enumerate(urls),
        )
        db.commit()

        # Cheap two-point fingerprint: count + hash of first + last URL.
        # Catches all insertions, deletions, and URL edits at the boundaries.
        # Middle-URL mutations without a count change are rare in practice;
        # callers who need stronger guarantees can call verify_db_integrity()
        # with full=True (see below) before each query session.
        import hashlib
        boundary = "".join([
            urls[0]  if urls else "",
            urls[-1] if urls else "",
        ]).encode()
        self.index_fingerprint = (len(urls), hashlib.sha256(boundary).hexdigest())


    def verify_db_integrity(self, db: sqlite3.Connection, full: bool = False):
        """Raise ``RuntimeError`` if the DB has changed since this Indexer was built.

        Arguments:
            full:
                If ``False`` (default), checks page count + a hash of the
                first and last URL in search_rowid order — three O(log N)
                queries, effectively O(1).  Catches all insertions, deletions,
                and boundary-URL edits.
                If ``True``, hashes every URL in rowid order — O(N) but
                detects any mid-corpus reordering or URL mutation.

        Called automatically by :meth:`load`.
        """
        import hashlib
        stored_count, stored_hash = self.index_fingerprint

        # Always check count first — cheapest possible signal.
        current_count = db.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
        if current_count != stored_count:
            raise RuntimeError(
                f"Page count changed since Indexer was built "
                f"({stored_count} → {current_count}). Rebuild with Indexer(db, ...)."
            )

        if full:
            # Hash every URL in order — catches any mid-corpus change.
            h = hashlib.sha256()
            for (url,) in db.execute("SELECT url FROM pages ORDER BY search_rowid"):
                h.update(url.encode())
            current_hash = h.hexdigest()
        else:
            # Hash only the boundary URLs — three index-only lookups.
            first = db.execute(
                "SELECT url FROM pages ORDER BY search_rowid ASC  LIMIT 1"
            ).fetchone()
            last  = db.execute(
                "SELECT url FROM pages ORDER BY search_rowid DESC LIMIT 1"
            ).fetchone()
            boundary = (
                (first[0] if first else "") + (last[0] if last else "")
            ).encode()
            current_hash = hashlib.sha256(boundary).hexdigest()

        if current_hash != stored_hash:
            raise RuntimeError(
                "DB page ordering has changed since this Indexer was built. "
                "Rebuild with Indexer(db, ...)."
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
        """Filter pages by arbitrary SQL queries, returning ``search_rowid`` integers.

        With ``candidate_indices`` supplied, the query is scoped to that small set
        via a VALUES CTE — no URL ↔ index conversion, pure integer joins.
        Without ``candidate_indices``, the full table is scanned.

        The returned integers map directly to numpy array positions in
        ``self.vectors`` and ``self.ranker``.
        """
        if sql_params is None:
            sql_params = []

        if candidate_indices is not None:
            # Keep a comfortable margin below the historical 999-variable limit.
            max_chunk = max(1, 900 - len(sql_params))
            matched: list[int] = []
            indices_list = [int(i) for i in candidate_indices]

            for start in range(0, len(indices_list), max_chunk):
                chunk = indices_list[start : start + max_chunk]
                placeholders = ", ".join(["(?)" for _ in chunk])
                # Integer CTE — no URL strings, no back-and-forth conversion.
                # CROSS JOIN forces SQLite to drive from the tiny candidate list.
                query = f"""
                    WITH candidates(rid) AS (VALUES {placeholders})
                    SELECT pages.search_rowid
                    FROM candidates
                    CROSS JOIN pages ON pages.search_rowid = candidates.rid
                    {sql_query}
                """
                cursor = db.execute(query, [*chunk, *sql_params])
                matched.extend(row[0] for row in cursor.fetchall())

            return matched

        # No candidate set: scan the full table.
        query = f"SELECT search_rowid FROM pages {sql_query} ORDER BY search_rowid"
        cursor = db.execute(query, sql_params)
        return [row[0] for row in cursor.fetchall()]


    def save(self, name: str):
        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"), compress=0, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    @timeit()
    def load(cls, name: str, db: sqlite3.Connection):
        """Load an existing trained model by its name from the `../models` folder."""
        try:
            model = joblib.load(get_models_folder(name) + ".joblib")
        except FileNotFoundError:
            model = joblib.load(get_models_folder(name) + ".joblib.bz2")
            
        if not isinstance(model, Indexer):
            raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))

        # Guard against a DB that has been modified (new crawl, deletion, VACUUM
        # after schema migration) since this Indexer was last built.
        # O(1) by default; pass full=True for a thorough check.
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


    def vectorize_query(self, tokenized_query: list[str]) -> np.ndarray:
        """Prepare a text search query: cleanup, tokenize and get the centroid vector.

        Returns:
            tuple[vector, norm, tokens]
        """

        # Get the the centroid of the word embedding vector
        vector = self.word2vec.get_features(tokenized_query, embed="IN", use_sif=True)

        # Remove the principal component from the vector
        vector = self.normalize_pc(vector)

        return vector


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
    def rank_ai(self, tokens: list[str], fast: bool = False, clip: bool = True, n_clusters: int = 3) -> np.ndarray:
        """Cosine-similarity ranking against document centroid vectors.

        Arguments:
            tokens:     tokenised query (output of ``tokenize_query``).
            fast:       use a single dot-product against the aggregate query
                        vector instead of the per-token dual-embedding loop.
            clip:       clamp scores to [0, 1] (recommended for RRF blending).
            n_clusters: if > 0 **and** cluster data has been loaded, restrict
                        the matmul to the documents that belong to the
                        ``n_clusters`` closest clusters.  Documents outside
                        the selected clusters keep a score of 0, so the
                        BM25+ component of the blend can still surface them.
                        Set to 0 (default) to score the full corpus.
        """

        if fast:
            # The following seems very close to the next in terms of results.
            # Experimentally, I saw very little difference in rankings, at least not in the first results.
            # The by-the-book is perhaps more immune to keywords stuffing and more sensitive to structure.
            # Differences appear in the tail of the ranking, mostly.
            # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
            query_vec = self.vectorize_query(tokens)

            if self.cluster_centroids is not None:
                candidate_indices = self._cluster_candidate_indices(query_vec, n_clusters=n_clusters)
                if candidate_indices:
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
                mean_vec = self.vectorize_query(tokens)
                candidate_indices = self._cluster_candidate_indices(mean_vec, n_clusters=n_clusters)

            for token in tokens:
                # Compute the cosine similarity of centroids between query and documents,
                # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
                vector = self.word2vec.get_wordvec(token, embed="IN", normalize=True)
                if vector is not None:
                    vector = self.normalize_pc(vector)
                    if candidate_indices is not None:
                        aggregate[candidate_indices] += np.nan_to_num(
                            self.vectors[candidate_indices] @ vector
                        )
                    else:
                        aggregate += np.nan_to_num(np.dot(self.vectors, vector))
                    weights += 1.
                else:
                    print(f"token {token} was not found in embedding vocabulary")

            # Cosine similarity is bounded in [-1; 1].
            # 0 means unrelated or orthogonal.
            # Negative means we are in the opposite direction to the request.
            # In that case, let BM25+ weighting take over and don't over-penalize
            # results.
            if clip:
                return np.clip(aggregate / weights if weights > 0. else aggregate, 0., 1.)
            else:
                return aggregate / weights if weights > 0. else aggregate

    @timeit()
    def _cluster_candidate_indices(self, query_vec: np.ndarray, n_clusters: int = 3) -> np.ndarray | None:
        """Return the row indices of all documents in the ``n_clusters`` nearest
        clusters to ``query_vec``.

        The centroid comparison is a (K x D) matmul where K <= 500, adding
        ~0.1 ms overhead while reducing the subsequent per-document matmul from
        N rows to roughly ``N / K * n_clusters`` rows.

        Arguments:
            query_vec:  normalised query vector, shape (D,).
            n_clusters: number of nearest clusters to include.
        """
        # Score every cluster centroid against the query.
        # Centroids are unit-normalised (mean of normalised vectors, then
        # re-normalised in get_clusters), so dot product == cosine similarity.
        cluster_similarity = self.cluster_centroids @ query_vec
        cluster_indices = np.argpartition(cluster_similarity, -n_clusters)[-n_clusters:]

        # cluster_doc_indices keys are ordered identically to centroid rows
        # (both built from the same np.unique(labels) pass).
        labels_ordered = list(self.cluster_doc_indices.keys())
        return np.concatenate([
            self.cluster_doc_indices[labels_ordered[pos]]
            for pos in cluster_indices
        ])


    def rrf(self, ranks_1: np.ndarray, ranks_2: np.ndarray, coeff: float = 60) -> np.ndarray:
        """Reciprocal Rank Fusion
        
        Aggregate 2 sets of page rankings obtained from different semantic geometries and weighted differently.

        From _Reciprocal rank fusion outperforms condorcet and individual rank learning methods_,
        Gordon V. Cormack, Charles L A Clarke, Stefan Buettcher.
        https://dl.acm.org/doi/10.1145/1571941.1572114
        """
        return 1. / (coeff + ranks_1) + 1. / (coeff + ranks_2)


    @timeit()
    def rank(self, db: sqlite3.Connection, tokens: list[str], method: search_methods,
             n_results: int = 500, fine_search: bool = False, 
             sql_query: str = "", sql_params: list[str] = []) -> list[tuple[int, str, float]]:
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
                    - `grep` uses direct string and regex search.

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

        Note:
            Both SQL search into the database and Python filtering into the index are supported,
            and can be combined. The local index is a partial copy of the database and is already
            a Python object, so it will be faster to filter if you only need to parse the copied data
            to filter in/out.

        Returns:
            list: the list of best-matching results as (rank, url, similarity) tuples.

        [1]: https://eng.aurelienpierre.com/2024/03/designing-an-ai-search-engine-from-scratch-in-the-2020s/#accounting-for-words-patterns
        """

        # Note: match needs at least Python 3.10
        match method:
            case search_methods.AI:
                # Aggregate vector embedding method with the ranking from BM25+ to it for each URL.
                # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
                # RRF works very poorly here, so we do "alpha blending".
                aggregates = 0.98 * self.rank_ai(tokens) + 0.02 * self.rank_fuzzy(tokens)
            case search_methods.FUZZY:
                aggregates = self.rank_fuzzy(tokens)
            case _:
                raise ValueError("Unknown ranking method (%s)" % method)

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

        # Fetch URLs for the top-k results in one SQL query.
        # O(k · log N) with idx_pages_search_rowid — far cheaper than loading
        # all N URLs into RAM.  Chunked to respect SQLite's variable limit.
        best_indices_list = best_indices.tolist()
        rowid_to_url: dict[int, str] = {}
        for start in range(0, len(best_indices_list), 900):
            chunk = best_indices_list[start : start + 900]
            ph = ",".join("?" * len(chunk))
            rowid_to_url.update(db.execute(
                f"SELECT search_rowid, url FROM pages WHERE search_rowid IN ({ph})",
                chunk,
            ).fetchall())

        # Preserve relevance order; a KeyError here means a page was deleted
        # after the last Indexer build — verify_db_integrity() guards against
        # this in production.
        best_elems  = [rowid_to_url[i] for i in best_indices_list]
        best_scores = aggregates[best_indices]

        if self.collocations and len(tokens) > 2 and fine_search:
            indexed_query = self.word2vec.tokens_to_indices(tokens)
            ranked = self.find_query_pattern(
                indexed_query,
                zip(best_indices_list, best_elems, best_scores.tolist()),
            )
            return sorted(ranked, key=lambda x: x[2], reverse=True)

        # Data is already sorted descending from the argsort above.
        return list(zip(best_indices_list, best_elems, best_scores.tolist()))


    def get_related(self, tokens: list[str], n: int = 15, k: int = 5) -> list:
        """Get the n closest keywords from the query."""

        vector = self.word2vec.get_features(tokens)

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
                                      `self.vectors` / `self.ranker` (i.e. ``search_rowid`` positions).
        """

        # 1. Cluster document vectors
        num_cpu = os.cpu_count() or 1
        n_clusters = int(self.pages / 200)
        birch = MiniBatchKMeans(n_clusters=n_clusters, batch_size=512 * num_cpu, random_state=0)
        labels = birch.fit_predict(self.vectors)       # shape (N,), dtype int32/int64
        unique_labels = np.unique(labels)              # shape (K,)

        # 2. Associate docs with their clusters now, speed things up later
        self.cluster_doc_indices: dict[int, np.ndarray] = {
            int(l): np.where(labels == l)[0].astype(np.int32)
            for l in unique_labels
        }

        # 2. Derive per-cluster statistics
        # Centroid: mean of all member vectors (which are already L2-normalised).
        # Re-normalising the mean gives the "direction" of the cluster.
        self.cluster_centroids = birch.cluster_centers_.astype(np.float32)
        self.cluster_centroids /= (
            np.linalg.norm(self.cluster_centroids, axis=1, keepdims=True) + 1e-8
        )

        # Max cosine *distance* from centroid — 0 means perfectly tight,
        # 2 means perfectly spread.  Because vectors are unit-normalised,
        # cosine similarity = dot product, so distance = 1 − similarity.
        self.max_radii = np.array([
            1.0 - (self.vectors[idx] @ self.cluster_centroids[i]).min()
            for i, idx in self.cluster_doc_indices.items()
        ], dtype=np.float32)                           # shape (K,)

        # Human-legible keywords: the 5 vocabulary tokens whose input embedding
        # is most similar to the cluster centroid direction.
        for i, c in enumerate(self.cluster_centroids):
            print(f"cluster {i}/{n_clusters} :", [word for word, _ in self.word2vec.wv.similar_by_vector(c, topn=5)])
            

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