from scipy.signal import convolve2d
from enum import IntEnum

import sqlite3
import joblib
import pickle
import numpy as np
import regex as re

from rank_bm25 import BM25Plus
from collections import Counter
from sklearn.decomposition import PCA

from .utils import get_models_folder, timeit
from .nlp import Word2Vec
    
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
    SEARCH_TABLE_COLUMNS = {
        "id": "INTEGER PRIMARY KEY",
        "doc_index": "list",
        "vectors": "array",
        "bm25_k1": "REAL",
        "bm25_b": "REAL",
        "bm25_delta": "REAL",
        "bm25_corpus_size": "INTEGER",
        "bm25_avgdl": "REAL",
        "bm25_doc_lens": "array",
        "bm25_denom_const": "array",
        "bm25_idf": "array",
        "bm25_doc_ids": "array",
        "bm25_tfs": "array",
        "bm25_indptr": "array",
        "w2v_index_to_key": "list",
        "w2v_vectors": "array",
        "w2v_syn1": "array",
        "w2v_syn1neg": "array",
    }

    @timeit()
    def __init__(self,
                 db: sqlite3.Connection,
                 name: str,
                 word2vec: Word2Vec,
                 strip_collocations: bool = False,
                 principal_components: int = 1):
        """Search engine based on word similarity.

        Arguments:
            training_set (list): list of Data elements. If the list is empty, it will try to find a pre-trained model matching the `path` name.
            path : path to save the trained model for reuse, as a Python joblib.
            name (str): name under which the model will be saved for la ter reuse.
            word2vec (Word2Vec): the instance of word embedding model.
            strip_collocations: remove the matrix of collocations in documents, which is the list of word tokens represented by their index in the
            word2vec dictionnary. It is used for [core.nlp.Indexer.find_query_patterns][], which is optional and significatively slower
            (but not significatively better), so if you don't plan on using it, removing collocations saves some RAM and I/O.
            principal_components (int): number of principal components to compute and remove from the index dataset. 
            This helps to make queries more selective and specific in the presence of boilerplate text and formatting language in the sampling.

        NOTE:
            The class is optimized to run online, on server: load fast when spawning a new server-side worker,
            use RAM sparingly.
            
        """

        self.sql: str = ""
        """Cache the previous SQL filtering conditions"""

        self.word2vec: Word2Vec = word2vec
        """Word2Vec embedding language model"""
        
        self.init_search_table(db)
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

        ########################################
        # 3. Save heavy numpy arrays to database
        ########################################

        # Save the whole Word2Vec language model and BM25+ ranker constants to database.
        # This has several benefits:
        #   1. the Indexer class, once saved as a joblib/pickled artifact, is much leaner
        #      and faster to decompress and load in RAM on server,
        #   2. we freeze the whole state of the NLP stack (vocabulary, language model, document embeddings)
        #      in a single, consistent place, so there is no version mismatches anymore:
        #      the database is an unit of processing.
        self.save_search_vectors(db, self.vectors) # those have principal components already removed
        self.save_search_word2vec(db)
        self.save_search_ranker(db, self.ranker)

        # Storing a pre-built index list as a single BLOB in database is 2.5 times faster
        # to restore at runtime than unrolling it from SQL `SELECT url FROM pages`.
        self.index: list[str] = self.build_index(db)
        """LUT of document URLs as ordered when building the ranker, lazily loaded from the database."""

        self.url_to_index: dict[str, int] = self.build_index_reverse()
        """Reverse LUT of `self.index`"""

        self.save_search_index(db, self.index)

        ###############
        # 4. Misc stats
        ###############
        self.words = len(word2vec.wv)
        self.pages = self.ranker.corpus_size

        # 5. Save the pickled object to disk for reuse
        self.save(name)


    def __getstate__(self):
        # Remove huge Numpy arrays from the instance before pickling it to save.
        # They were saved in database at __init__()
        # and will be reloaded from there when loading the pickled object
        import copy

        state = self.__dict__.copy()
        word2vec = copy.copy(self.word2vec)

        if hasattr(word2vec, "wv"):
            del word2vec.wv
        if hasattr(word2vec, "syn1"):
            del word2vec.syn1
        if hasattr(word2vec, "syn1neg"):
            del word2vec.syn1neg

        state["word2vec"] = word2vec
        state["db"] = None
        state["index"] = None
        state["vectors"] = None
        state["ranker"] = None
        state["url_to_index"] = None
        return state


    def init_search_table(self, db: sqlite3.Connection):
        """
        Create or migrate the one-row search cache table.
        """
        cursor = db.cursor()

        column_sql = ", ".join(f"{name} {kind}" for name, kind in self.SEARCH_TABLE_COLUMNS.items())
        cursor.execute(f"CREATE TABLE IF NOT EXISTS search ({column_sql})")

        existing_columns = { row[1] for row in cursor.execute("PRAGMA table_info(search)") }

        for name, kind in self.SEARCH_TABLE_COLUMNS.items():
            if name not in existing_columns:
                cursor.execute(f"ALTER TABLE search ADD COLUMN {name} {kind}")

        db.commit()


    def init_pages_search_indexes(self, db: sqlite3.Connection):
        """
        Add persistent indexes for the user-facing search filters.

        The URL primary key already gives us an index for URL lookups. The
        category indexes help equality filters and provide a stable ordering
        path for category-only queries. Substring filters such as
        `NOT LIKE '%github.com%'`, `instr(parsed, ?)`, and `REGEXP` cannot use a
        normal B-tree index, so `filter_contents()` narrows those to ranked
        candidate URLs before SQLite evaluates them.
        """
        cursor = db.cursor()

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_category_url
            ON pages(category, url)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_pages_category_coalesce_url
            ON pages(COALESCE(category, ''), url)
        """)

        db.commit()


    def save_search_values(self, db: sqlite3.Connection, values: dict):
        """
        Store one or more precomputed search cache values in the database.
        """

        unknown_columns = set(values) - set(self.SEARCH_TABLE_COLUMNS)
        if unknown_columns:
            raise ValueError("Unknown search cache columns: %s" % sorted(unknown_columns))

        cursor = db.cursor()
        columns = list(values)
        placeholders = ", ".join(["?" for _ in columns])
        insert_columns = ", ".join(["id"] + columns)
        update_columns = ", ".join(
            f"{column} = excluded.{column}"
            for column in columns
        )

        cursor.execute(f"""
            INSERT INTO search ({insert_columns})
            VALUES (?, {placeholders})
            ON CONFLICT(id) DO UPDATE SET {update_columns}
        """, (1, *[values[column] for column in columns]))

        db.commit()


    def get_search_values(self, db: sqlite3.Connection, columns: list[str]) -> any:
        """
        Fetch one or more precomputed search cache values from the database.
        """

        unknown_columns = set(columns) - set(self.SEARCH_TABLE_COLUMNS)
        if unknown_columns:
            raise ValueError("Unknown search cache columns: %s" % sorted(unknown_columns))

        cursor = db.cursor()
        row = cursor.execute(f"""
            SELECT {", ".join(columns)}
            FROM search
            WHERE id = 1
        """).fetchone()

        if row is None:
            return None

        return row


    def save_search_index(self, db: sqlite3.Connection, index: list[str]):
        """
        Store the URL lookup table into `search.doc_index`.
        """
        self.save_search_values(db, {"doc_index": index})


    def save_search_ranker(self, db: sqlite3.Connection, ranker: BM25PlusCSR):
        """
        Store the BM25+ CSR ranker state into `search`.
        """
        self.save_search_values(db, {
            "bm25_k1": float(ranker.k1),
            "bm25_b": float(ranker.b),
            "bm25_delta": float(ranker.delta),
            "bm25_corpus_size": ranker.corpus_size,
            "bm25_avgdl": float(ranker.avgdl),
            "bm25_doc_lens": ranker.doc_lens,
            "bm25_denom_const": ranker.denom_const,
            "bm25_idf": ranker.idf,
            "bm25_doc_ids": ranker.doc_ids,
            "bm25_tfs": ranker.tfs,
            "bm25_indptr": ranker.indptr,
        })


    def save_search_vectors(self, db: sqlite3.Connection, vectors: np.ndarray):
        """
        Store the single global vector matrix into `search.vectors`.
        """
        self.save_search_values(db, {"vectors": vectors})


    def save_search_word2vec(self, db: sqlite3.Connection):
        """
        Store Word2Vec vocabulary and embedding matrices into the search cache.
        """
        self.save_search_values(db, {
            "w2v_index_to_key": list(self.word2vec.wv.index_to_key),
            "w2v_vectors": self.word2vec.wv.vectors,
            "w2v_syn1": getattr(self.word2vec, "syn1", None),
            "w2v_syn1neg": getattr(self.word2vec, "syn1neg", None),
        })


    @timeit()
    def build_index(self, db: sqlite3.Connection) -> list[str]:
        # We index URLs because they are the database primary key
        # and therefore guaranteed to be constant over time.
        # So the index is a LUT of URLs as they were ordered in DB when reading it.
        cursor = db.execute("SELECT url FROM pages ORDER BY url")
        return [item[0] for item in cursor.fetchall()]
    

    @timeit()
    def build_index_reverse(self) -> dict[str, int]:
        self.url_to_index = {
            url: i
            for i, url in enumerate(self.index)
        }
        return self.url_to_index
    

    @timeit()
    def get_index(self, db: sqlite3.Connection):
        self.index = self.get_search_values(db, ["doc_index"])[0]


    @timeit()
    def get_doc_vectors(self, db: sqlite3.Connection):
        # Note: `nlp.Word2Vec.get_features`` already normalizes output,
        # so this is assumed vectorized if using our internal workflows
        # through `batching.batch_vectorize`.
        self.vectors = self.get_search_values(db, ["vectors"])[0]


    @timeit()
    def get_ranker(self, db: sqlite3.Connection):
        row = self.get_search_values(db, [
            "bm25_k1",
            "bm25_b",
            "bm25_delta",
            "bm25_corpus_size",
            "bm25_avgdl",
            "bm25_doc_lens",
            "bm25_denom_const",
            "bm25_idf",
            "bm25_doc_ids",
            "bm25_tfs",
            "bm25_indptr",
        ])

        self.ranker = BM25PlusCSR.from_cache(*row)


    @timeit()
    def get_word2vec(self, db: sqlite3.Connection):
        row = self.get_search_values(db, [
            "w2v_index_to_key",
            "w2v_vectors",
            "w2v_syn1",
            "w2v_syn1neg",
        ])

        if row is None or row[0] is None or row[1] is None:
            raise RuntimeError("Word2Vec embeddings were not inited")

        from gensim.models import KeyedVectors

        self.word2vec.wv = KeyedVectors(vector_size=self.word2vec.vector_size)
        self.word2vec.wv.add_vectors(row[0], row[1])

        if row[2] is not None:
            self.word2vec.syn1 = row[2]
        if row[3] is not None:
            self.word2vec.syn1neg = row[3]


    def normalize_pc(self, vector: np.ndarray) -> np.ndarray:
        """Remove the principal component of the dataset to the vector.
        This helps removing stopwords, webpage boilerplates (menu, sidebars),
        formatting language and SEO junk, and makes cosine similarity between
        query and documents more specific.

        Taken from _A simple but tough-to-beat baseline for sentence embeddings_,
        Sanjeev Arora, Yingyu Liang, Tengyu Ma. https://openreview.net/pdf?id=SyK00v5xx

        Arguments:
            vector: can be a single vector (1D) or a document-wise stack of vectors (2D).
            We always consider the embedding vector to be on the last axis, document-wise
            vectors should be vertically stacked.
        Return:
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
        """Filter pages by arbitrary SQL queries"""
        if sql_params is None:
            sql_params = []

        if candidate_indices is not None:
            # Ranking already reduces the search space to a small URL set
            # (`n_results` is 800 from the web app). Applying substring/regex
            # filters to that set avoids scanning every `pages.parsed`/`url`.
            candidate_urls = [self.index[int(i)] for i in candidate_indices]
            matched_indices = []

            # Keep a comfortable margin below SQLite builds that still use the
            # historical 999-bound-parameter limit.
            max_candidate_params = max(1, 900 - len(sql_params))

            for start in range(0, len(candidate_urls), max_candidate_params):
                chunk = candidate_urls[start:start + max_candidate_params]
                placeholders = ", ".join(["(?)" for _ in chunk])
                query = f"""
                    WITH candidate_urls(candidate_url) AS (VALUES {placeholders})
                    SELECT pages.url
                    FROM candidate_urls
                    JOIN pages ON pages.url = candidate_urls.candidate_url
                    {sql_query}
                """
                params = [*chunk, *sql_params]
                cursor = db.execute(query, params)
                matched_indices.extend(self.url_to_index[row[0]] for row in cursor.fetchall())

            return matched_indices

        query = f"""
            SELECT url
            FROM pages
            {sql_query}
            ORDER BY url
        """
        cursor = db.execute(query, sql_params)
        return [self.url_to_index[row[0]] for row in cursor.fetchall()]


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
        
        # Reload all class properties from database
        model.get_word2vec(db)
        model.get_ranker(db)
        model.get_doc_vectors(db)
        model.get_index(db)
        model.build_index_reverse()

        return model


    def tokenize_query(self, query:str, language: str = None, meta_tokens: bool = True, n_grams: bool = True) -> list[str]:
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
        vector = self.word2vec.get_features(tokenized_query, embed="IN")

        # Remove the principal component from the vector
        vector = self.normalize_pc(vector)

        return vector


    @timeit()
    def find_query_pattern(self,
                           indexed_query: np.ndarray[np.int32],
                           documents: list[tuple[int, str, float]],
                           fast=False) -> list[tuple[int, str, float]]:
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
            indexed_query: the search query tokens translated into their integer indices in the Word2Vec vocabulary.
            Use [core.nlp.Word2Vec.tokens_to_indices][] to convert the tokenized query.
            documents: a symbolic list of documents, as a `(index, url, similarity)` tuple.
            fast: if `True`, uses a simplified variant that is 6 times faster and only uses local averages.
            Results from this method are rather inaccurate, for example, for a request like `token_1 token_2`,
            sentences repeating `token_1` twice will score as much as sentences containing the desired sequence
            `token_1 token_2`. If `False`, use the convolutional filter.

        References:
            Text Matching as Image Recognition, Liang Pang, Yanyan Lan, Jiafeng Guo, Jun Xu, Shengxian Wan, and Xueqi Cheng. (2016).
            https://arxiv.org/pdf/1602.06359.pdf

        """
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
    def rank_ai(self, tokens: list[str], fast: bool = False, clip: bool = True) -> np.ndarray:

        if fast:
            # The following seems very close to the next in terms of results.
            # Experimentally, I saw very little difference in rankings, at least not in the first results.
            # The by-the-book is perhaps more immune to keywords stuffing and more sensitive to structure.
            # Differences appear in the tail of the ranking, mostly.
            # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
            return np.nan_to_num(np.dot(self.vectors, self.vectorize_query(tokens)))
        
        else:

            # This is the by-the-book dual embedding space as defined in
            # https://arxiv.org/pdf/1602.01137.pdf
            aggregate = np.zeros(self.vectors.shape[0], dtype=np.float32)
            weights = 0.
            for token in tokens:
                # Compute the cosine similarity of centroids between query and documents,
                # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
                vector = self.word2vec.get_wordvec(token, embed="IN", normalize=True)
                if vector is not None:
                    vector = self.normalize_pc(vector)
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
            db: the SQLite database holding the indexed set of document. This database must absolutely be up-to-date
            with the one used to instanciate this class, regarding row ordering of documents,
            otherwise rowid mismatches are to be expected between fuzzy, AI and regex searches.
            tokens: the tokenized query.
            method (str): `ai`, `fuzzy` or `grep`. `ai` use word embedding and meta-tokens with dual-embedding space, `fuzzy` uses meta-tokens with BM25Okapi stats model, `grep` uses direct string and regex search.
            filter_callback (callable): a function returning a boolean to filter in/out the results of the ranker. Its first argument will be a [core.crawler.web_page][] object from the list [core.nlp.Indexer.index][], the next arguments will be passed through from `**kargs` directly.
            pattern: optional pattern/text search to add on top of AI search
            n_results: number of results to retain
            fine_search: optionally refine the search using a 2D interaction matrix. See [1]
            sql_query: SQL query to narrow-down the search, for example `WHERE field = value`. Supports PCRE regex with `WHERE field REGEXP 'pattern'`.
            sql_params: the SQL parameters such that:
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

        # Note : match needs at least Python 3.10
        match method:
            case search_methods.AI:
                # Aggregate vector embedding method with the ranking from BM25+ to it for each URL.
                # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
                # RRF works very poorly here, so we do "alpha blending"
                aggregates = 0.98 * self.rank_ai(tokens) + 0.02 * self.rank_fuzzy(tokens)
            case search_methods.FUZZY:
                aggregates = self.rank_fuzzy(tokens)
            case _:
                raise ValueError("Unknown ranking method (%s)" % method)

        # Virtual array sorting, that is sort the aggregates relevance coeffs by order of relevance,
        # but only do it on row indices, so we don't actually sort the table iself
        n_results = min(n_results, aggregates.size - 1)
        best_indices = np.argpartition(aggregates, -n_results)[-n_results:]
        best_indices = best_indices[np.argsort(aggregates[best_indices])[::-1]]

        if sql_query != "":
            # Get the intersection of the indices above with the indices from SQL query
            # but keep the ordering from the ranking above
            mask = np.isin(
                best_indices,
                self.filter_contents(db, sql_query, sql_params, candidate_indices=best_indices)
            )
            best_indices = best_indices[mask]

        best_indices = best_indices[-n_results:]
        best_elems = [self.index[i] for i in best_indices]
        best_similarity = aggregates[best_indices]

        ranked = zip(best_indices, best_elems, best_similarity)

        # Now is time for the really heavy stuff: find tokens in sequential order
        if self.collocations and len(tokens) > 2 and fine_search:
            indexed_query = self.word2vec.tokens_to_indices(tokens)
            ranked = self.find_query_pattern(indexed_query, ranked)

        return sorted([(index, page, similarity)
                       for index, page, similarity in ranked], key=lambda x:x[2], reverse=True)


    def get_related(self, tokens: list[str], n: int = 15, k: int = 5) -> list:
        """Get the n closest keywords from the query."""

        vector = self.word2vec.get_features(tokens)

        # wv.similar_by_vector returns a list of (word, distance) tuples
        from_query = [elem for elem in self.word2vec.wv.similar_by_vector(vector, topn=n)]
        from_tokens = [elem for token in tokens for elem in self.word2vec.wv.most_similar(token, topn=k)]

        # sort by relevance
        related = sorted(from_query + from_tokens, key=lambda x:x[1], reverse=True)

        return list(set([elem[0] for elem in related if elem[0] not in tokens]))
