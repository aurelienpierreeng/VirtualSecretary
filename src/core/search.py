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
            training_set (list): list of Data elements. If the list is empty, it will try to find a pre-trained model matching the `path` name.
            path : path to save the trained model for reuse, as a Python joblib.
            name (str): name under which the model will be saved for la ter reuse.
            word2vec (Word2Vec): the instance of word embedding model.
            strip_collocations: remove the matrix of collocations in documents, which is the list of word tokens represented by their index in the
            word2vec dictionnary. It is used for [core.nlp.Indexer.find_query_patterns][], which is optional and significatively slower
            (but not significatively better), so if you don't plan on using it, removing collocations saves some RAM and I/O.
            principal_components (int): number of principal components to compute and remove from the index dataset. 
            This helps to make queries more selective and specific in the presence of boilerplate text and formatting language in the sampling.

        """

        if word2vec:
            self.word2vec = word2vec
        else:
            raise ValueError("wv needs to be a dictionnary-like map")

        self.index: list | None = None
        """Store the list of reducted web_pages"""

        self.vectors_all: np.ndarray | None = None
        """Store the list of document-wise vector embeddings, where the vector represents
        the normalized centroid of tokens vectors contained the document.
        Documents are on the first axis.
        """

        # TODO
        self.collocations: np.ndarray | None = None # if strip_collocations else [doc[2] for doc in docs]
        """Store the list of document tokens encoded by their index number in the
        Word2Vec vocabulary. Unknown tokens are discarded. This gives a symbolic
        and more compact representation of tokens collocations in documents (32 bits/token).

        Documents are on the first axis.
        """

        cursor = db.execute("SELECT tokenized FROM pages")
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

        self.ranker = BM25PlusCSR(corpus_token_indices, self.word2vec, k1=1.8, b=0.4, delta=0.8)
        # Use our own implementation of BM25+, which gives similar ranking albeit with different coeffs
        # but runs 7 times faster. Otherwise:
        # self.ranker = BM25Plus(corpus_token_indices, k1=1.7, b=0.3, delta=0.65)
        # BM25+ values from https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
                               
        # Get stats
        self.words = len(word2vec.wv)
        self.pages = self.ranker.corpus_size

        self.index = []
        vectors_all = []

        cursor = db.execute("SELECT rowid, url, datetime, category, title, excerpt, vectorized FROM pages")
        for item in cursor.fetchall():
            self.index.append({ "index": item[0], "url": item[1], "datetime": item[2], "category": item[3], "title": item[4], "excerpt": item[5] })
            vectors_all.append(item[6])

        # Note: `nlp.Word2Vec.get_features`` already normalizes output,
        # so this is assumed vectorized if using our internal workflows
        # through `nlp.batch_vectorize`.
        self.vectors_all = np.array(vectors_all, dtype=np.float32)

        # Compute the principal axis direction and normalize document vectors with it
        pca = PCA(n_components=principal_components)
        pca.fit(self.vectors_all)
        self.pc = pca.components_
        """Principal component of the dataset vectors (normalized)"""

        # Remove the principal component on the document vector stack
        self.vectors_all = self.normalize_pc(self.vectors_all)

        self.sql = None
        """Cache the previous SQL filtering conditions"""

        self.save(name)


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
    def get_contents(self, db: sqlite3.Connection, sql: str = "") -> list[int]:
        """Lazily load the list of `web_page` from a DB extraction"""
        cursor = db.execute("SELECT rowid FROM pages " + sql)
        return [item[0] - 1 for item in cursor.fetchall()]


    def save(self, name: str):
        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"), compress='lz4', protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    @timeit()
    def load(cls, name: str):
        """Load an existing trained model by its name from the `../models` folder."""
        try:
            model = joblib.load(get_models_folder(name) + ".joblib")
            if isinstance(model, Indexer):
                return model
        except FileNotFoundError:
            model = joblib.load(get_models_folder(name) + ".joblib.bz2")
            if isinstance(model, Indexer):
                return model
            else:
                raise AttributeError("Model of type %s can't be loaded by %s" % (type(model), str(cls)))


    def tokenize_query(self, query:str, language: str = None, meta_tokens: bool = True) -> list[str]:
        return self.word2vec.tokenizer.tokenize_document(query, language=language, meta_tokens=meta_tokens)


    def vectorize_query(self, tokenized_query: list[str]) -> tuple[np.ndarray, float, list[str]]:
        """Prepare a text search query: cleanup, tokenize and get the centroid vector.

        Returns:
            tuple[vector, norm, tokens]
        """

        if not tokenized_query:
            return np.array([]), 0., []

        # Get the the centroid of the word embedding vector
        vector = self.word2vec.get_features(tokenized_query, embed="IN")

        # Remove the principal component from the vector
        vector = self.normalize_pc(vector)

        return vector, 1.0, tokenized_query


    @timeit()
    def find_query_pattern(self,
                           indexed_query: np.ndarray[np.int32],
                           documents: list[tuple[int, str, float]],
                           fast=False) -> np.ndarray[np.float32]:
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
        if not isinstance(tokens, list):
            raise ValueError("Wrong query type (%s) for FUZZY ranking method. Should be a list of strings." % type(tokens))

        symbolic_tokens = [self.word2vec.wv.key_to_index[word]
                           for word in tokens
                           if word in self.word2vec.wv.key_to_index]

        return self.ranker.get_scores(symbolic_tokens)

    @timeit()
    def rank_ai(self, query: tuple, fast: bool = False) -> np.ndarray:
        if not isinstance(query, tuple):
            raise ValueError("Wrong query type (%s) for AI ranking method. Should be a `(vector, norm, tokens)` tuple" % type(query))

        query_vector = query[0]
        query_tokens = query[2]

        if fast:
            # The following seems very close to the next in terms of results.
            # Experimentally, I saw very little difference in rankings, at least not in the first results.
            # The by-the-book is perhaps more immune to keywords stuffing and more sensitive to structure.
            # Differences appear in the tail of the ranking, mostly.
            # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
            return np.nan_to_num(np.dot(self.vectors_all, query_vector))
        else:
            # This is the by-the-book dual embedding space as defined in
            # https://arxiv.org/pdf/1602.01137.pdf
            aggregate = np.zeros(self.vectors_all.shape[0], dtype=np.float32)
            weights = 0.
            for token in query_tokens:
                # Compute the cosine similarity of centroids between query and documents,
                # Note: self.vector_all and vector are already normalized if using `self.vectorize_query`
                vector = self.word2vec.get_wordvec(token, embed="IN", normalize=True)
                vector = self.normalize_pc(vector)
                if vector is not None:
                    aggregate += np.nan_to_num(np.dot(self.vectors_all, vector))
                    weights += 1.

            return aggregate / weights
        
    def rrf(self, ranks_1: np.ndarray, ranks_2: np.ndarray) -> np.ndarray:
        """Reciprocal Rank Fusion
        
        Aggregate 2 sets of page rankings obtained from different semantic geometries and weighted differently.

        From _Reciprocal rank fusion outperforms condorcet and individual rank learning methods_,
        Gordon V. Cormack, Charles L A Clarke, Stefan Buettcher.
        https://dl.acm.org/doi/10.1145/1571941.1572114
        """
        return 1. / (60. + ranks_1) + 1. / (60. + ranks_2)


    @timeit()
    def rank(self, db: sqlite3.Connection, query: str|tuple|re.Pattern, method: search_methods,
             n_results: int = 500, fine_search: bool = False, sql: str = "", 
             filter_callback: callable = None, callback_data: dict = None) -> list[tuple[str, float]]:
        """Apply a label on a post based on the trained model.

        Arguments:
            query (str | tuple | re.Pattern): the query to search. `re.Pattern` is available only with the `grep` method.
            method (str): `ai`, `fuzzy` or `grep`. `ai` use word embedding and meta-tokens with dual-embedding space, `fuzzy` uses meta-tokens with BM25Okapi stats model, `grep` uses direct string and regex search.
            filter_callback (callable): a function returning a boolean to filter in/out the results of the ranker. Its first argument will be a [core.crawler.web_page][] object from the list [core.nlp.Indexer.index][], the next arguments will be passed through from `**kargs` directly.
            pattern: optional pattern/text search to add on top of AI search
            n_results: number of results to retain
            fine_search: optionally refine the search using a 2D interaction matrix. See [1]
            sql: SQL query to narrow-down the search, for example `WHERE field = value`. Supports PCRE regex with `WHERE field REGEXP 'pattern'`.
            filter_callback: an user-defined callback filtering index items, having the signature
            `filter_callback(page: dict[str], callback_data: dict) -> bool`. It will return `True`
            to include a page or `False` to exclude it. The page is a `dict` of `self.index` objects,
            having the keys:
                - `index`: int, rowid of the `web_page` in the database,
                - `url`: str
                - `title`: str
                - `category`: str
                - `excerpt`: str
                - `datetime`: datetime.datetime
            callback_data: arbitrary user data passed on as the second argument to `filter_callback()` if given

        Note:
            Both SQL search into the database and Python filtering into the index are supported,
            and can be combined. The local index is a partial copy of the database and is already
            a Python object, so it will be faster to filter if you only need to parse the copied data
            to filter in/out.

        Returns:
            list: the list of best-matching results as (url, similarity) tuples.

        [1]: https://eng.aurelienpierre.com/2024/03/designing-an-ai-search-engine-from-scratch-in-the-2020s/#accounting-for-words-patterns
        """

        # Note : match needs at least Python 3.10
        match method:
            case search_methods.AI:
                # Aggregate vector embedding method with the ranking from BM25+ to it for each URL.
                # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
                # This can yield negative results that are still "valid". Offset similarity score by one.
                # RRF works very poorly here, so we do "alpha blending"
                aggregates = 0.98 * self.rank_ai(query) + 0.02 * self.rank_fuzzy(query[2])
            case search_methods.FUZZY:
                aggregates = self.rank_fuzzy(query[2])
            case _:
                raise ValueError("Unknown ranking method (%s)" % method)


        # Virtual array sorting, that is sort the aggregates relevance coeffs by order of relevance,
        # but only do it on row indices, so we don't actually sort the table iself
        n_results = min(n_results, aggregates.size - 1)
        best_indices = np.argpartition(aggregates, -n_results)

        if sql != "":
            # Get the intersection of the indices above with the indices from SQL query
            # but keep the ordering from the ranking above
            mask = np.isin(best_indices, self.get_contents(db, sql))
            best_indices = best_indices[mask]

        if filter_callback:
            # Use Python filtering on self.index items
            best_indices = [index for index in best_indices
                            if filter_callback(self.index[index], callback_data)]

        best_indices = best_indices[-n_results:]
        best_elems = [self.index[i] for i in best_indices]
        best_similarity = aggregates[best_indices]

        ranked = zip(best_indices, best_elems, best_similarity)

        # Now is time for the really heavy stuff: find tokens in sequential order
        if self.collocations and isinstance(query, tuple) and isinstance(query[2], list) and len(query[2]) > 2 and fine_search:
            indexed_query = self.word2vec.tokens_to_indices(query[2])
            ranked = self.find_query_pattern(indexed_query, ranked)

        return sorted([(index, page, similarity)
                       for index, page, similarity in ranked
                       if similarity > 0.], key=lambda x:x[2], reverse=True)


    def get_related(self, post: tuple[np.ndarray, float, list[str]], n: int = 15, k: int = 5) -> list:
        """Get the n closest keywords from the query."""

        if not isinstance(post, tuple):
            raise TypeError("The argument should be either a (vector, norm) tuple or a string")

        vector = post[0]
        tokens = post[2]

        # wv.similar_by_vector returns a list of (word, distance) tuples
        from_query = [elem for elem in self.word2vec.wv.similar_by_vector(vector, topn=n)]
        from_tokens = [elem for token in tokens for elem in self.word2vec.wv.most_similar(token, topn=k)]

        # sort by relevance
        related = sorted(from_query + from_tokens, key=lambda x:x[1], reverse=True)

        return list(set([elem[0] for elem in related if elem[0] not in tokens]))
