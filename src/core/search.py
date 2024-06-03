from scipy.signal import convolve2d
from enum import IntEnum

import sqlite3
import joblib
import pickle
import numpy as np
import regex as re

from rank_bm25 import BM25Plus

from .utils import get_models_folder, timeit
from .nlp import Word2Vec


class search_methods(IntEnum):
    """Search methods available"""
    AI = 1
    FUZZY = 2
    GREP = 3

class Indexer():
    @timeit()
    def __init__(self,
                 db: sqlite3.Connection,
                 name: str,
                 word2vec: Word2Vec,
                 strip_collocations: bool = False):
        """Search engine based on word similarity.

        Arguments:
            training_set (list): list of Data elements. If the list is empty, it will try to find a pre-trained model matching the `path` name.
            path : path to save the trained model for reuse, as a Python joblib.
            name (str): name under which the model will be saved for la ter reuse.
            word2vec (Word2Vec): the instance of word embedding model.
            strip_collocations: remove the matrix of collocations in documents, which is the list of word tokens represented by their index in the
            word2vec dictionnary. It is used for [core.nlp.Indexer.find_query_patterns][], which is optional and significatively slower
            (but not significatively better), so if you don't plan on using it, removing collocations saves some RAM and I/O.

        """
        if word2vec:
            self.word2vec = word2vec
        else:
            raise ValueError("wv needs to be a dictionnary-like map")

        self.index: list | None = None
        """Store the list of reducted web_pages"""

        self.vectors_all: np.ndarray | None = None
        """Store the list of document-wise vector embeddings, where the vector represents
        the (un-normalized) centroid of tokens vectors contained the document.

        Documents are on the first axis.
        """

        self.all_norms: np.ndarray | None = None
        """Store the list of L2 norms for each document vector representation."""

        # TODO
        self.collocations: np.ndarray | None = None # if strip_collocations else [doc[2] for doc in docs]
        """Store the list of document tokens encoded by their index number in the
        Word2Vec vocabulary. Unknown tokens are discarded. This gives a symbolic
        and more compact representation of tokens collocations in documents (32 bits/token).

        Documents are on the first axis.
        """

        #cursor = db.execute("pragma table_info(pages)")
        #print(cursor.fetchall())
        cursor = db.execute("SELECT tokenized FROM pages")

        # Values from https://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf
        self.ranker = BM25Plus([[word for sentence in doc[0] for word in sentence] for doc in cursor.fetchall()], k1=1.7, b=0.3, delta=0.65)
        self.pages = self.ranker.corpus_size
        self.words = len(word2vec.wv)

        self.save(name)

    @timeit()
    def get_contents(self, db: sqlite3.Connection):
        """Lazily load the list of `web_page` from a DB extraction"""

        # TODO: need to check that the current state of the DB is compatible with the DB used to train the ranker
        # aka web pages are at least in the same order, and perhaps have the same content.
        # Problem is how fast can this be made ?

        if self.index is None:
            cursor = db.execute("SELECT url, date, datetime, category, content, vectorized, title, excerpt FROM pages")
            self.index = []
            self.vectors_all = []

            for item in cursor.fetchall():
                self.index.append({"url": item[0], "date": item[1], "datetime": item[2], "category": item[3], "content": item[4], "title": item[6], "excerpt": item[7] })
                self.vectors_all.append(item[5])

            self.vectors_all = np.array(self.vectors_all, dtype=np.float32)
            self.all_norms = np.linalg.norm(self.vectors_all, axis=1)


    def save(self, name: str):
        # Save the model to a reusable object
        joblib.dump(self, get_models_folder(name + ".joblib"), compress='lz4', protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
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
        norm = np.linalg.norm(vector)
        norm = 1.0 if norm == 0.0 else norm

        return vector, norm, tokenized_query


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
        which means it needs to run an a subset of the search index, after previous methods were tried,
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
    def rank_grep(self, query: re.Pattern|str) -> np.ndarray:
        if not (isinstance(query, str) or isinstance(query, re.Pattern)):
            raise ValueError("Wrong query type (%s) for GREP ranking method. Should be string or regular expression pattern" % type(query))

        results = np.array([len(re.findall(query, item["content"], timeout=60))
                            for item in self.index], dtype=np.float64)
        max_rank = np.amax(results)
        if max_rank > 0.: results /= max_rank
        return results


    @timeit()
    def rank_fuzzy(self, tokens: list[str]) -> np.ndarray:
        if not isinstance(tokens, list):
            raise ValueError("Wrong query type (%s) for FUZZY ranking method. Should be a list of strings." % type(tokens))

        return self.ranker.get_scores(tokens)

    @timeit()
    def rank_ai(self, query: tuple, fast: bool = False) -> np.ndarray:
        if not isinstance(query, tuple):
            raise ValueError("Wrong query type (%s) for AI ranking method. Should be a `(vector, norm, tokens)` tuple" % type(query))

        vector = query[0]
        norm = query[1]
        tokens = query[2]

        if fast:
            # The following seems very close to the next in terms of results.
            # Experimentally, I saw very little difference in rankings, at least not in the first results.
            # The by-the-book is perhaps more immune to keywords stuffing and more sensitive to structure.
            # Differences appear in the tail of the ranking, mostly.
            return np.nan_to_num(np.dot(self.vectors_all, vector) / (norm * self.all_norms))
        else:
            # This is the by-the-book dual embedding space as defined in
            # https://arxiv.org/pdf/1602.01137.pdf
            aggregate = np.zeros(self.all_norms.shape, dtype=np.float32)
            n = 0

            for token in tokens:
                # Compute the cosine similarity of centroids between query and documents,
                vector = self.word2vec.get_wordvec(token, embed="IN", normalize=False)
                if vector is not None:
                    norm = np.linalg.norm(vector)
                    aggregate += np.nan_to_num(np.dot(self.vectors_all, vector) / (norm * self.all_norms))
                    n += 1

            return aggregate / n


    @timeit()
    def rank(self, db: sqlite3.Connection, query: str|tuple|re.Pattern, method: search_methods,
             filter_callback: callable = None, pattern: str | re.Pattern = None, n_results: int = 500, fine_search: bool = False,
             **kargs) -> list[tuple[str, float]]:
        """Apply a label on a post based on the trained model.

        Arguments:
            query (str | tuple | re.Pattern): the query to search. `re.Pattern` is available only with the `grep` method.
            method (str): `ai`, `fuzzy` or `grep`. `ai` use word embedding and meta-tokens with dual-embedding space, `fuzzy` uses meta-tokens with BM25Okapi stats model, `grep` uses direct string and regex search.
            filter_callback (callable): a function returning a boolean to filter in/out the results of the ranker. Its first argument will be a [core.crawler.web_page][] object from the list [core.nlp.Indexer.index][], the next arguments will be passed through from `**kargs` directly.
            pattern: optional pattern/text search to add on top of AI search
            n_results: number of results to retain
            fine_search: optionally refine the search using a 2D interaction matrix. See [1]
            **kargs: arguments passed as-is to the `filter_callback`

        Returns:
            list: the list of best-matching results as (url, similarity) tuples.

        [1]: https://eng.aurelienpierre.com/2024/03/designing-an-ai-search-engine-from-scratch-in-the-2020s/#accounting-for-words-patterns
        """

        self.get_contents(db)

        # Note : match needs at least Python 3.10
        match method:
            case search_methods.AI:
                # Aggregate vector embedding method with the ranking from BM25+ to it for each URL.
                # Coeffs adapted from https://arxiv.org/pdf/1602.01137.pdf
                # This can yield negative results that are still "valid". Offset similarity score by one.
                aggregates = 1. + 0.97 * self.rank_ai(query) + 0.03 * self.rank_fuzzy(query[2])
            case search_methods.FUZZY:
                aggregates = self.rank_fuzzy(query)
            case search_methods.GREP:
                aggregates = self.rank_grep(query)
            case _:
                raise ValueError("Unknown ranking method (%s)" % method)

        ## Filters : there is a catch
        ## We set the similarity coeff to 0 for documents that DON'T match the filter criteria.
        ## We don't actually remove those elements.
        ## Given that we take the 500 best docs below, it is equivalent to removing them if enough close results pop up.
        ## If we don't have enough close results, it's just a desperate attempt to return something, bypassing filters.

        # Filter out documents based on URL filter if provided
        if filter_callback:
            # Create a boolean vector
            urls_match = np.array([filter_callback(page, **kargs) for page in self.index])
            aggregates *= urls_match

        # Filter out documents content NOT matching the pattern
        if pattern:
            content_match = np.array([len(re.findall(pattern, item["content"], timeout=5)) > 0 for item in self.index])
            aggregates *= content_match

        # Sort out whatever remains by similarity coeff and keep only the n first elements
        best_indices = np.argpartition(aggregates, -n_results)[-n_results:]
        best_elems = [self.index[i] for i in best_indices]
        best_similarity = aggregates[best_indices]

        if method == search_methods.AI:
            # Offset back the similarity coeff
            best_similarity -= 1.

        ranked = zip(best_indices, best_elems, best_similarity)

        # Now is time for the really heavy stuff: find tokens in sequential order
        if self.collocations and isinstance(query, tuple) and isinstance(query[2], list) and len(query[2]) > 2 and fine_search:
            indexed_query = self.word2vec.tokens_to_indices(query[2])
            ranked = self.find_query_pattern(indexed_query, ranked)

        return sorted([(index, page, similarity) for index, page, similarity in ranked if similarity > 0.], key=lambda x:x[2], reverse=True)


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
