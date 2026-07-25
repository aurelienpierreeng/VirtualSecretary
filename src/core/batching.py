"""High-performance, paralellized high-level methods to process large corpora of documents.

Interfaces NLP processing with database entries, for efficient RAM management.

Database structure is hard-coded and expects conformation to data structures defined in [core.database][] and [core.types][]

© 2026 - Aurélien Pierre
"""

from .patterns import *
from .utils import get_models_folder, typography_undo, clean_whitespaces, timeit, guess_date, sanitize_unicode
from .language import *
from .crawler import web_page
from .types import compute_content_hash
from .nlp import *
from .database import *
from .deduplicator import *

from concurrent import futures
from collections import Counter
import unicodedata as ud
import multiprocessing
import sqlite3
import os
from datetime import datetime


TOKENIZER: nlp.Tokenizer | None = None
WORD2VEC: nlp.Word2Vec | None = None
TITLE_WEIGHT: float = 0.5
USE_SIF: bool = True
SIF_SMOOTHING: float = 1e-3
BODY_TOP_K: int = 48


def _guess_dates_batch(batch: list[tuple[int, str]]) -> list[tuple[int, str]]:
    out = []
    for rowid, text_date in batch:
        out.append((guess_date(text_date), rowid))

    return out


@timeit()
def batch_guess_dates(db: sqlite3.Connection, chunksize: int = 2048):
    """
    High-throughput parallel datetime parsing.
    """

    num_cpu = os.cpu_count() or 1

    cursor = db.execute("SELECT rowid, date FROM pages")
    execute = db.executemany

    # Prebatch to reduce IPC overhead
    batches = []

    while True:
        batch = cursor.fetchmany(chunksize)

        if not batch:
            break

        batches.append(batch)

    with db:  # single transaction
        with futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
            # chunksize is 1 because our chunk is already a batch list
            for results in executor.map(_guess_dates_batch, batches, chunksize=1):
                execute("UPDATE pages SET datetime=? WHERE rowid=?", results)


def _init_batch_normalize_worker(tokenizer):
    global TOKENIZER
    TOKENIZER = tokenizer



SHARED_DB: sqlite3.Connection | None = None
SHARED_TABLE_NAME: str | None = None


def _init_batch_normalize_process_worker(tokenizer, db_path: str):
    """Initializer for process pool workers: set tokenizer and open read-only sqlite DB."""
    global TOKENIZER, SHARED_DB
    TOKENIZER = tokenizer
    # Open a separate connection per process
    SHARED_DB = sqlite3.connect(db_path, check_same_thread=False, detect_types=sqlite3.PARSE_DECLTYPES)
    SHARED_DB.row_factory = sqlite3.Row
    SHARED_DB.execute("PRAGMA temp_store = MEMORY")
    # Enable WAL and set a busy timeout so readers wait instead of failing when writes occur
    SHARED_DB.execute("PRAGMA journal_mode = WAL")
    SHARED_DB.execute("PRAGMA busy_timeout = 5000")


def _batch_normalize_process_worker(indices: list[int]) -> list[tuple[int, str, str, str, str, None | datetime, None | str, str, int]]:
    """Worker that reads documents from the shared sqlite DB by index, normalizes and returns results."""
    global SHARED_DB, TOKENIZER
    cur = SHARED_DB.cursor()
    out: list[tuple[int, str, str, str, str, 'None | datetime', None | str, str, int]] = []

    normalize = TOKENIZER.normalize_text

    for i in indices:
        cur.execute(f'SELECT title, content, excerpt, date, lang FROM pages WHERE rowid=?', (i,))
        row = cur.fetchone()
        if row is None:
            continue

        title = clean_whitespaces(sanitize_unicode(row['title']))
        content = clean_whitespaces(sanitize_unicode(row['content']))
        parsed = normalize(f"{title}\n\n{content}")
        length = len(parsed)
        datetime = guess_date(row['date'])
        lang = parse_lang_to_iso639_1(row['lang'])

        if lang is None:   
            lang = detect_language(parsed)

        if row['excerpt'] is None:
            excerpt = content[:800]
        else:
            excerpt = row['excerpt']

        content_hash = compute_content_hash(parsed)

        out.append((i, title, content, excerpt, parsed, datetime, lang, content_hash, length))

    return out


@timeit()
def batch_parse_web_page(documents: sqlite3.Connection, tokenizer: Tokenizer, chunksize: int = 512, cores: int | None = None,
                         only_none: bool = False):
    """High-performance parallel parsing for [core.types.web_page][] objects

    This function is meant to cleanup text encoding issues and multi-spacings in `web_page` title and content.
    It prepares the `web_page["parsed"]` field from title and content for the next stages of tokenization,
    and updates language (using declared ISO code or machine-learned detection).

    It is needed to call it before [core.deduplicator.Deduplicator][], so the content duplication
    has a clean parsed version to compare web pages.

    Arguments:
        documents:
            any database having [core.types.web_page][] rows stored in a `pages` table
            and stored on the filesystem. It cannot be a memory-hosted database: each parallel
            worker will open its own copy by file path.

        tokenizer:
            we only use it for the the [core.nlp.Tokenizer.normalize_text][] method

        chunksize:
            number of SQLite rows to process at once, too many is not helpful since some batches
            may take longer than others, depending on text length.

        cores: CPU cores to use for parallel processing.

        only_none:
            parse only the rows that have not been parsed yet (`parsed IS NULL`). Each worker
            recomputes `parsed`/`content_hash`/`length`/`lang` from the raw `title`/`content`,
            never from the existing `parsed`, so already-parsed rows are byte-for-byte identical
            on re-run and safe to skip. Use this on an incrementally-updated index (freshly-crawled
            pages arrive already parsed via the temporary DB) to avoid re-normalizing the whole
            corpus every day. If `False` (default), the whole database is re-parsed, which is what
            you want when the normalization logic itself changed.
    """
    # Determine number of worker threads/processes
    if cores is None or cores is True:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = int(cores)

    cursor = documents.cursor()

    # collect rowids in chunks to avoid large memory usage
    if only_none:
        rowid_cursor = cursor.execute('SELECT rowid FROM pages WHERE parsed IS NULL ORDER BY rowid')
    else:
        rowid_cursor = cursor.execute('SELECT rowid FROM pages ORDER BY rowid')
    batches = []
    current = []
    for row in rowid_cursor:
        current.append(row[0])
        if len(current) >= chunksize:
            batches.append(list(current))
            current.clear()

    if current:
        batches.append(list(current))

    ctx = multiprocessing.get_context("fork")

    with futures.ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_batch_normalize_process_worker,
        initargs=(tokenizer, database.get_db_filename(documents)),
    ) as executor:
        # Collect updates and commit in batches to minimize write-lock churn
        pending_updates = []
        for results in executor.map(_batch_normalize_process_worker, batches):
            for rowid, title, content, excerpt, parsed, datetime, lang, content_hash, length in results:
                pending_updates.append((title, content, excerpt, parsed, datetime, lang, content_hash, length, rowid))

                if len(pending_updates) >= 2048:
                    cursor.executemany('UPDATE pages SET title=?, content=?, excerpt=?, parsed=?, datetime=?, lang=?, content_hash=?, length=? WHERE rowid=?', pending_updates)
                    documents.commit()
                    pending_updates.clear()

        if pending_updates:
            cursor.executemany('UPDATE pages SET title=?, content=?, excerpt=?, parsed=?, datetime=?, lang=?, content_hash=?, length=? WHERE rowid=?', pending_updates)
            documents.commit()

    return documents


def _init_tokenizer_worker(tokenizer):
    global TOKENIZER
    TOKENIZER = tokenizer


def _batch_tokenize_worker(inputs: tuple[int, str, str | None]) -> tuple[str | None, list[list[str]], int]:
    # Unroll SQL params
    rowid, parsed, lang = inputs
    lang = parse_lang_to_iso639_1(lang)

    if lang is None:
        lang = detect_language(parsed)

    # Tokenize without stemming/lemmatization and keep stopwords
    tokenized = TOKENIZER.tokenize_document_per_sentence(parsed, lang, n_grams=False,
                                                         normalize=False, meta_tokens=True, 
                                                         stem=False, remove_stopwords=False)

    return lang, tokenized, rowid # keep order in sync with updating SQL query



@timeit()
def batch_tokenize(db: sqlite3.Connection, 
                   tokenizer: Tokenizer, 
                   chunksize: int = 512, 
                   urls: list[str] | None = None,
                   only_none: bool = True):
    """Tokenize a list of `web_pages` in a non-destructive way, in parallel, in a RAM-friendly way, directly in database.

    Populate the `tokenized` database column from the `parsed` column. This needs to run after
    [core.batching.batch_parse_web_page][] and prepares n-gram training if any, or stemming.
    
    Note:
        The tokenization is forced non-destructive and doesn't apply stemming,
        stopwords removal, normalization, or n-grams. Original sentences can be reconstructed
        from joining back the list of tokens.

    Arguments:
        urls: 
            list of URLs to tokenize. If None, the whole database is processed.

        only_none: 
            stem only the new entries that have not been tokenized already. If `False`,
            force-update the whole database. It has no effect when `urls` are explicitely specified
    """

    num_cpu = os.cpu_count()
    batch_size = (num_cpu or 1) * chunksize

    if urls is not None:
        where_sql = f"WHERE url IN ({ ','.join(['?' for _ in urls]) })"
        params = urls
    elif only_none:
        where_sql = "WHERE tokenized IS NULL"
        params = []
    else:
        where_sql = ""
        params = []

    # cursor.rowcount is -1 for SELECT, so count explicitly (mirrors the WHERE) to report
    # exactly how many rows this run will (re)tokenize — the observable measure of how
    # incremental the update is.
    row_count = db.execute(f"SELECT COUNT(*) FROM pages {where_sql}", params).fetchone()[0]
    cursor = db.execute(f"SELECT rowid, parsed, lang FROM pages {where_sql}", params)

    processed_batches = 0
    num_batches = int(np.ceil(row_count / batch_size))
    print(f"Batch tokenization: {row_count} to update, {num_batches} batches")

    with futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_tokenizer_worker,
        initargs=(tokenizer,),
    ) as executor:       
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            results = executor.map(_batch_tokenize_worker, batch, chunksize=chunksize)
            db.executemany('UPDATE pages SET lang=?, tokenized=? WHERE rowid=?', results)
            db.commit()

            processed_batches += 1
            print(f"Batch {processed_batches} over {num_batches } processed")


def _batch_stem_worker(inputs: tuple[int, list[list[str]], str | None]) -> tuple[str | None, list[list[str]], int]:
    # Unroll SQL params
    rowid, tokenized, lang = inputs
    lang = parse_lang_to_iso639_1(lang)

    # Finish the filtering of existing tokens
    if TOKENIZER.supports_ngrams:
        stemmed = [TOKENIZER.post_filter_tokens(TOKENIZER.replace_ngrams(sentence), lang, 
                                                normalize=True, meta_tokens=True, 
                                                stem=True, remove_stopwords=True)
                   for sentence in tokenized]
    else:
        stemmed = [TOKENIZER.post_filter_tokens(sentence, lang, 
                                                normalize=True, meta_tokens=True, 
                                                stem=True, remove_stopwords=True)
                   for sentence in tokenized]

    return lang, stemmed, rowid # keep order in sync with updating SQL query


def _batch_stem_pairs_worker(inputs: tuple[int, list[list[str]], str | None]):
    """Like `_batch_stem_worker`, but ALSO returns the (stem, token) occurrence pairs for the row,
    so `batch_stem(build_stem_tokens=True)` can populate the `stem_tokens` reverse-lookup table in
    the SAME pass — no separate re-stemming step. The pairs are built per-token (as
    `core.nlp.StemTokenIndex` does) because the `stemmed` sentences don't preserve the token→stem
    correspondence the reverse lookup needs."""
    rowid, tokenized, lang = inputs
    lang = parse_lang_to_iso639_1(lang)

    if TOKENIZER.supports_ngrams:
        stemmed = [TOKENIZER.post_filter_tokens(TOKENIZER.replace_ngrams(sentence), lang,
                                                normalize=True, meta_tokens=True,
                                                stem=True, remove_stopwords=True)
                   for sentence in tokenized]
    else:
        stemmed = [TOKENIZER.post_filter_tokens(sentence, lang,
                                                normalize=True, meta_tokens=True,
                                                stem=True, remove_stopwords=True)
                   for sentence in tokenized]

    counter = Counter()
    for sentence in tokenized:
        for token in sentence:
            stem = TOKENIZER.normalize_token(token, lang, meta_tokens=True, stem=True,
                                             normalize=True, remove_stopwords=True)
            if stem and token:
                counter[(stem, token)] += 1

    return lang, stemmed, rowid, list(counter.items())


@timeit()
def batch_stem(db: sqlite3.Connection,
               tokenizer: Tokenizer,
               chunksize: int = 512,
               urls: list[str] | None = None,
               only_none: bool = True,
               build_stem_tokens: bool = False):
    """Tokenize and stem a list of `web_pages` in parallel, in a RAM-friendly way, directly in database.

    Populate the `stemmed` database column from the `tokenized` column. This needs to run after
    [core.batching.batch_tokenize][]. The tokenization is destructive and apply stemming,
    stopwords removal, normalization and n-grams if available.

    Arguments:
        urls: 
            list of URLs to tokenize. If None, the whole database is processed.
            
        only_none: 
            stem only the new entries that have not been stemmed already. If `False`,
            force-update the whole database. It has no effect when `urls` are explicitely specified
    """

    num_cpu = os.cpu_count()
    batch_size = (num_cpu or 1) * chunksize

    if urls is not None:
        where_sql = f"WHERE url IN ({ ','.join(['?' for _ in urls]) })"
        params = urls
    elif only_none:
        where_sql = "WHERE stemmed IS NULL"
        params = []
    else:
        where_sql = ""
        params = []

    # cursor.rowcount is -1 for SELECT, so count explicitly to report how many rows this
    # run will (re)stem — the observable measure of how incremental the update is.
    row_count = db.execute(f"SELECT COUNT(*) FROM pages {where_sql}", params).fetchone()[0]
    cursor = db.execute(f"SELECT rowid, tokenized, lang FROM pages {where_sql}", params)

    processed_batches = 0
    num_batches = int(np.ceil(row_count / batch_size))
    print(f"Batch stemming: {row_count} to update, {num_batches} batches"
          + (" (+ stem_tokens)" if build_stem_tokens else ""))

    # Optional side-output: build the `stem_tokens` reverse-lookup table (stem → token → frequency)
    # in the SAME pass, so it never needs a separate re-stemming step. Only meaningful over the full
    # corpus (not a `urls=`/`only_none` subset), so it is rebuilt fresh here.
    if build_stem_tokens:
        db.execute("""CREATE TABLE IF NOT EXISTS stem_tokens (
                          stem TEXT NOT NULL, token TEXT NOT NULL,
                          occurrences INTEGER NOT NULL DEFAULT 0,
                          PRIMARY KEY (stem, token)) WITHOUT ROWID""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_stem_tokens_stem_freq ON stem_tokens(stem, occurrences DESC)")
        db.execute("DELETE FROM stem_tokens")
        db.commit()
    worker = _batch_stem_pairs_worker if build_stem_tokens else _batch_stem_worker

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_tokenizer_worker,
        initargs=(tokenizer,),
    ) as executor:
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            results = list(executor.map(worker, batch, chunksize=chunksize))

            if build_stem_tokens:
                # results are (lang, stemmed, rowid, pairs); split the UPDATE tuple from the pairs.
                db.executemany('UPDATE pages SET lang=?, stemmed=? WHERE rowid=?',
                               ((lang, stemmed, rowid) for lang, stemmed, rowid, _ in results))
                merged = Counter()
                for _, _, _, pairs in results:
                    for key, c in pairs:
                        merged[key] += c
                if merged:
                    db.executemany(
                        "INSERT INTO stem_tokens(stem, token, occurrences) VALUES (?, ?, ?) "
                        "ON CONFLICT(stem, token) DO UPDATE SET occurrences = occurrences + excluded.occurrences",
                        ((stem, token, c) for (stem, token), c in merged.items()))
            else:
                db.executemany('UPDATE pages SET lang=?, stemmed=? WHERE rowid=?', results)
            db.commit()

            processed_batches += 1
            print(f"Batch {processed_batches} over {num_batches} processed")


def _init_vectorizer_worker(word2vec, title_weight: float = 0.5, use_sif: bool = True,
                            sif_smoothing: float = 1e-3, body_top_k: int = 48):
    global WORD2VEC, TITLE_WEIGHT, USE_SIF, SIF_SMOOTHING, BODY_TOP_K
    WORD2VEC = word2vec
    TITLE_WEIGHT = title_weight
    USE_SIF = use_sif
    SIF_SMOOTHING = sif_smoothing
    BODY_TOP_K = body_top_k


def _batch_vectorize_worker(inputs: tuple[int, list[list[str]], str | None, str | None]) -> tuple[np.ndarray[np.float32], int]:
    rowid, stemmed, title, lang = inputs

    # Body centroid: SIF-weighted mean of OUT vectors over the document
    # (title + content, as stored in `stemmed`). NOTE: tokens are
    # per-sentence/paragraph, so `stemmed` is a list of lists.
    #
    # Length-aware pooling (BODY_TOP_K): keep only the most salient tokens so a
    # long page is represented by its topical content, not diluted toward the
    # corpus mean by its long tail of low-salience words. Short pages (fewer
    # than BODY_TOP_K unique tokens) are unaffected.
    body_vec = WORD2VEC.get_features(
        [word for sentence in stemmed for word in sentence],
        embed="OUT", use_sif=USE_SIF, sif_smoothing=SIF_SMOOTHING, top_k=BODY_TOP_K,
    )

    # Title boost: a focused page repeats its subject in the title, but in the
    # body mean that single line is drowned by hundreds of body tokens, so long
    # on-topic documents get a diluted centroid. Blend a separate title centroid
    # back in (re-stemmed the same way as the body) so keyword-in-title
    # documents point toward those keywords.
    vector = body_vec
    if title and TITLE_WEIGHT > 0.:
        title_tokens = WORD2VEC.tokenizer.tokenize_document_flat(
            WORD2VEC.tokenizer.normalize_text(title),
            language=parse_lang_to_iso639_1(lang),
            n_grams=True, normalize=True, meta_tokens=True,
            stem=True, remove_stopwords=True,
        )
        title_vec = WORD2VEC.get_features(title_tokens, embed="OUT", use_sif=USE_SIF, sif_smoothing=SIF_SMOOTHING)

        # Both centroids are unit vectors (or zero when empty); weighted sum then
        # renormalize gives a direction pulled toward the title.
        blended = body_vec + TITLE_WEIGHT * title_vec
        norm = np.linalg.norm(blended)
        if norm > 0.:
            vector = blended / norm

    return vector, rowid # keep in sync with SQL query


@timeit()
def batch_vectorize(db: sqlite3.Connection, word2vec: Word2Vec, chunksize: int = 256, title_weight: float = 0.5,
                    use_sif: bool = True, sif_smoothing: float = 1e-3, body_top_k: int = 48,
                    only_none: bool = True):
    """Vectorize the documents of the `db` database using the provided embedding
    model, using all available cores.

    Reads the `stemmed` and `title` columns and writes the `vectorized` column.
    Each document vector is the SIF-weighted OUT-embedding centroid of the body,
    blended with a separate centroid of the (re-stemmed) title weighted by
    `title_weight`, then L2-normalized. Title-boosting counteracts the centroid
    dilution that buries long, focused pages under their own body text.

    Arguments:
        title_weight:  relative weight of the title centroid in the blend. `0`
                       reproduces the plain body-only centroid.
        use_sif:       SIF-weight terms when building the centroids.
        sif_smoothing: SIF smoothing constant `a` (see [core.nlp.WordEmbedding.SIF][]).
        body_top_k:    length-aware pooling for the body centroid: keep only the
                       `body_top_k` most salient tokens per document so long
                       pages are de-diluted (see [core.nlp.WordEmbedding.get_features][]).
                       `0` disables it (plain full-document centroid). The title
                       centroid is always built from all title tokens.
        only_none:     vectorize only the rows that have not been vectorized yet
                       (`vectorized IS NULL`). On a daily index update this skips
                       every unchanged page. Retraining the embedding model
                       (chantal-02) or the tokenizer (chantal-01) wipes the
                       `vectorized` column, which forces a full re-vectorization
                       on the next run. Set to `False` to force re-vectorizing the
                       whole database in place (e.g. when only vectorization
                       hyper-parameters changed, without a model retrain).
    """

    num_cpu = os.cpu_count() or 1
    if only_none:
        cursor = db.execute('SELECT rowid, stemmed, title, lang FROM pages WHERE vectorized IS NULL')
    else:
        cursor = db.execute('SELECT rowid, stemmed, title, lang FROM pages')
    batch_size = num_cpu * chunksize

    with futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_vectorizer_worker,
        initargs=(word2vec, title_weight, use_sif, sif_smoothing, body_top_k),
    ) as executor:
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            results = executor.map(_batch_vectorize_worker, batch, chunksize=chunksize)
            db.executemany('UPDATE pages SET vectorized=? WHERE rowid=?', results)
            db.commit()
