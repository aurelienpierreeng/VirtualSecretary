"""High-performance, paralellized high-level methods to process large corpora of documents.

Interfaces NLP processing with database entries, for efficient RAM management.

Database structure is hard-coded and expects conformation to data structures defined in [core.database][] and [core.types][]

© 2026 - Aurélien Pierre
"""

from .patterns import *
from .utils import get_models_folder, typography_undo, clean_whitespaces, timeit, guess_date, sanitize_unicode
from .language import *
from .crawler import web_page
from .nlp import *
from .database import *
from .deduplicator import *

from concurrent import futures
import unicodedata as ud
import multiprocessing
import sqlite3
import os
from datetime import datetime
import hashlib


TOKENIZER: nlp.Tokenizer | None = None
WORD2VEC: nlp.Word2Vec | None = None


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


def _batch_normalize_process_worker(indices: list[int]) -> list[tuple[int, str, str, str, None | datetime, None | str, str, int]]:
    """Worker that reads documents from the shared sqlite DB by index, normalizes and returns results."""
    global SHARED_DB, TOKENIZER
    cur = SHARED_DB.cursor()
    out: list[tuple[int, str, str, str, 'None | datetime', None | str, str, int]] = []

    normalize = TOKENIZER.normalize_text

    for i in indices:
        cur.execute(f'SELECT title, content, date, lang FROM pages WHERE rowid=?', (i,))
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

        content_hash = hashlib.sha1(parsed.encode("utf-8")).hexdigest()
        
        out.append((i, title, content, parsed, datetime, lang, content_hash, length))

    return out


@timeit()
def batch_parse_web_page(documents: sqlite3.Connection, tokenizer: Tokenizer, chunksize: int = 512, cores: int | None = None):
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
    """
    # Determine number of worker threads/processes
    if cores is None or cores is True:
        num_workers = os.cpu_count() or 1
    else:
        num_workers = int(cores)

    cursor = documents.cursor()

    # collect rowids in chunks to avoid large memory usage
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
            for rowid, title, content, parsed, datetime, lang, content_hash, length in results:
                pending_updates.append((title, content, parsed, datetime, lang, content_hash, length, rowid))

                if len(pending_updates) >= 2048:
                    cursor.executemany('UPDATE pages SET title=?, content=?, parsed=?, datetime=?, lang=?, content_hash=?, length=? WHERE rowid=?', pending_updates)
                    documents.commit()
                    pending_updates.clear()

        if pending_updates:
            cursor.executemany('UPDATE pages SET title=?, content=?, parsed=?, datetime=?, lang=?, content_hash=?, length=? WHERE rowid=?', pending_updates)
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
        cursor = db.execute(
            f"""
            SELECT rowid, parsed, lang
            FROM pages
            WHERE url IN ({ ",".join(["?" for _ in urls]) })
            """,
            urls,
        )
    elif only_none:
        cursor = db.execute('SELECT rowid, parsed, lang FROM pages WHERE tokenized is NULL')
    else:
        cursor = db.execute('SELECT rowid, parsed, lang FROM pages')

    processed_batches = 0
    row_count = cursor.rowcount
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


@timeit()
def batch_stem(db: sqlite3.Connection, 
               tokenizer: Tokenizer, 
               chunksize: int = 512, 
               urls: list[str] | None = None,
               only_none: bool = True):
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
        cursor = db.execute(
            f"""
            SELECT rowid, tokenized, lang
            FROM pages
            WHERE url IN ({ ",".join(["?" for _ in urls]) })
            """,
            urls,
        )
    elif only_none:
        cursor = db.execute('SELECT rowid, tokenized, lang FROM pages WHERE stemmed IS NULL')
    else:
        cursor = db.execute('SELECT rowid, tokenized, lang FROM pages')

    processed_batches = 0

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_tokenizer_worker,
        initargs=(tokenizer,),
    ) as executor:       
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            results = executor.map(_batch_stem_worker, batch, chunksize=chunksize)
            db.executemany('UPDATE pages SET lang=?, stemmed=? WHERE rowid=?', results)
            db.commit()

            processed_batches += 1
            print(f"Batch {processed_batches} processed")


def _init_vectorizer_worker(word2vec):
    global WORD2VEC
    WORD2VEC = word2vec


def _batch_vectorize_worker(inputs: tuple[int, list[list[str]]]) -> tuple[np.ndarray[np.float32], int]:
    rowid, tokenized = inputs

    # NOTE: tokens are per-sentence/paragraph, so it's a list of list
    vector = WORD2VEC.get_features([word for sentence in tokenized for word in sentence], embed="OUT", use_sif=True)

    #TODO:
    #indices = word2vec.tokens_to_indices(tokens)

    return vector, rowid # keep in sync with SQL query


@timeit()
def batch_vectorize(db: sqlite3.Connection, word2vec: Word2Vec, chunksize: int = 256):
    """Vectorize a column of the `db` database using the provided `word2vec` model
    using all available cores.

    Works on the `tokenized` column of the database and writes the `vectorized` column.
    Vectors are normalized as per `nlp.Word2Vec.get_features()` output.    
    """

    num_cpu = os.cpu_count() or 1
    cursor = db.execute('SELECT rowid, stemmed FROM pages')
    batch_size = num_cpu * chunksize

    with futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        initializer=_init_vectorizer_worker,
        initargs=(word2vec,),
    ) as executor:  
        while True:
            batch = cursor.fetchmany(batch_size)
            if not batch:
                break

            results = executor.map(_batch_vectorize_worker, batch, chunksize=chunksize)
            db.executemany('UPDATE pages SET vectorized=? WHERE rowid=?', results)
            db.commit()
