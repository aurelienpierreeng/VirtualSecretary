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


def _batch_normalize_worker(batch) -> list[tuple[int, str, str, str, any]]:
    normalize = TOKENIZER.normalize_text
    out: list[tuple[int, str, str, str, any]] = []

    for i, doc in batch:
        title = clean_whitespaces(sanitize_unicode(doc["title"]))
        content = clean_whitespaces(sanitize_unicode(doc["content"]))
        parsed = normalize(f"{title}\n\n{content}")
        datetime = guess_date(doc["date"])
        out.append((i, title, content, parsed, datetime))

    return out


@timeit()
def batch_parse_web_page(documents: list[web_page], tokenizer: Tokenizer, chunksize: int = 512, cores: int | bool = False):
    """High-performance parallel parsing for [src.core.types.web_page] objects
    
    This function is meant to cleanup text encoding issues and multi-spacings in `web_page` title and content.
    It prepares the `web_page["parsed"]` field from title and content for the next stages of tokenization.
    
    It is needed to call it before [src.core.deduplicator.Deduplicator.dedup()][], so the content duplication
    has a clean parsed version to compare web pages.
    """
    num_cpu = cores or os.cpu_count() or 1

    # Pre-batch work to drastically reduce IPC overhead
    batches = []
    current = []

    for i, doc in enumerate(documents):
        current.append((i, doc))

        if len(current) >= chunksize:
            batches.append(current)
            current = []

    if current:
        batches.append(current)

    ctx = multiprocessing.get_context("fork")

    with futures.ProcessPoolExecutor(
        max_workers=num_cpu,
        mp_context=ctx,
        initializer=_init_batch_normalize_worker,
        initargs=(tokenizer,),
    ) as executor:
        for results in executor.map(_batch_normalize_worker, batches):
            for i, title, content, parsed, datetime in results:
                doc = documents[i]
                doc["title"] = title
                doc["content"] = content
                doc["parsed"] = parsed
                doc["datetime"] = datetime

    return documents


def _init_tokenizer_worker(tokenizer):
    global TOKENIZER
    TOKENIZER = tokenizer


def _batch_tokenize_worker(inputs: tuple[int, str, str | None]) -> tuple[str | None, list[list[str]], int]:
    # Unroll SQL params
    rowid, parsed, lang = inputs
    lang = parse_lang_to_iso639_1(lang)

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
        urls: list of URLs to tokenize. If None, the whole database is processed.
        only_none: stem only the new entries that have not been tokenized already. If `False`,
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
            print(f"Batch {processed_batches} processed")


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
    [core.batching.batch_tokenized][]. The tokenization is destructive and apply stemming,
    stopwords removal, normalization and n-grams if available.

    Arguments:
        urls: list of URLs to tokenize. If None, the whole database is processed.
        only_none: stem only the new entries that have not been stemmed already. If `False`,
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
    vector = WORD2VEC.get_features([word for sentence in tokenized for word in sentence], embed="OUT")

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
