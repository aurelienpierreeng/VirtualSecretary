"""High-performance, paralellized high-level methods to process large corpora of documents.

Interfaces NLP processing with database entries, for efficient RAM management.

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