"""High-performance, paralellized high-level methods to process large corpora of documents.

Mostly applies NLP processing on database entries, for efficient RAM management.

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