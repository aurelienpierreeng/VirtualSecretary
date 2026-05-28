"""
Create an SQLite database of `web_pages` to be used by a search engine.

© 2024 - Aurélien Pierre
"""

import sqlite3
import io
import numpy as np
import regex as re

import json
from datetime import datetime
from collections.abc import Iterable
from pathlib import Path
import os
import shutil

from .utils import get_models_folder
from .types import web_page, sanitize_web_page
from .patterns import *

type_map = {
    str: "TEXT",
    int: "INTEGER",
    datetime: "DATETIME",
    np.ndarray: "ARRAY",
    list: "LIST",
}

# Define codecs for numpy arrays with SQLite types
def adapt_array(arr: np.ndarray):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text: str):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

def load_list_pickle(text: str):
    return json.loads(text)

def dump_list_pickle(blob: list[str]):
    return json.dumps(blob)

sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
sqlite3.register_adapter(list, dump_list_pickle)
sqlite3.register_converter("list", load_list_pickle)


def create_db(name: str) -> sqlite3.Connection:
    """Create the `pages` table if needed and add any missing columns.
    This doesn't destroy existing tables, rows or columns, so it's safe
    to run on any database.

    Warning:
        Columns are inferred directly from `web_page.__annotations__`.
        Existing columns are preserved unchanged.
    
    The `url` column is used as the PRIMARY KEY.
    """

    connector = open_db(name, mode="bulk")
    
    cursor = connector.cursor()

    keys = list(web_page.__annotations__.items())

    # Create initial schema
    columns = []

    for key, value in keys:
        sql_type = type_map.get(value)

        if sql_type is None:
            continue

        # url acts as the primary key
        if key == "url":
            columns.append(f"{key} {sql_type} PRIMARY KEY")
        else:
            columns.append(f"{key} {sql_type}")

    cursor.execute(f"CREATE TABLE IF NOT EXISTS pages ({", ".join(columns)})")

    # Fetch existing columns
    existing_columns = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(pages)")
    }

    # Add newly-added fields from web_page annotations
    for key, value in keys:
        if key in existing_columns:
            continue

        sql_type = type_map.get(value)

        if sql_type is None:
            continue

        cursor.execute(f"ALTER TABLE pages ADD COLUMN {key} {sql_type}")

    connector.commit()

    print(cursor.execute("PRAGMA table_info(pages)").fetchall())

    return connector


def cleanup_temp_db():
    base = Path.home().joinpath('.virtualsecretary')
    base.mkdir(parents=True, exist_ok=True)

    # Remove stale temp DB files to free space if any
    for old in base.glob('tmp-*.db*'):
        try:
            old.unlink()
        except Exception:
            pass


def create_temp_db(min_free=2.0) -> sqlite3.Connection:
    """Create a temporary SQLite database file (in /dev/shm when available) and
    initialize the `pages` table according to `web_page` annotations.
    
    Arguments:
        min_free: 
            minimum available disk space in GiB required to create the temporary database.
            This is checked at runtime and the function will raise an error if the condition is not met.


    Returns:
        the sqlite3.Connection opened in bulk mode.

    WARNING:
        the temporary SQLite database doesn't use `web_page` URL as primary key, to allow
        later deduplication.
    """
    # Remove old temp DB if any
    cleanup_temp_db()

    # Prefer a per-user location to avoid /tmp diskspace issues on production systems.
    base = Path.home().joinpath('.virtualsecretary')
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    fname = f'tmp-{os.getpid()}-{timestamp}.db'
    path = base.joinpath(fname)

    # Ensure the target filesystem has at least a safety margin of free space.
    total, used, free = shutil.disk_usage(str(base))
    if free < min_free * 1024 * 1024 * 1024 :
        raise RuntimeError(f"Not enough free space in {base} ({free} bytes). Please free space or change your `min_free` setting.")

    # Create connection with bulk pragmas
    # Enable WAL and tune timeouts to allow many concurrent readers with one writer.
    db = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30)
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute("PRAGMA temp_store = MEMORY")
    db.execute("PRAGMA cache_size = -200000")
    db.execute("PRAGMA mmap_size = 8000000000")
    db.execute("PRAGMA busy_timeout = 30000")
    db.execute("PRAGMA auto_vacuum = INCREMENTAL;")

    cursor = db.cursor()
    keys = list(web_page.__annotations__.items())

    # Create initial schema
    columns = []

    for key, value in keys:
        sql_type = type_map.get(value)

        if sql_type is None:
            continue

        columns.append(f"{key} {sql_type}")

    cursor.execute(f"CREATE TABLE IF NOT EXISTS pages ({', '.join(columns)})")
    db.commit()

    return db


def open_db(name: str, mode: str = "rw") -> sqlite3.Connection:
    """Open an SQLite database with workload-specific optimizations.

    Arguments:
        name: Database identifier/path passed to `get_models_folder()`.
        mode:
            - "rw": Generic read/write mode.
            - "ro": Read-only immutable mode optimized for serving/search workloads.
            - "bulk": Bulk-ingestion mode optimized for large batch writes.

    Returns:
        sqlite3.Connection
    """

    path = Path(get_models_folder(name))

    common_kwargs = {
        "detect_types": sqlite3.PARSE_DECLTYPES,
    }

    if mode == "ro":
        uri = f"file:{path}?mode=ro&immutable=1"

        db = sqlite3.connect(uri, uri=True,
            isolation_level=None,  # autocommit
            **common_kwargs
        )

        db.execute("PRAGMA query_only = ON")
        db.execute("PRAGMA synchronous = OFF")
        db.execute("PRAGMA temp_store = MEMORY")

        # 256 MB page cache per process
        db.execute("PRAGMA cache_size = -262144")

        # 30 GB max mmap window
        db.execute("PRAGMA mmap_size = 30000000000")

    elif mode == "bulk":
        db = sqlite3.connect(path, **common_kwargs)

        db.execute("PRAGMA journal_mode = WAL")
        db.execute("PRAGMA busy_timeout = 5000")
        db.execute("PRAGMA synchronous = NORMAL")
        db.execute("PRAGMA temp_store = MEMORY")

        # ~200 MB page cache
        db.execute("PRAGMA cache_size = -200000")

        # Larger mmap can help indexing workloads too
        db.execute("PRAGMA mmap_size = 8000000000")

    elif mode == "rw":
        db = sqlite3.connect(path, **common_kwargs)

        db.execute("PRAGMA journal_mode = TRUNCATE")
        db.execute("PRAGMA synchronous = NORMAL")
        db.execute("PRAGMA temp_store = MEMORY")

    else:
        raise ValueError(f"Invalid SQLite mode: {mode!r}")
    
    db.execute("PRAGMA auto_vacuum = INCREMENTAL;")
    db.commit()

    # Add regex support to SQLite3
    def regexp(pattern, string):
        return re.search(pattern, string, re.IGNORECASE, concurrent=True) is not None

    db.create_function("regexp", 2, regexp, deterministic=True)

    return db


def get_db_filename(db: sqlite3.Connection) -> str:
    return db.execute("PRAGMA database_list").fetchone()[2]


def close_db(db: sqlite3.Connection):
   db.execute("PRAGMA incremental_vacuum;")
   db.commit()
   db.close()


def compress_db(db: sqlite3.Connection, delete_query: str | None = None, delete_params: tuple | None = None, delete_columns: list[str] | None = None):
    """
    Optionally delete rows, then reclaim SQLite disk space.

    Args:
        db: SQLite connection
        delete_query: full DELETE SQL query
        delete_params: optional SQL parameters
    """        

    if delete_query:
        cursor = db.cursor()
        cursor.execute(f"DELETE from pages WHERE {delete_query}", delete_params or ())
        deleted = cursor.rowcount
        db.commit()
        print(f"Deleted {deleted} rows WHERE {delete_query}")

    if delete_columns:
        # validate columns exist (important for safety)
        cur = db.execute("PRAGMA table_info(pages)")
        valid_columns = {row[1] for row in cur.fetchall()}
        columns = [c for c in delete_columns if c in valid_columns]

        if columns:
            set_clause = ", ".join(f"{col} = NULL" for col in columns)
            db.execute(f"UPDATE pages SET {set_clause}")
            db.commit()
            print(f"Deleted columns {", ".join(columns)}")

    # Memory-friendly vacuum
    db.execute("PRAGMA incremental_vacuum;")
    db.commit()

    # For some reason, the above is not enough to really remove old stuff.
    # Problem is, the following needs twice the size of the DB available on disk.
    db.execute("PRAGMA VACUUM")
    db.commit()


def is_primary_key(db: sqlite3.Connection, table: str, column: str) -> bool:
    """
    Check whether `column` is part of the PRIMARY KEY of `table`.
    """

    cur = db.execute(f"PRAGMA table_info({table})")

    for row in cur.fetchall():
        name = row[1]
        pk = row[5]

        if name == column:
            return pk > 0

    return False


def populate_db(db: sqlite3.Connection, pages: list[web_page], batch_size: int = 4096):
    """Insert or update `web_page` records into the SQLite database.

    Existing rows are matched using the PRIMARY KEY `url`.

    Warning:
        Array-like Python values are converted to `bytearray`
        then to `bytes` in order to be handled as `BLOB`
        by SQLite.
    """

    cursor = db.cursor()
    keys = tuple(web_page.__annotations__.keys())
    insert_columns = ",".join(keys)
    placeholders = ",".join("?" for _ in keys)

    query = f"""
        INSERT INTO pages ({insert_columns})
        VALUES ({placeholders})
    """

    # If URL is the primary key, we update existing URL 
    # to ensure unicity. Else we append everything
    if is_primary_key(db, "pages", "url"):

        update_columns = ",".join(
            f"{k}=excluded.{k}"
            for k in keys
            if k != "url"
        )

        query += f"""
            ON CONFLICT(url) DO UPDATE SET
            {update_columns}
        """

    batch = []
    append = batch.append
    execute = cursor.executemany

    with db:  # single transaction
        for page in pages:
            row = sanitize_web_page(page, to_db=True)
            append(tuple(row[k] for k in keys))

            if len(batch) >= batch_size:
                execute(query, batch)
                batch.clear()

        if batch:
            execute(query, batch)


def migrate_url_to_primary_key(db: sqlite3.Connection):
    """Rebuild the `pages` table using `url` as PRIMARY KEY
    for older databases that didn't use a primary key.
    """

    cursor = db.cursor()

    # Check current schema
    table_info = cursor.execute("PRAGMA table_info(pages)").fetchall()

    # Abort if url is already primary key
    for column in table_info:
        # column format:
        # (cid, name, type, notnull, dflt_value, pk)
        if column[1] == "url" and column[5] == 1:
            print("url is already PRIMARY KEY")
            return

    # Get current column definitions
    columns = []
    column_names = []

    for _, name, col_type, *_ in table_info:
        column_names.append(name)

        if name == "url":
            columns.append(f"{name} {col_type} PRIMARY KEY")
        else:
            columns.append(f"{name} {col_type}")

    columns_sql = ", ".join(columns)
    names_sql = ", ".join(column_names)

    cursor.execute("BEGIN TRANSACTION")

    try:
        # Create replacement table
        cursor.execute(f"CREATE TABLE pages_new ({columns_sql})")

        # Copy rows
        cursor.execute(f"""
            INSERT OR REPLACE INTO pages_new ({names_sql})
            SELECT {names_sql}
            FROM pages
        """)

        # Remove old table
        cursor.execute("DROP TABLE pages")

        # Rename replacement
        cursor.execute("""
            ALTER TABLE pages_new
            RENAME TO pages
        """)

        db.commit()

        print("Migration completed successfully.")

    except Exception:
        db.rollback()
        raise


def merge_databases(old_db: sqlite3.Connection, new_db: sqlite3.Connection):
    """Merge two `pages` databases.

    Rows from `old_db` are inserted into `new_db`
    only if their URL does not already exist.

    Existing rows in `new_db` are preserved unchanged.

    Only columns existing in BOTH databases are copied.
    """

    old_cursor = old_db.cursor()
    new_cursor = new_db.cursor()

    old_path = old_cursor.execute("PRAGMA database_list").fetchone()[2]
    new_cursor.execute("ATTACH DATABASE ? AS old_db", (old_path,))

    try:
        old_columns = {row[1] for row in new_cursor.execute("PRAGMA old_db.table_info(pages)")}

        new_columns = {row[1] for row in new_cursor.execute("PRAGMA table_info(pages)")}

        # Keep only shared columns
        shared_columns = sorted(old_columns & new_columns)

        if "url" not in shared_columns:
            raise RuntimeError(
                "Both databases must contain a `url` column."
            )

        columns_sql = ", ".join(shared_columns)

        query = f"""
            INSERT OR IGNORE INTO pages ({columns_sql})
            SELECT {columns_sql}
            FROM old_db.pages
        """

        new_cursor.execute("BEGIN")
        new_cursor.execute(query)
        inserted = new_cursor.rowcount
        new_db.commit()

        print(f"Merged {inserted} new rows.")

    except Exception:
        # critical: reset transaction state
        new_db.rollback()
        raise

    finally:
        try:
            new_cursor.execute("DETACH DATABASE old_db")
        except sqlite3.OperationalError:
            # safe ignore: detach can fail after rollback/state error
            pass


def update_pages_from_database( target_db: sqlite3.Connection, source_db: sqlite3.Connection) -> list[str]:
    """
    Update rows in `target_db.pages` from `source_db.pages`
    using `url` as PRIMARY KEY.

    Only shared columns are updated.

    Returns
        missing_urls: URLs present in target_db but absent from source_db.
    """

    target_cursor = target_db.cursor()
    source_cursor = source_db.cursor()

    source_path = source_cursor.execute("PRAGMA database_list").fetchone()[2]

    target_cursor.execute("ATTACH DATABASE ? AS source_db", (source_path,))

    try:
        # Shared columns
        source_columns = { row[1] for row in target_cursor.execute("PRAGMA source_db.table_info(pages)") }
        target_columns = { row[1] for row in target_cursor.execute("PRAGMA table_info(pages)")}
        shared_columns = sorted((source_columns & target_columns) - {"url"})

        if not shared_columns:
            raise RuntimeError("No shared columns to update.")

        # Build SET clause
        set_clause = ", ".join(
            f"{col} = ("
            f"SELECT s.{col} "
            f"FROM source_db.pages s "
            f"WHERE s.url = pages.url"
            f")"
            for col in shared_columns
        )

        target_cursor.execute("BEGIN")

        # Update only rows existing in source
        query = f"""
            UPDATE pages
            SET {set_clause}
            WHERE EXISTS (
                SELECT 1
                FROM source_db.pages s
                WHERE s.url = pages.url
            )
        """

        target_cursor.execute(query)

        updated = target_cursor.rowcount

        # Get missing URLs
        missing_urls = [
            row[0]
            for row in target_cursor.execute("""
                SELECT url
                FROM pages
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM source_db.pages s
                    WHERE s.url = pages.url
                )
            """)
        ]

        target_db.commit()

        print(f"Updated {updated} rows.")
        print(f"{len(missing_urls)} URLs not found in source DB.")

        return missing_urls

    except Exception:
        target_db.rollback()
        raise

    finally:
        try:
            target_cursor.execute(
                "DETACH DATABASE source_db"
            )
        except sqlite3.OperationalError:
            pass


def import_pages(source_db: str, destination_db: str, where_clause: str = "1=1", params: tuple = ()) -> int:
    """
    Import rows from one SQLite database into another.

    Rows are copied from `source.pages` into `destination.pages`.
    Existing rows are updated on conflict of the `url` primary key.

    All columns are automatically discovered from the destination
    `pages` schema, so the function adapts automatically if the
    schema evolves.

    Args:
        source_db:
            Path to the source SQLite database.

        destination_db:
            Path to the destination SQLite database.

        where_clause:
            SQL WHERE clause applied to `source.pages`.

            Example:
                "domain = ? AND date >= ?"

        params:
            Optional SQL parameters used by the WHERE clause.

    Returns:
        Number of affected rows.

    Example:
        import_pages(
            source_db="old.db",
            destination_db="new.db",
            where_clause="domain = ?",
            params=("example.com",)
        )
    """

    with sqlite3.connect(get_models_folder(destination_db)) as db:
        db.row_factory = sqlite3.Row

        # Attach source DB under alias "source"
        db.execute("ATTACH DATABASE ? AS source", (get_models_folder(source_db),))

        # Discover destination schema
        columns = [
            row["name"]
            for row in db.execute("PRAGMA table_info(pages)")
        ]

        # Discover source schema (attached as `source`). If the source
        # database is missing some columns present in the destination,
        # select NULL for those columns to keep the INSERT SELECT robust.
        source_columns = {
            row["name"]
            for row in db.execute("PRAGMA source.table_info(pages)")
        }

        select_list = ", ".join(
            (col if col in source_columns else f"NULL AS {col}")
            for col in columns
        )

        quoted_columns = ", ".join(columns)

        # Build ON CONFLICT update clause
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column != "url"
        )

        sql = f"""
            INSERT INTO pages ({quoted_columns})
            SELECT {select_list}
            FROM source.pages
            WHERE {where_clause}
            ON CONFLICT(url) DO UPDATE SET
                {updates}
        """

        cursor = db.execute(sql, params)

        db.commit()

        db.execute("DETACH DATABASE source")
        print(f"Imported {cursor.rowcount} rows from {source_db} to {destination_db}")
        return cursor.rowcount


class SQLitePageCorpus:
    """
    Lazily stream rows from an SQLite request, avoiding full copy.

    Example:
        ```python
            corpus = SQLitePageCorpus(
                db,
                \"""
                SELECT tokenized
                FROM pages
                WHERE lang IN ('fr', 'en')
                \""",
                max_depth=0
            )
        ```
        - `max_depth=0` will not flatten the content, so it will return
          the original `list[list[str]]` (list of sentences, aka list of list of words),
        - `max_depth=1` flattens documents, to it will return
          `list[str]` (list of words)
    """

    def __init__(self, db, query, params=(), atomic_types=(str, bytes), max_depth=None, yield_rows=False):
        self.db = db
        self.query = query
        self.params = params
        self.atomic_types = atomic_types
        self.max_depth = max_depth
        self.yield_rows = yield_rows

        self._length = None


    def __iter__(self):
        """Iterate over the SQLite query rows with no full copy"""
        cursor = self.db.execute(self.query, self.params)

        for row in cursor:
            if not row:
                continue

            if self.yield_rows:
                yield row
                continue

            for value in row:
                yield from self._flatten(value)


    def __len__(self):
        if self._length is not None:
            return self._length

        count = 0

        cursor = self.db.execute(self.query, self.params)

        for row in cursor:
            if not row:
                continue

            for value in row:
                count += sum(1 for _ in self._flatten(value))

        self._length = count

        return count
        

    def _flatten(self, obj, depth=0):
        """Recursively flatten nested iterables up to `depth` recursions."""

        if obj is None:
            return

        if isinstance(obj, self.atomic_types):
            yield obj
            return

        if (isinstance(obj, Iterable) and (self.max_depth is None or depth < self.max_depth)):
            for item in obj:
                yield from self._flatten(item, depth + 1)
            return

        yield obj


def normalize_wayback_urls(db):
    cur = db.cursor()

    cur.execute("""
        SELECT url, title, datetime
        FROM pages
        WHERE url LIKE '%web.archive.org/%'
    """)

    wayback_rows = cur.fetchall()

    for old_url, title, dt in wayback_rows:
        original_url = wayback_extract_url(old_url)
        if not original_url:
            continue

        # --- derive domain from canonical URL ---
        address = split_url(original_url)
        if not address:
            continue

        protocol, domain, page, params, anchor = address

        # --- check if canonical URL already exists ---
        cur.execute(
            "SELECT 1 FROM pages WHERE url = ?",
            (original_url,)
        )
        exists = cur.fetchone()

        if exists:
            # conflict: keep canonical, delete archive
            cur.execute(
                "DELETE FROM pages WHERE url = ?",
                (old_url,)
            )
            continue

        # --- otherwise replace archive row with canonical ---
        cur.execute("""
            UPDATE pages
            SET url = ?, domain = ?, title = ?
            WHERE url = ?
        """, (original_url, domain, title, old_url))

    db.commit()