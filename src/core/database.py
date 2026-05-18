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

from .utils import get_models_folder
from .types import web_page, sanitize_web_page

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

    type_map = {
        str: "TEXT",
        int: "INTEGER",
        datetime: "DATETIME",
        np.ndarray: "ARRAY",
        list: "LIST",
    }

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

        db.execute("PRAGMA journal_mode = TRUNCATE")
        db.execute("PRAGMA synchronous = NORMAL")
        db.execute("PRAGMA temp_store = MEMORY")

        # ~200 MB page cache
        db.execute("PRAGMA cache_size = -200000")

        # Larger mmap can help indexing workloads too
        db.execute("PRAGMA mmap_size = 8000000000")

        # db.execute("VACUUM")

    elif mode == "rw":
        db = sqlite3.connect(path, **common_kwargs)

        db.execute("PRAGMA journal_mode = TRUNCATE")
        db.execute("PRAGMA synchronous = NORMAL")
        db.execute("PRAGMA temp_store = MEMORY")

    else:
        raise ValueError(f"Invalid SQLite mode: {mode!r}")

    # Add regex support to SQLite3
    def regexp(pattern, string):
        return re.search(pattern, string, re.IGNORECASE, concurrent=True) is not None

    db.create_function("regexp", 2, regexp, deterministic=True)

    return db


def close_db(db: sqlite3.Connection):
   db.execute("ANALYZE")
   db.close()


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

    update_columns = ",".join(f"{k}=excluded.{k}" for k in keys if k != "url")

    query = f"""
        INSERT INTO pages ({insert_columns})
        VALUES ({placeholders})
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


def add_content_hash_column(db: sqlite3.Connection):
    """
    Add and populate/update `content_hash` column from the `parsed` column,
    for content integrity and deduplication purposes.
    """

    import hashlib

    cursor = db.cursor()

    # Add column if missing
    columns = { row[1] for row in cursor.execute("PRAGMA table_info(pages)") }

    if "content_hash" not in columns:
        cursor.execute("""
            ALTER TABLE pages
            ADD COLUMN content_hash TEXT
        """)

    # Stream rows to avoid RAM explosion
    rows = cursor.execute("""
        SELECT rowid, parsed
        FROM pages
    """)

    updates = []

    for rowid, content in rows:
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()
        updates.append((digest, rowid))

        # batch updates
        if len(updates) >= 1024:
            cursor.executemany("""
                UPDATE pages
                SET content_hash = ?
                WHERE rowid = ?
            """, updates)

            db.commit()
            updates.clear()

    if updates:
        cursor.executemany("""
            UPDATE pages
            SET content_hash = ?
            WHERE rowid = ?
        """, updates)

    db.commit()

    # THIS index is cheap and scalable
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_pages_content_hash
        ON pages(content_hash)
    """)

    db.commit()


def deduplicate_pages(db: sqlite3.Connection):
    """
    Safely deduplicate identical contents in the database.

    Keeps:
        1. shortest URL
        2. newest datetime
        3. lowest rowid as deterministic final tie-breaker
    """
    add_content_hash_column(db)

    cursor = db.cursor()
    
    before = cursor.execute("""
        SELECT COUNT(*)
        FROM pages
    """).fetchone()[0]

    cursor.execute("BEGIN")

    try:
        # Important: only rows with non-null hashes
        cursor.execute("""
            CREATE TEMP TABLE keep_rowids AS
            SELECT MIN(rowid) AS rowid
            FROM pages
            WHERE content_hash IS NOT NULL
            GROUP BY content_hash
        """)

        # Delete only rows NOT selected
        # AND only among rows sharing same actual content
        cursor.execute("""
            DELETE FROM pages
            WHERE rowid NOT IN (
                SELECT rowid
                FROM keep_rowids
            )
            AND content_hash IS NOT NULL
        """)

        after = cursor.execute("""
            SELECT COUNT(*)
            FROM pages
        """).fetchone()[0]

        removed = before - after

        cursor.execute("""
            DROP TABLE keep_rowids
        """)

        db.commit()

        print(f"Removed {removed} duplicate rows.")

    except Exception:
        db.rollback()
        raise


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

        quoted_columns = ", ".join(columns)

        # Build ON CONFLICT update clause
        updates = ", ".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column != "url"
        )

        sql = f"""
            INSERT INTO pages ({quoted_columns})
            SELECT {quoted_columns}
            FROM source.pages
            WHERE {where_clause}
            ON CONFLICT(url) DO UPDATE SET
                {updates}
        """

        cursor = db.execute(sql, params)

        db.commit()

        db.execute("DETACH DATABASE source")
        print(f"{cursor.rowcount} rows imported from {source_db} to {destination_db}")
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