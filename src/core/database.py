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

from .utils import get_models_folder, timeit
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


def create_temp_db(min_free: float = 2.0, filename: str | None = None) -> sqlite3.Connection:
    """Create a temporary SQLite database file (in /dev/shm when available) and
    initialize the `pages` table according to `web_page` annotations.
    
    Arguments:
        min_free: 
            minimum available disk space in GiB required to create the temporary database.
            This is checked at runtime and the function will raise an error if the condition is not met.
        filename: 
            the full path and filename to save the temporary database, if it needs to be reused at some point.    
        
    Returns:
        the sqlite3.Connection opened in bulk mode.

    WARNING:
        the temporary SQLite database doesn't use `web_page` URL as primary key, to allow
        later deduplication.
    """

    # Prefer a per-user location to avoid /tmp diskspace issues on production systems.
    if filename:
        path = Path(filename)
    else:
        base = Path.home().joinpath('.virtualsecretary')
        base.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        fname = f'tmp-{os.getpid()}-{timestamp}.db'
        path = base.joinpath(fname)

    # Ensure the target filesystem has at least a safety margin of free space.
    total, used, free = shutil.disk_usage(path.parent)
    if free < min_free * 1024 * 1024 * 1024 :
        raise RuntimeError(f"Not enough free space in {path} ({free} bytes). Please free space or change your `min_free` setting.")

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
            row = sanitize_web_page(page)
            append(tuple(row[k] for k in keys))

            if len(batch) >= batch_size:
                execute(query, batch)
                batch.clear()

        if batch:
            execute(query, batch)


def db_to_list(db: sqlite3.Connection) -> list[web_page]:
    """Extract all `web_page` rows from the `pages` table in `db` as a list of `web_page`"""   

    fields = web_page.__annotations__.keys()
    
    query = f"""
    SELECT {",".join(fields)}
    FROM pages
    """
 
    return [web_page(**dict(zip(fields, row))) for row in db.execute(query)]


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


def _table_columns(
    conn: sqlite3.Connection,
    table: str,
    schema: str | None = None,
) -> list[str]:
    """Column names for *table*, optionally inside an attached *schema*."""
    pragma = (
        f"PRAGMA {schema}.table_info({table})" if schema
        else f"PRAGMA table_info({table})"
    )
    return [row[1] for row in conn.execute(pragma)]   # row[1] = name column


def _table_pk(conn: sqlite3.Connection, table: str, schema: str | None = None) -> list[str]:
    """Return Primary Key column names in key-sequence order (empty list if none)."""
    pragma = (
        f"PRAGMA {schema}.table_info({table})" if schema
        else f"PRAGMA table_info({table})"
    )
    # PRAGMA row layout: (cid, name, type, notnull, dflt_value, pk)
    # pk > 0  →  column is part of the PK; its value is the 1-based key position.
    pairs = sorted(
        (row[5], row[1])
        for row in conn.execute(pragma)
        if row[5] > 0
    )
    return [name for _, name in pairs]


def _on_conflict_sql(columns: list[str], pk_cols: list[str]) -> str:
    """
    Build the trailing ON CONFLICT … fragment for an upsert.

    Returns an empty string when *pk_cols* is empty (no PK → plain INSERT).
    Returns DO NOTHING when all columns are part of the PK (nothing to update).
    """
    if not pk_cols:
        return ""

    non_pk = [col for col in columns if col not in pk_cols]
    target = "(" + ", ".join(pk_cols) + ")"

    if not non_pk:
        return f"ON CONFLICT{target} DO NOTHING"

    updates = ", ".join(f"{col}=excluded.{col}" for col in non_pk)
    return f"ON CONFLICT{target} DO UPDATE SET {updates}"


def _upsert_fragments(columns: list[str], pk: str = "url") -> tuple[str, str]:
    """Return (quoted_column_list, ON-CONFLICT update clause)."""
    quoted  = ", ".join(columns)
    updates = ", ".join(
        f"{col}=excluded.{col}" for col in columns if col != pk
    )
    return quoted, updates


def _import_via_attach(
    source_path: str,
    dest: sqlite3.Connection,
    where_clause: str,
    params: tuple,
) -> int:
    dest.execute("ATTACH DATABASE ? AS _src", (source_path,))
    cursor = None
    try:
        dest_cols = _table_columns(dest, "pages")
        src_cols  = set(_table_columns(dest, "pages", schema="_src"))
        pk_cols   = _table_pk(dest, "pages")

        select_list = ", ".join(
            col if col in src_cols else f"NULL AS {col}"
            for col in dest_cols
        )
        quoted      = ", ".join(dest_cols)
        on_conflict = _on_conflict_sql(dest_cols, pk_cols)

        cursor = dest.execute(f"""
            INSERT INTO pages ({quoted})
            SELECT {select_list} FROM _src.pages WHERE {where_clause}
            {on_conflict}
        """, params)
        return cursor.rowcount
    finally:
        # cursor.close() finalises the SQLite statement (statement-level resources).
        # dest.commit() ends the implicit transaction that Python opened for the
        # INSERT — that transaction holds a SHARED lock on _src at the *connection*
        # level, which persists after the statement finishes and is the actual
        # reason DETACH raises "database is locked".  Committing releases it.
        # When import_pages owns the connection (both args are paths) the subsequent
        # dest.commit() call in the caller becomes a harmless no-op.
        if cursor is not None:
            cursor.close()
        dest.commit()
        dest.execute("DETACH DATABASE _src")


def _import_via_bridge(
    source: sqlite3.Connection,
    dest: sqlite3.Connection,
    where_clause: str,
    params: tuple,
) -> int:
    dest_cols = _table_columns(dest, "pages")
    src_cols  = set(_table_columns(source, "pages"))
    pk_cols   = _table_pk(dest, "pages")                              # ← dynamic

    select_list = ", ".join(
        col if col in src_cols else f"NULL AS {col}"
        for col in dest_cols
    )

    rows = source.execute(
        f"SELECT {select_list} FROM pages WHERE {where_clause}", params
    ).fetchall()

    if not rows:
        return 0

    quoted       = ", ".join(dest_cols)
    placeholders = ", ".join("?" * len(dest_cols))
    on_conflict  = _on_conflict_sql(dest_cols, pk_cols)               # ← dynamic

    dest.executemany(f"""
        INSERT INTO pages ({quoted})
        VALUES ({placeholders})
        {on_conflict}
    """, rows)

    return len(rows)


@timeit()
def import_pages(
    source_db: str | sqlite3.Connection,
    destination_db: str | sqlite3.Connection,
    where_clause: str = "1=1",
    params: tuple = ()
) -> int:
    """
    Import rows from one SQLite database into another.

    Both *source_db* and *destination_db* may be either a filesystem
    path (str) or an active ``sqlite3.Connection`` handle.  Passing a
    Connection is the only way to target a ``:memory:`` database, since
    those cannot be addressed by path.

    **Connection lifecycle**
        - *Path supplied* – the function opens, commits, and closes the
          connection itself (original behaviour).
        - *Connection supplied* – the caller retains full control; the
          connection is neither committed nor closed here, so the import
          can participate in a larger transaction.

    Rows are copied from ``source.pages`` into ``destination.pages``.
    Existing rows are updated on conflict of the ``url`` primary key.
    Columns present in the destination but absent from the source receive
    NULL.  Both schemas are discovered at runtime, so the function adapts
    automatically if either evolves.

    Args:
        source_db:
            Path to, or an open connection for, the source SQLite database.

        destination_db:
            Path to, or an open connection for, the destination SQLite
            database.

        where_clause:
            SQL WHERE clause applied to ``source.pages``.
            Example: ``"domain = ? AND date >= ?"``

        params:
            Positional parameters bound to *where_clause*.

    Returns:
        Number of affected rows.

    Examples::

        # File → file (unchanged from before)
        import_pages("old.db", "new.db", "domain = ?", ("example.com",))

        # In-memory source → file destination
        import_pages(mem_conn, "new.db")

        # File source → in-memory destination (e.g. for tests)
        import_pages("prod.db", mem_conn, "date >= ?", ("2024-01-01",))

        # Both in-memory
        import_pages(src_conn, dst_conn)
    """
    src_is_conn = isinstance(source_db, sqlite3.Connection)
    dst_is_conn = isinstance(destination_db, sqlite3.Connection)

    dest = (
        destination_db if dst_is_conn
        else sqlite3.connect(get_models_folder(destination_db))
    )

    try:
        if src_is_conn:
            # Live connections cannot be addressed via ATTACH; bridge through Python.
            rowcount = _import_via_bridge(source_db, dest, where_clause, params)
        else:
            # File paths can be ATTACHed for a single-statement INSERT … SELECT.
            rowcount = _import_via_attach(
                get_models_folder(source_db), dest, where_clause, params
            )

        dest.commit()
        compress_db(dest)

    finally:
        if not dst_is_conn:
            dest.close()

    src_label = "<memory>" if src_is_conn else source_db
    dst_label = "<memory>" if dst_is_conn else destination_db
    print(f"Imported {rowcount} rows from {src_label} to {dst_label}")
    return rowcount


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