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

from .utils import get_models_folder, ensure_decompressed, timeit
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


def create_db(name: str, url_primary_key: bool = True) -> sqlite3.Connection:
    """Create the `pages` table if needed and add any missing columns.
    This doesn't destroy existing tables, rows or columns, so it's safe
    to run on any database.

    Warning:
        Columns are inferred directly from `web_page.__annotations__`.
        Existing columns are preserved unchanged.

    Arguments:
        url_primary_key:
            ``True`` (default) — ``url`` is the PRIMARY KEY, i.e. a UNIQUE index. This is
            required by the ``ON CONFLICT(url) DO UPDATE`` upsert used by
            [core.database.import_pages][], and it silently collapses same-URL rows to one.
            ``False`` — ``url`` is a plain (NON-unique) indexed column. Use this for a
            **canonical dataset** the crawler writes into directly, where the same URL may
            legitimately appear several times (a page mined via special HTML tags, as an
            external whole-body capture, and under several parameter URLs) and duplication is
            resolved by content-hash deduplication rather than enforced by the schema.
            Lookups (``WHERE url = ?`` / ``url IN (…)``) stay O(log N) via the plain index;
            only the uniqueness constraint and upsert capability are dropped.

    Note:
        This only affects a **freshly created** ``pages`` table. On an existing database the
        table (and its key) are left as-is — switching an already-populated DB between the two
        modes requires an explicit table rebuild.
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

        # url is the PRIMARY KEY only when uniqueness/upsert is wanted; otherwise it is a
        # plain column indexed below, so the same URL may appear multiple times.
        if key == "url" and url_primary_key:
            columns.append(f"{key} {sql_type} PRIMARY KEY")
        else:
            columns.append(f"{key} {sql_type}")

    cursor.execute(f"CREATE TABLE IF NOT EXISTS pages ({", ".join(columns)})")

    # A non-unique index on url keeps URL lookups fast when url is not the PRIMARY KEY.
    if not url_primary_key:
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url)")

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


def ensure_web_page_columns(db: sqlite3.Connection) -> list[str]:
    """Add any `web_page` columns missing from an existing `pages` table, on a LIVE connection.

    Fixes schema drift: a source tarball written before a column was introduced (e.g. `dataset`,
    added mid-2026) yields a `pages` table lacking it, so a later `populate_db` INSERT of the full
    `web_page` tuple fails with "table pages has no column named …". Seeding a working DB from such
    a tarball via `Connection.backup()` copies the OLD schema, so callers must run this afterwards.
    Idempotent; returns the list of columns added. Mirrors create_db's add-missing-columns pass but
    targets an open connection instead of a named DB.
    """
    existing = {row[1] for row in db.execute("PRAGMA table_info(pages)")}
    added: list[str] = []
    for key, value in web_page.__annotations__.items():
        if key in existing:
            continue
        sql_type = type_map.get(value)
        if sql_type is None:
            continue
        db.execute(f"ALTER TABLE pages ADD COLUMN {key} {sql_type}")
        added.append(key)
    if added:
        db.commit()
    return added


# ─────────────────────────────────────────────────────────────────────────────────────
# Dataset provenance (multi-source membership)
#
# A page can legitimately originate from several sources (e.g. the same content crawled for
# `ansel` and `ansel-old`, or an archived page recovered from a previous DB). To keep that
# provenance through content deduplication — which collapses a content_hash group to one row —
# the `dataset` column stores a SET of source names encoded as a delimited string
# ",a,b,c," (leading/trailing commas as sentinels), so membership is a simple LIKE '%,name,%'.
# ─────────────────────────────────────────────────────────────────────────────────────

def dataset_tag(*names: str) -> str | None:
    """Encode source names as the canonical ',a,b,' membership string (sorted, de-duplicated).
    Accepts bare names or already-encoded tags (which are split and merged). Returns None if empty."""
    out: set[str] = set()
    for n in names:
        if not n:
            continue
        for part in str(n).strip(",").split(","):
            if part:
                out.add(part)
    return ("," + ",".join(sorted(out)) + ",") if out else None


def dataset_membership_clause(sources: list[str], column: str = "dataset") -> tuple[str, list[str]]:
    """Build an SQL predicate matching rows whose `dataset` set contains ANY of *sources*,
    plus the bound parameters. Example: ``("(dataset LIKE ? OR dataset LIKE ?)", ["%,a,%", "%,b,%"])``."""
    clause = " OR ".join(f"{column} LIKE ?" for _ in sources)
    params = [f"%,{s},%" for s in sources]
    return f"({clause})", params


def merge_provenance_by_content_hash(db: sqlite3.Connection) -> int:
    """Set every row's `dataset` to the UNION of all `dataset` sets sharing its `content_hash`,
    so provenance survives content deduplication (which keeps a single row per content_hash).

    Run BEFORE content-election dedup: while every source's copy still exists, each content_hash
    group's rows all get the merged tag, so whichever row the election keeps carries the full set.
    Only groups spanning more than one source are rewritten. Returns the number of such groups.

    Implementation note: writes are keyed on `rowid` (the intrinsic key) in a single pass, NOT
    on `content_hash` — `UPDATE … WHERE content_hash=?` would full-scan the table per group (no
    content_hash index exists at this stage), which is O(groups × rows) and pathologically slow.
    """
    from collections import defaultdict
    groups: dict[str, set] = defaultdict(set)
    row_hash: list[tuple[int, str]] = []
    for rowid, content_hash, ds in db.execute(
        "SELECT rowid, content_hash, dataset FROM pages "
        "WHERE content_hash IS NOT NULL AND dataset IS NOT NULL"
    ):
        row_hash.append((rowid, content_hash))
        for name in str(ds).strip(",").split(","):
            if name:
                groups[content_hash].add(name)

    # Precompute the merged tag only for multi-source groups.
    merged_tag = {
        content_hash: dataset_tag(*names)
        for content_hash, names in groups.items()
        if len(names) > 1
    }
    updates = [(merged_tag[ch], rowid) for rowid, ch in row_hash if ch in merged_tag]
    if updates:
        db.executemany("UPDATE pages SET dataset = ? WHERE rowid = ?", updates)
        db.commit()
    return len(merged_tag)


def rebuild_provenance_index(db: sqlite3.Connection) -> int:
    """(Re)build the ``page_datasets`` normalized index: one ``(dataset, page_rowid)`` row per
    source a page belongs to, expanded from the delimited ``pages.dataset`` sets.

    Why it matters: membership via ``dataset LIKE '%,x,%'`` can't use an index (leading wildcard),
    so each per-source pull in the derivation is a FULL TABLE SCAN. On a warm cache that scan is
    ~0.1s, but on the multi-GB canonical read cold from disk it re-reads the whole table every
    time — 50 sources × the full DB of I/O. This index turns each pull into
    ``rowid IN (SELECT page_rowid FROM page_datasets WHERE dataset = ?)`` — an index lookup that
    reads only the matching rows. Cost is one scan to build it, versus 50 scans without.

    Rebuild after any write to ``pages``. ``page_rowid`` is the pages rowid, stable under the
    incremental compaction used here; a full ``VACUUM``/repack renumbers rowids, so rebuild after
    one. Returns the number of (dataset, page) memberships indexed.
    """
    db.execute("DROP TABLE IF EXISTS page_datasets")
    db.execute("CREATE TABLE page_datasets (dataset TEXT, page_rowid INTEGER)")

    def expand():
        for rowid, ds in db.execute("SELECT rowid, dataset FROM pages WHERE dataset IS NOT NULL"):
            for name in str(ds).strip(",").split(","):
                if name:
                    yield (name, rowid)

    db.executemany("INSERT INTO page_datasets (dataset, page_rowid) VALUES (?, ?)", expand())
    db.execute("CREATE INDEX idx_page_datasets_dataset ON page_datasets (dataset)")
    db.execute("CREATE INDEX idx_page_datasets_rowid ON page_datasets (page_rowid)")
    db.commit()
    n = db.execute("SELECT COUNT(*) FROM page_datasets").fetchone()[0]
    print(f"Provenance index rebuilt: {n} (dataset, page) memberships")
    return n


def dataset_rowids_clause(source: str, column: str = "rowid") -> tuple[str, tuple]:
    """SQL predicate + params selecting pages whose provenance set contains *source* via the
    indexed ``page_datasets`` table (build it first with :func:`rebuild_provenance_index`). Example:
    ``("rowid IN (SELECT page_rowid FROM page_datasets WHERE dataset = ?)", ("pixls",))``."""
    return f"{column} IN (SELECT page_rowid FROM page_datasets WHERE dataset = ?)", (source,)


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
    # auto_vacuum must be set before journal_mode=WAL: the WAL switch writes the
    # header on a fresh file, after which the auto_vacuum change is ignored.
    db.execute("PRAGMA auto_vacuum = INCREMENTAL;")
    db.execute("PRAGMA journal_mode = WAL")
    db.execute("PRAGMA synchronous = NORMAL")
    db.execute("PRAGMA temp_store = MEMORY")
    db.execute("PRAGMA cache_size = -200000")
    db.execute("PRAGMA mmap_size = 8000000000")
    db.execute("PRAGMA busy_timeout = 30000")

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


def delete_temp_db(db: sqlite3.Connection):
    """Close and delete a temporary database in one shot."""
    filename = get_db_filename(db)
    db.close()
    os.unlink(filename)


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

    path = Path(ensure_decompressed(get_models_folder(name)))

    common_kwargs = {
        "detect_types": sqlite3.PARSE_DECLTYPES,
        "check_same_thread": False,
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

        # Enable in-place free-page reclaim (compress_db's cheap path). On a new
        # file this takes effect immediately; on a pre-existing one it is pending
        # until the next full repack (VACUUM), which compress_db performs. This
        # MUST run before journal_mode=WAL: switching journal mode writes the DB
        # header on a fresh file, after which the auto_vacuum change is ignored.
        db.execute("PRAGMA auto_vacuum = INCREMENTAL")

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

    # Add regex support to SQLite3
    def regexp(pattern, string):
        return re.search(pattern, string, re.IGNORECASE, concurrent=True) is not None

    db.create_function("regexp", 2, regexp, deterministic=True)

    return db


def get_db_filename(db: sqlite3.Connection) -> str:
    return db.execute("PRAGMA database_list").fetchone()[2]


def close_db(db: sqlite3.Connection):
   # A read-only connection (open_db(mode="ro") sets query_only = ON, or the file
   # is immutable) can neither vacuum nor commit — just close it.
   if db.execute("PRAGMA query_only").fetchone()[0]:
       db.close()
       return

   # incremental_vacuum does its work as its result rows are stepped, so it must
   # be drained to run fully (a bare execute frees at most one page).
   db.execute("PRAGMA incremental_vacuum").fetchall()
   db.commit()
   db.close()


def ensure_incremental_autovacuum(db: sqlite3.Connection) -> bool:
    """Guarantee the database uses ``auto_vacuum = INCREMENTAL`` so that
    [core.database.compress_db][] can reclaim space via the cheap in-place
    ``PRAGMA incremental_vacuum`` instead of a full ``VACUUM`` copy.

    ``auto_vacuum`` is a header setting that only takes effect on a fresh file or
    after a full ``VACUUM``. A DB created before this policy (or opened in a mode
    that never set it) reads back ``auto_vacuum = 0 (NONE)``; for those, every
    ``compress_db(repack=False)`` silently falls back to a full 8 GB rewrite. This
    performs the **one-time** conversion (set the pragma, then a single ``VACUUM``);
    subsequent daily runs are cheap and this becomes a no-op.

    Returns:
        ``True`` if a conversion VACUUM was performed, ``False`` if the DB already
        carried an auto-vacuum mode (nothing to do).
    """
    mode = db.execute("PRAGMA auto_vacuum").fetchone()[0]
    if mode != 0:
        return False  # already INCREMENTAL (2) or FULL (1)

    print("Converting DB to auto_vacuum=INCREMENTAL (one-time full VACUUM)…")
    # The pragma is recorded but only applied by the next VACUUM. VACUUM cannot run
    # inside a transaction, so force autocommit. In WAL mode, checkpoint/switch to a
    # rollback journal first so the rewrite lands in the main file (mirrors compress_db).
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute("PRAGMA journal_mode = DELETE")
    db.execute("PRAGMA auto_vacuum = INCREMENTAL")
    prev_isolation = db.isolation_level
    db.isolation_level = None
    try:
        db.execute("VACUUM")
    finally:
        db.isolation_level = prev_isolation
    return True


def compress_db(db: sqlite3.Connection, delete_query: str | None = None, delete_params: tuple | None = None, delete_columns: list[str] | None = None, repack: bool = False):
    """
    Optionally delete rows, then reclaim SQLite disk space.

    Two reclaim strategies, picked automatically:

    * Incremental (cheap, default): when the database was created with
      ``auto_vacuum = INCREMENTAL`` (see :func:`open_db`), free pages are
      returned to the OS *in place* via ``PRAGMA incremental_vacuum``. No full
      copy is made, so this needs no scratch space and cannot hit the
      "database or disk is full" trap. It does **not** defragment.

    * Full repack (``repack=True``, or as a fallback when the DB predates the
      ``auto_vacuum`` setting): rewrites the whole DB tightly via
      ``VACUUM INTO`` + online backup. Defragments and, as a side effect,
      applies any pending ``auto_vacuum`` mode change so legacy DBs convert to
      incremental on their first full repack.

    Args:
        db: SQLite connection
        delete_query: full DELETE SQL query
        delete_params: optional SQL parameters
        delete_columns: columns to NULL out before reclaiming space
        repack: force a full defragmenting rewrite (use for slim deliverables)
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

        # content_hash is the SHA-1 of `parsed`; it must never outlive the text it
        # fingerprints. If `parsed` is nulled here, null content_hash alongside it so a
        # later incremental parse/dedup can't trust a hash that describes absent content.
        if "parsed" in columns and "content_hash" in valid_columns and "content_hash" not in columns:
            columns.append("content_hash")

        if columns:
            set_clause = ", ".join(f"{col} = NULL" for col in columns)
            db.execute(f"UPDATE pages SET {set_clause}")
            db.commit()
            print(f"Deleted columns {", ".join(columns)}")

    # Reclaim disk space on disk.
    db.commit()
    db_path = get_db_filename(db)

    # Cheap path: when the DB carries auto_vacuum (INCREMENTAL/FULL), return
    # free pages to the OS in place. No copy, no scratch file, so this can never
    # hit the "disk is full" trap and costs almost no I/O. A pending auto_vacuum
    # change reads back as 0 here, so this only triggers once the DB is actually
    # converted (which a prior full repack below does for legacy files).
    auto_vacuum = db.execute("PRAGMA auto_vacuum").fetchone()[0]

    if not repack and auto_vacuum != 0:
        # NB: incremental_vacuum is a result-producing PRAGMA that does its work
        # as its rows are stepped — it must be drained (fetchall) to run fully.
        db.execute("PRAGMA incremental_vacuum").fetchall()
        db.commit()
        # In WAL mode the truncation only lands in the main file on checkpoint.
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return

    # Full repack path.
    #
    # In WAL mode the pages freed by the UPDATE/DELETE above pile up in the
    # `-wal` sidecar; without a checkpoint they never fold back into the main
    # file, so it does not shrink and a stale (smaller) `-wal` lingers next to
    # the deliverable. We therefore checkpoint the WAL and switch to a rollback
    # journal so the rewrite below touches the *main* file and leaves no `-wal`.
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    db.execute("PRAGMA journal_mode = DELETE")

    # In-memory databases (":memory:" → empty path) have no file to compact on
    # disk; a plain VACUUM reorganizes them without ever touching disk.
    if not db_path:
        prev_isolation = db.isolation_level
        db.isolation_level = None
        try:
            db.execute("VACUUM")
        finally:
            db.isolation_level = prev_isolation
        return

    # Plain VACUUM copies the whole database into a scratch file placed in
    # SQLite's temp directory, which defaults to /var/tmp — frequently the root
    # filesystem, which may be small/near-full. That raises "database or disk is
    # full" even when the volume holding the DB has plenty of room.
    #
    # VACUUM INTO instead writes a freshly compacted copy *directly* to a named
    # file, so we can land it on the DB's own (roomy) volume. We then fold that
    # copy back into the live connection with the online backup API: the backup
    # writes through this connection's own pager, so the caller's connection
    # stays valid and immediately sees the compacted content (no close/reopen).
    # VACUUM INTO also applies any pending auto_vacuum mode change, so legacy
    # DBs convert to INCREMENTAL here and use the cheap path from then on.
    db_dir = os.path.dirname(os.path.abspath(db_path))
    tmp_path = os.path.join(db_dir, f".{os.path.basename(db_path)}.vacuum")

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # VACUUM cannot run inside a transaction; force autocommit for it.
    prev_isolation = db.isolation_level
    db.isolation_level = None
    try:
        db.execute("VACUUM INTO ?", (tmp_path,))
    finally:
        db.isolation_level = prev_isolation

    compacted = sqlite3.connect(tmp_path)
    try:
        # source.backup(target) → overwrite the live DB with the compacted copy,
        # truncating the destination file to the compacted size on completion.
        compacted.backup(db)
    finally:
        compacted.close()
        os.remove(tmp_path)


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


def _on_conflict_sql(
    columns: list[str],
    pk_cols: list[str],
    preserve_derived: list[str] | None = None,
    hash_column: str = "content_hash",
) -> str:
    """
    Build the trailing ON CONFLICT … fragment for an upsert.

    Returns an empty string when *pk_cols* is empty (no PK → plain INSERT).
    Returns DO NOTHING when all columns are part of the PK (nothing to update).

    Arguments:
        preserve_derived:
            columns whose existing value must be KEPT when the row's content is
            unchanged, and only overwritten (typically reset to NULL by a
            freshly-crawled source) when the content changed. "Unchanged" is
            decided by comparing the destination and source *hash_column*. This
            avoids invalidating expensive derived artifacts (tokenized, stemmed,
            vectorized) for pages that were merely re-crawled without changing.
            Columns listed here that are part of the PK, or equal to
            *hash_column*, are ignored.

        hash_column:
            the column holding the content fingerprint used to detect changes.
    """
    if not pk_cols:
        return ""

    non_pk = [col for col in columns if col not in pk_cols]
    target = "(" + ", ".join(pk_cols) + ")"

    if not non_pk:
        return f"ON CONFLICT{target} DO NOTHING"

    preserve = set(preserve_derived or ())
    preserve.discard(hash_column)
    preserve.difference_update(pk_cols)

    assignments = []
    for col in non_pk:
        if col in preserve:
            # Keep the existing derived value when the content fingerprint is
            # unchanged (NULL-safe compare); otherwise take the incoming value
            # (NULL from a freshly-crawled source), which forces recomputation.
            assignments.append(
                f"{col}=CASE WHEN pages.{hash_column} IS excluded.{hash_column} "
                f"THEN pages.{col} ELSE excluded.{col} END"
            )
        else:
            assignments.append(f"{col}=excluded.{col}")

    updates = ", ".join(assignments)
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
    preserve_derived: list[str] | None = None,
    skip_unchanged: bool = False,
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
        on_conflict = _on_conflict_sql(dest_cols, pk_cols, preserve_derived)

        # Skip source rows whose (url, content_hash) already exist in the destination —
        # only genuinely new / content-changed rows are imported (NULL-safe compare).
        unchanged_filter = ""
        if skip_unchanged and "content_hash" in src_cols and "content_hash" in dest_cols:
            unchanged_filter = """
              AND NOT EXISTS (
                  SELECT 1 FROM pages d
                  WHERE d.url = _src.pages.url
                    AND d.content_hash IS _src.pages.content_hash
              )
            """

        cursor = dest.execute(f"""
            INSERT INTO pages ({quoted})
            SELECT {select_list} FROM _src.pages WHERE {where_clause}
            {unchanged_filter}
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
    preserve_derived: list[str] | None = None,
    skip_unchanged: bool = False,
    existing_keys: set | None = None,
) -> int:
    dest_cols = _table_columns(dest, "pages")
    src_cols  = set(_table_columns(source, "pages"))
    pk_cols   = _table_pk(dest, "pages")                              # ← dynamic

    select_list = ", ".join(
        col if col in src_cols else f"NULL AS {col}"
        for col in dest_cols
    )

    quoted       = ", ".join(dest_cols)
    placeholders = ", ".join("?" * len(dest_cols))
    on_conflict  = _on_conflict_sql(dest_cols, pk_cols, preserve_derived)  # ← dynamic
    insert_sql = f"INSERT INTO pages ({quoted}) VALUES ({placeholders}) {on_conflict}"

    # Skip source rows whose (url, content_hash) already exist in the destination, so the merge
    # scales with the delta. `existing_keys` lets a caller doing MANY imports into one dest (e.g.
    # the search-index derivation, 50 per-source pulls) load that set ONCE instead of re-querying
    # the growing dest every call.
    do_skip = skip_unchanged and "content_hash" in src_cols and "content_hash" in dest_cols
    if do_skip:
        url_i  = dest_cols.index("url")
        hash_i = dest_cols.index("content_hash")
        existing = existing_keys if existing_keys is not None else set(
            dest.execute("SELECT url, content_hash FROM pages"))

    # Stream in batches instead of fetchall(): a source's rows carry full `content`/`parsed`
    # text (tens of KB each), so materializing an entire source at once can consume gigabytes
    # and OOM. fetchmany caps resident memory to one batch regardless of source size.
    cursor = source.execute(f"SELECT {select_list} FROM pages WHERE {where_clause}", params)
    total = 0
    BATCH = 2048
    while True:
        batch = cursor.fetchmany(BATCH)
        if not batch:
            break
        if do_skip:
            batch = [r for r in batch if (r[url_i], r[hash_i]) not in existing]
        if batch:
            dest.executemany(insert_sql, batch)
            total += len(batch)

    dest.commit()
    return total


@timeit()
def import_pages(
    source_db: str | sqlite3.Connection,
    destination_db: str | sqlite3.Connection,
    where_clause: str = "1=1",
    params: tuple = (),
    preserve_derived: list[str] | None = None,
    skip_unchanged: bool = False,
    existing_keys: set | None = None,
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

        preserve_derived:
            columns whose existing value in the destination must be preserved
            when a conflicting (same-``url``) row's content is unchanged, and
            only overwritten when the content changed (detected via
            ``content_hash``). Use this when merging a freshly-crawled source
            that has not computed these derived columns yet, so re-crawling an
            unchanged page does not wipe its expensive artifacts
            (e.g. ``["tokenized", "stemmed", "vectorized"]``). ``None`` keeps
            the plain "overwrite everything" upsert behaviour.

        skip_unchanged:
            when ``True``, source rows whose ``(url, content_hash)`` pair already
            exists in the destination are not imported at all. This makes the merge
            scale with the delta: re-importing a source whose pages are mostly
            unchanged touches only the genuinely new or content-changed rows, instead
            of upserting every row every run. Requires a ``content_hash`` column on
            both sides (ignored otherwise). Combine with *preserve_derived* so the few
            changed rows still keep/refresh their derived columns correctly.

    Returns:
        Number of affected rows (rows actually imported; with *skip_unchanged* this is
        the size of the delta).

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
            rowcount = _import_via_bridge(source_db, dest, where_clause, params, preserve_derived, skip_unchanged, existing_keys)
        else:
            # File paths can be ATTACHed for a single-statement INSERT … SELECT.
            rowcount = _import_via_attach(
                get_models_folder(source_db), dest, where_clause, params, preserve_derived, skip_unchanged
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


# ─────────────────────────────────────────────────────────────────────────────────────
# Delta sync
#
# The crawler maintains the canonical DB on the (weak, always-on) crawl server. The (powerful,
# per-task) machine keeps its own copy and only needs the pages that changed since it last
# synced. Because every write stamps `crawled = now`, the delta is simply the rows with
# `crawled` newer than the target's own MAX(crawled) — no separate sync-state to track.
#
# Deletions are intentionally NOT propagated: the canonical is an archive that preserves
# material even after it goes offline (see chantal-96 archival), so the target only ever
# gains rows. A page whose content changed re-arrives with a new `crawled` and replaces its
# old row via delete-by-url in apply_delta.
# ─────────────────────────────────────────────────────────────────────────────────────

def latest_crawled(db_name: str) -> str | None:
    """MAX(`crawled`) of a canonical DB — the watermark a puller passes as `since` to get only
    newer rows. Returns None for an empty/absent DB."""
    try:
        db = open_db(db_name, mode="ro")
    except Exception:
        return None
    try:
        return db.execute("SELECT MAX(crawled) FROM pages").fetchone()[0]
    finally:
        db.close()


def export_delta(source_db: str, out_name: str, since: str | None = None) -> tuple[int, str | None]:
    """Write into a fresh, transfer-sized SQLite DB (`out_name`) every source row with
    ``crawled > since`` (all rows when *since* is None), preserving each row's `dataset`
    provenance. Run on the crawl server; ship `out_name` to the puller.

    Returns ``(row_count, max_crawled_in_delta)``.
    """
    out = create_db(out_name, url_primary_key=False)  # same schema, no PK (multi-variant allowed)
    if since is None:
        n = import_pages(source_db, out)
    else:
        n = import_pages(source_db, out, where_clause="crawled > ?", params=(since,))
    max_crawled = out.execute("SELECT MAX(crawled) FROM pages").fetchone()[0]
    close_db(out)
    print(f"Delta export: {n} rows crawled after {since!r} → {out_name}")
    return n, max_crawled


def apply_delta(delta_name: str, target_db: str) -> int:
    """Merge a delta DB (from :func:`export_delta`) into the target canonical copy.

    For every URL present in the delta, the target's existing rows for that URL are dropped and
    the delta's authoritative (already source-deduplicated, provenance-merged) rows inserted —
    so an updated page replaces its old version while brand-new pages are simply added. A light
    in-place dedup then resolves any content collisions the delta introduces against pre-existing
    target rows. Returns the number of rows applied.
    """
    tgt = create_db(target_db, url_primary_key=False)
    ensure_incremental_autovacuum(tgt)

    delta = open_db(delta_name, mode="ro")
    urls = [u for (u,) in delta.execute("SELECT DISTINCT url FROM pages WHERE url IS NOT NULL")]
    delta.close()

    # Replace the target's rows for the changed/new URLs, then insert the delta's rows.
    tgt.executemany("DELETE FROM pages WHERE url = ?", [(u,) for u in urls])
    tgt.commit()
    n = import_pages(delta_name, tgt)

    # Resolve any content duplicates the delta created against pre-existing target rows.
    # Deferred import: deduplicator imports database at module load, so a top-level import here
    # would be circular.
    from . import deduplicator
    deduplicator.Deduplicator(threshold=1.0).run_incremental(tgt, changed_urls=urls)

    compress_db(tgt)
    close_db(tgt)
    print(f"Delta apply: {n} rows merged into {target_db} ({len(urls)} URLs touched)")
    return n


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


def inspect_db(db: sqlite3.Connection, message: str = "") -> None:
    """Print useful metadata and statistics about a SQLite database.

    Arguments:
        db: active database connection
        message: optional additional message to indentify several inspections if any.
    
    """

    cur = db.cursor()

    # Database info
    print("=" * 80)
    print("DATABASE", get_db_filename(db), message)
    print("=" * 80)

    print("SQLite version:", sqlite3.sqlite_version)

    db_info = cur.execute("PRAGMA database_list").fetchall()

    for _, name, path in db_info:
        print(f"\n{name}:")
        print(f"  path: {path or ':memory:'}")

        if path and Path(path).exists():
            size = Path(path).stat().st_size / (1024 * 1024)
            print(f"  size: {size:.2f} MB")

    # Tables
    tables = [
        row[0]
        for row in cur.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
    ]

    print("\n" + "=" * 80)
    print("TABLES")
    print("=" * 80)

    for table in tables:
        print(f"\n[{table}]")

        row_count = cur.execute(
            f'SELECT COUNT(*) FROM "{table}"'
        ).fetchone()[0]

        print(f"Rows: {row_count:,}")

        columns = cur.execute(
            f'PRAGMA table_info("{table}")'
        ).fetchall()

        print("\nColumns:")

        for cid, name, col_type, notnull, default, pk in columns:

            null_count = cur.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE "{name}" IS NULL'
            ).fetchone()[0]

            empty_count = None

            if col_type.upper() in ("TEXT", "", "VARCHAR", "CHAR"):
                empty_count = cur.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE "{name}" = ""'
                ).fetchone()[0]

            print(
                f"  - {name:<20}"
                f"type={col_type or 'UNKNOWN':<10}"
                f" pk={pk}"
                f" nulls={null_count:,}"
                + (
                    f" empty={empty_count:,}"
                    if empty_count is not None
                    else ""
                )
            )

        # Indexes
        indexes = cur.execute(
            f'PRAGMA index_list("{table}")'
        ).fetchall()

        if indexes:
            print("\nIndexes:")

            for _, index_name, unique, origin, partial in indexes:
                cols = [
                    row[2]
                    for row in cur.execute(
                        f'PRAGMA index_info("{index_name}")'
                    )
                ]

                print(
                    f"  - {index_name}"
                    f" ({', '.join(cols)})"
                    f"{' UNIQUE' if unique else ''}"
                )

    print("\n" + "=" * 80)

    # Global stats
    page_count = cur.execute(
        "PRAGMA page_count"
    ).fetchone()[0]

    page_size = cur.execute(
        "PRAGMA page_size"
    ).fetchone()[0]

    freelist = cur.execute(
        "PRAGMA freelist_count"
    ).fetchone()[0]

    print("GLOBAL STATS")
    print("=" * 80)
    print(f"Pages:          {page_count:,}")
    print(f"Page size:      {page_size:,} bytes")
    print(f"Free pages:     {freelist:,}")
    print(f"Used size:      {(page_count * page_size) / 1024 / 1024:.2f} MB")
    print()