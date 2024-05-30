"""
Create an SQLite database of `web_pages` to be used by a search engine.

© 2024 - Aurélien Pierre
"""

import sqlite3
import io
import numpy as np

from itertools import batched
import pickle
import json
from datetime import datetime

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

def load_pickle(text):
    return json.loads(text)

def dump_pickle(blob):
    return json.dumps(blob)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)
sqlite3.register_adapter(list, dump_pickle)
sqlite3.register_converter("list", load_pickle)


def create_db(name: str) -> sqlite3.Connection:
  """Create or recreate (overwrite) a new table of `web_page` items.

  Warning: The columns are initialized straight from the keys of `web_page`.
  """
  # Note: detect_types is mandatory for custom types support
  connector = sqlite3.connect(get_models_folder(name), detect_types=sqlite3.PARSE_DECLTYPES)
  cursor = connector.cursor()

  columns = []

  for key, value in web_page.__annotations__.items():
      if value == str:
          columns.append(f"{key} TEXT")
      elif value == int:
          columns.append(f"{key} INTEGER")
      elif value == datetime:
          columns.append(f"{key} DATETIME")
      elif value == np.ndarray:
          columns.append(f"{key} ARRAY")
      elif value == list:
          columns.append(f"{key} LIST")

  cursor.execute("DROP TABLE IF EXISTS pages")
  cursor.execute(f"CREATE TABLE pages({", ".join(columns)})")
  print(cursor.execute("pragma table_info(pages)").fetchall())
  return connector


def open_db(name: str) -> sqlite3.Connection:
  """Create or recreate (overwrite) a new table of `web_page` items.

  Warning: The columns are initialized straight from the keys of `web_page`.
  """
  # Note: detect_types is mandatory for custom types support
  return sqlite3.connect(get_models_folder(name), detect_types=sqlite3.PARSE_DECLTYPES, isolation_level=None)


def close_db(db: sqlite3.Connection):
   db.execute("VACUUM")
   db.close()


def populate_db(db: sqlite3.Connection, pages: list[web_page]):
  """Write a list of `web_page` to the SQLite database.

  Warning: array-like Python values are converted to `bytearray` then to `bytes` in order
  to be handled as `BLOB` by SQLite. Proper decoding care will be needed when fetching
  DB entries.
  """

  # Process by batches of 512 records for good memory vs. speed trade-off
  pages_tuple = batched([tuple(sanitize_web_page(elem, to_db=True).values()) for elem in pages], 512)
  cursor = db.cursor()
  columns = ", ".join(["?" for key in web_page.__annotations__.keys()])
  for batch in pages_tuple:
    cursor.executemany(f"INSERT INTO pages VALUES({columns})", list(batch))
    db.commit()
