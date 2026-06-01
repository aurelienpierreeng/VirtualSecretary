# Build Your Own Search Engine at Home

**Virtual Secretary** can be used for building domain-specific, semantics-aware search engines from scratch — without relying on any external API, proprietary cloud service, or pre-packaged solution. You harvest the data yourself, you train the language model yourself, and you run everything on your own hardware.

This tutorial walks through the complete pipeline: from fetching web pages and PDFs to serving a search engine in your browser. Every step uses real code drawn from the framework and is designed to run on a standard Linux machine (a recent CPU with 8+ cores and 16 GB of RAM is comfortable; less is possible but slower).

!!! tip "Live demo"
    A fully-operational instance of the photography-domain search engine built with this pipeline can be explored at **[chantal.aurelienpierre.com](https://chantal.aurelienpierre.com)**. The implementation-specific scripts and app code for that deployment are not public, but this tutorial reproduces the same patterns generically.

---

## Architecture Overview

The pipeline is linear: each stage produces artefacts consumed by the next. Steps 1–2 of the acquisition phase are covered in the **[Crawling Pages](6-crawling-pages.md)** guide; this tutorial picks up from Step 1 of the processing phase.

```
Web / PDF sources
       │
       ▼
  Crawling & PDFs   →  list[web_page]         (see Crawling Pages guide)
       │
       ▼
  1. Database        →  pages SQLite table     (on disk)
       │
       ▼
     Deduplication   →  cleaned pages table    (in-place, see Crawling Pages guide)
       │
       ▼
  2. Text processing →  parsed / tokenized /   (DB columns)
                        stemmed columns
       │
       ▼
  3. Language model  →  Tokenizer + Word2Vec   (model files)
       │
       ▼
  4. Vectorization   →  vectorized column      (DB column)
       │
       ▼
  5. Categories      →  category column        (DB column)
       │
       ▼
  6. Search index    →  Indexer joblib file    (model files)
       │
       ▼
  7. Query API       →  rank() results         (in RAM)
       │
       ▼
  8. Flask app       →  /api  +  /             (HTTP)
```

The central data structure throughout is `web_page`, a `TypedDict` whose keys map directly to SQLite columns:

| Key | Type | Purpose |
|---|---|---|
| `url` | `str` | Primary key — the canonical address of the document |
| `title` | `str` | Page title |
| `content` | `str` | Raw human-readable text extracted from the page |
| `parsed` | `str` | Normalised version of the content (lowercase, ASCII) |
| `tokenized` | `list` | Tokenized sentences, each a list of string tokens |
| `stemmed` | `list` | Same but with stemming, stopword removal, and n-grams applied |
| `vectorized` | `np.ndarray` | Semantic centroid vector for the page |
| `lang` | `str` | 2-letter ISO language code |
| `date` / `datetime` | `str` / `datetime` | Publication date |
| `category` | `str` | Arbitrary user-defined label |
| `excerpt` | `str` | Short description for result previews |

---

## Data Acquisition — Crawling and PDFs

Before any NLP processing can begin, you need a corpus of text documents. Virtual Secretary provides crawlers for HTML websites (sitemap-based and recursive link-following), direct PDF mining with optional OCR, and a `web_page` data structure for wiring in custom REST API sources.

The full crawling reference — parameters, markup selectors, `robots.txt` behaviour, deduplication patterns, and worked examples for web, PDF, YouTube and GitHub sources — is covered in the dedicated guide:

➜ **[Crawling Pages](6-crawling-pages.md)**

The key invariant to keep in mind when reading the rest of this tutorial: crawl output always goes into `database.create_temp_db()` first, gets deduplicated there, and is only then promoted to the permanent `database.create_db()` store where `url` is the primary key.

---

## Step 1 — Restoring Data

### Merging from individual datasets

In this scenario, each crawled source is stored in a seperate dataset, saved with [`utils.save_data`][core.utils.save_data]. This is useful to split the crawling step (that can be deferred to weak hardware running 24/7) and language model training (that needs powerful hardware for just a couple of hours). So the datasets can be exchanged through FTP/SSH/Rsync etc. between both computers, and different sources can be re-crawled at different frequencies.

In `VirtualSecretary/src/user_scripts`, add:

```python
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

files = ["site-a", "site-b", "site-c"] 
# should match dataset names used with utils.save_data() earlier

# Create a temporary DB and stream archives into it to avoid building a huge list in RAM
tmp_db = database.create_temp_db()

for file in files:
    # Merge all files, with possible duplicates, into the temporary DB
    db = utils.open_data(file, scheme="sql")
    database.import_pages(db, tmp_db)
    db.close()

# Step 1: normalise text — required before deduplication can compare content
batching.batch_parse_web_page(tmp_db, nlp.Tokenizer())

# Step 2: deduplicate in the temp database, while there is no primary key to violate
dedup = deduplicator.Deduplicator(threshold=1.0, discard_params=False, fix_urls=False)
dedup.urls_to_ignore += ["translate.goog", "facebook.com", "flickr.com"]
dedup(tmp_db)

# Step 3: only now import into the permanent database (URL primary key enforced)
final_db = database.create_db("my-engine.db")
database.import_pages(source_db=tmp_db, destination_db=final_db)

# Close and cleanup the temp database
database.close_db(tmp_db)
database.cleanup_temp_db()

final_db.close() # or maybe don't close yet if you are going to work more on it
```

### Fetching a full model

In this scenario, you already saved all crawled sources into a single database using `database.create_db("my-engine.db")`.

```python
from core import crawler, database, deduplicator, nlp, batching

final_db = database.open_db("my-engine.db")
# You are technically ready to go, though you can still optionally run the following:

# Step 1: normalise text — required before deduplication can compare content
batching.batch_parse_web_page(final_db, nlp.Tokenizer())

# Step 2: deduplicate in the temp database, while there is no primary key to violate
dedup = deduplicator.Deduplicator(threshold=1.0, discard_params=False, fix_urls=False)
dedup.urls_to_ignore += ["translate.goog", "facebook.com", "flickr.com"]
dedup(final_db)

final_db.close() # or maybe don't close yet if you are going to work more on it
```

### Content filtering at SQL level

Some pages should never make it into the index — authentication pages, boilerplate, machine-translated copies. Delete them from the temp database before going further, so they don't waste NLP processing time later and pollute storage:

```python
cur = db.cursor() # runs on final or temporary DB just the same

# Remove GitHub blob viewer pages (only raw code, useless for NLP)
cur.execute("DELETE FROM pages WHERE url LIKE ?", ("%/blob/%",))

# Remove session-hijacking hints that end up in Discourse forum pages
cur.execute("DELETE FROM pages WHERE content LIKE ?",
            ("%you signed in with another tab or window%",))

# Remove an entire domain that has been superseded / moved
cur.execute("DELETE FROM pages WHERE url LIKE ?", ("%old-domain.example.com%",))

# Remove pages in unsupported languages before spending any NLP budget on them.
# Language is detected during batch_parse_web_page, so run this AFTER that step.
cur.execute("DELETE FROM pages WHERE lang NOT IN ('fr', 'en') OR lang IS NULL")

db.commit()
```

!!! tip "Filtering on import"
    `database.import_pages()` accepts a `where_clause` argument appended to the internal `SELECT`, so you can filter at import time rather than after:

    ```python
    database.import_pages(
        source_db    = pixls_db,
        destination_db = tmp_db,
        where_clause = "title NOT LIKE ? AND content NOT LIKE ?",
        params       = ("%playraw%", "%presentation%"),
    )
    ```

## Step 2 — Text Processing

With a clean database, the NLP pipeline kicks in. All processing functions write directly to the database columns and are safely resumable — they skip rows that already have a value unless you pass `only_none=False`.

### Parsing and normalisation

`batch_parse_web_page` normalises Unicode, strips multi-spaces, detects the language of each page, and writes the result to the `parsed` column. This is a prerequisite for deduplication and tokenisation.

```python
from core import batching, nlp

tokenizer = nlp.Tokenizer()  # base tokenizer, no vocabulary yet
batching.batch_parse_web_page(db, tokenizer)
```

This runs across all available CPU cores. Typical throughput on a modern machine is tens of thousands of short pages per minute.

### Tokenisation

```python
# Full tokeniser, optionally with custom vocabulary replacements
from core import language

tokenizer = nlp.Tokenizer(
    replacements  = language.REPLACEMENTS,   # e.g. map "colour" → "color"
    abbreviations = language.ABBREVIATIONS,  # e.g. expand "ca." → "circa"
)

# Tokenise without stemming — preserves enough structure for n-gram discovery.
# only_none=False forces re-tokenisation even for rows already processed.
batching.batch_tokenize(db, tokenizer, only_none=False)
```

The tokenised form stored in `tokenized` retains stopwords, punctuation meta-tokens, and original casing. This "lossless" representation is later consumed by the n-gram trainer and can be used to reconstruct approximate original sentences.

!!! tip "Language filtering before tokenisation"
    Snowball stemming — used in the next stages — only supports a defined set of languages. Tokenising pages in unsupported languages wastes CPU and produces noisy tokens. If your database still contains mixed-language content at this point, remove unsupported languages before running `batch_tokenize`:

    ```python
    # Keep only the languages your pipeline supports
    db.execute("DELETE FROM pages WHERE lang NOT IN ('fr', 'en') OR lang IS NULL")
    db.commit()
    ```

    This should ideally have been done earlier (see [Step 1](#step-1-restoring-data)), but it is safe to run it again here as a safeguard — particularly when the permanent database is populated from multiple sources built at different times.

### Training n-grams

Multi-word expressions like "signal-to-noise ratio", "colour science", or "Aurélien Pierre" should be treated as single vocabulary units. The tokeniser can learn them automatically from co-occurrence statistics:

```python
from core import database

# Train n-grams on French corpus
fr_corpus = database.SQLitePageCorpus(
    db,
    "SELECT tokenized FROM pages WHERE lang = 'fr'",
    max_depth=1,   # yields a flat list of token strings
)
tokenizer.train_ngrams(
    fr_corpus,
    # Common function words to exclude from multi-word expressions
    " le la l l' du de d d' sur sous en les des pour au aux "
)

# Train on English corpus
en_corpus = database.SQLitePageCorpus(
    db,
    "SELECT tokenized FROM pages WHERE lang = 'en'",
    max_depth=1,
)
tokenizer.train_ngrams(
    en_corpus,
    " a an the for of with at from to in on by and or "
)

tokenizer.save("my-tokenizer")
```

After saving, reload to verify integrity before committing to the slow Word2Vec training step:

```python
tokenizer = nlp.Tokenizer.load("my-tokenizer")

# Quick sanity check
text = "Aurélien Pierre developed Ansel from the darktable codebase"
tokens = tokenizer.tokenize_document_flat(
    tokenizer.normalize_text(text),
    normalize=True, stem=True, remove_stopwords=True, n_grams=True
)
print(tokens)
# e.g. ['aurelien_pi', 'develop', 'ansel', 'darktabl', 'codebas']
```

### Stemming

With n-grams trained and saved, re-process every page to produce the stemmed form used by Word2Vec. Stemming is destructive and will remove suffixes, double consonnants etc. so the text representation symbolically encodes semantics (topics) but looses syntax. Stemmed text cannot be reconstructed into original sentences, doesn't read nicely for humans, and looses some nuance, at the benefit of being better generalized for information retrieval tasks.

```python
batching.batch_stem(db, tokenizer, only_none=False)
```

If you ever need to re-stem only a specific subset — for instance after re-crawling one site — pass a URL list to avoid touching the rest of the database, because stemming is quite CPU-intensive so you may want to blindly run it on all data entries:

```python
updated_urls = ["https://ansel.photos/doc/page1", "https://ansel.photos/doc/page2"]
batching.batch_stem(db, tokenizer, urls=updated_urls)
```

---

## Step 3 — Training the Language Model

Virtual Secretary's semantic search is powered by Word2Vec with dual embedding spaces, as described in [Bojanowski et al. (2016)](https://arxiv.org/pdf/1602.01137.pdf). You train it directly on your own corpus, which means the resulting embedding space reflects the specific vocabulary and concepts of your domain — photography, medicine, law, or whatever you are building around.

```python
from core import nlp, database

db = database.open_db("my-engine.db", mode="ro")

# Only train on languages with enough data for reliable semantics
corpus = database.SQLitePageCorpus(
    db,
    "SELECT stemmed FROM pages WHERE lang IN ('fr', 'en')",
    max_depth=0,   # yields a list-of-list-of-strings (one inner list per sentence)
)

w2v = nlp.Word2Vec(
    corpus,
    "my-word2vec",
    vector_size  = 496,   # embedding dimensionality
    epochs       = 40,
    window       = 31,    # context window (wider = more thematic, narrower = more syntactic)
    min_count    = 10,    # ignore words appearing fewer than 10 times
    sample       = 1e-4,  # sub-sampling threshold for frequent words
    ns_exponent  = -0.5,  # negative sampling exponent
    negative     = 5,     # negative samples per positive example
    tokenizer    = nlp.Tokenizer.load("my-tokenizer"),
)

db.close()
```

Validate the embeddings before moving on:

```python
print(w2v.wv.most_similar("luminanc"))      # stemmed form of "luminance"
# [('brightn', 0.91), ('exposur', 0.88), ('lux', 0.86), ...]

print(w2v.wv.most_similar("raw_file"))      # an n-gram treated as single token
# [('dng', 0.89), ('raw_format', 0.87), ('nef', 0.84), ...]
```

If the top neighbours are semantically coherent, the model is ready.

!!! note "Training time"
    Expect 5–30 minutes on 8 cores for a corpus of 100 000 pages. Training is parallelised internally by Gensim.

---

## Step 4 — Vectorising the Index

The next step embeds every page as a single centroid vector that will later power the semantic ranking:

```python
from core import batching, nlp, database

db  = database.open_db("my-engine.db")
w2v = nlp.Word2Vec.load_model("my-word2vec")

batching.batch_vectorize(db, w2v)

db.close()
```

Each vector is written to the `vectorized` column as a binary blob. This step is idempotent and parallelised across all CPU cores.

---

## Step 5 — Setting Up Categories

Certain search filters (exclude forums, restrict to GitHub, etc.) rely on a `category` column that is set at crawl time or updated with a bulk SQL pass before building the index. Do this now if you haven't done so during crawling:

```python
db = database.open_db("my-engine.db")

db.execute("""
    UPDATE pages SET category = 'forum'
    WHERE url LIKE '%forum%'
       OR url LIKE '%discuss.%'
       OR url LIKE '%/t/%'
""")

db.execute("""
    UPDATE pages SET category = 'Github'
    WHERE url LIKE '%github.com%'
""")

db.commit()
db.close()
```

You can define as many categories as you want; the Indexer will make them available as filter facets at query time.

---

## Step 6 — Building the Search Index

The `Indexer` class builds the in-memory data structures required for fast ranking:

- a BM25+ sparse matrix over the vocabulary for exact keyword matching,
- the full matrix of page vectors for semantic matching,
- a URL-to-position lookup table.

```python
from core import search, database, nlp

db  = database.open_db("my-engine.db")
w2v = nlp.Word2Vec.load_model("my-word2vec")

# Build and immediately save the index
model = search.Indexer(
    db,
    "my-search-engine",   # name used to save the .joblib file
    w2v,
    principal_components = 2,   # number of principal components to subtract from query vectors
                                # to remove the "all topics" bias
)
db.close()
```

Test a few queries while the index is still live in RAM, before you're sure it serialised correctly:

```python
from core.utils import typography_undo

queries = [
    "install darktable on ubuntu",
    "difference between lightness and brightness",
    "meilleur appareil photo pour débutant",
]

for q in queries:
    tokens = model.tokenize_query(typography_undo(q))
    results = model.rank(db, tokens, search.search_methods.AI)
    print(f"\n[{q}]")
    for rank, url, score in results[:5]:
        print(f"  {score:.3f}  {url}")
```

Then verify the on-disk version loads cleanly:

```python
model = search.Indexer.load("my-search-engine", db)
```

---

## Step 7 — Querying the Index

The `Indexer.rank()` method supports three complementary search modes, which can be combined:

### Semantic (AI) search

Uses the dual Word2Vec embedding space — the most powerful mode for natural-language queries where the exact words may not appear in the document:

```python
tokens  = model.tokenize_query("how to manage exposure in darktable")
results = model.rank(db, tokens, search.search_methods.AI)
```

### Fuzzy (BM25+) search

Classic statistical keyword weighting. Useful for precise queries where exact word frequency matters:

```python
tokens  = model.tokenize_query("filmic rgb tone mapping")
results = model.rank(db, tokens, search.search_methods.FUZZY)
```

### SQL-filtered search

Any `rank()` call accepts an optional SQL `WHERE` clause that intersects with the semantic ranking, enabling powerful faceted search:

```python
# Only French-language results from forum sources
tokens  = model.tokenize_query("courbe des tons")
results = model.rank(
    db,
    tokens,
    search.search_methods.AI,
    n_results  = 500,
    sql_query  = "WHERE lang = ? AND category = ?",
    sql_params = ["fr", "forum"],
)

# Full-text filter — pages whose parsed content contains a specific string
results = model.rank(
    db, tokens, search.search_methods.AI,
    sql_query  = "WHERE instr(parsed, ?) > 0",
    sql_params = ["zone system"],
)

# PCRE regex filter — supported transparently via the custom SQLite extension
results = model.rank(
    db, tokens, search.search_methods.AI,
    sql_query  = "WHERE parsed REGEXP ?",
    sql_params = [r"colour\s+science"],
)
```

The result list is a list of `(index, url, similarity_score)` tuples, sorted best-first. Retrieve full page data with a follow-up database query:

```python
urls = [url for _, url, _ in results[:20]]
placeholders = ",".join(["?" for _ in urls])
cursor = db.execute(
    f"SELECT title, url, excerpt, datetime FROM pages WHERE url IN ({placeholders})",
    urls
)
for title, url, excerpt, date in cursor.fetchall():
    print(f"{title}\n  {url}\n  {excerpt[:120]}\n")
```

---

## Step 8 — The Web Interface

The section below shows a minimal implementation of a web API exposing the `search.Indexer` object to search queries and returning responses as JSON object for asynchronous AJAX loading or as a basis to build Rest APIs. This implementation uses additional SQL filtering to narrow-down the search results.

### Minimal Flask app

```python
# search_app.py
import os
os.environ['OPENBLAS_NUM_THREADS'] = '2'  # avoid OpenBLAS threading conflicts

from flask import Flask, request, jsonify, g
from core import database, search
from core.utils import typography_undo
import html

app = Flask(__name__)

DB_PATH    = "my-engine.db"
INDEX_NAME = "my-search-engine"

# ── Database helpers ──────────────────────────────────────────────────────────

def get_db():
    """Open one read-only DB connection per request."""
    if "_db" not in g:
        g._db = database.open_db(DB_PATH, mode="ro")
    return g._db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("_db", None)
    if db:
        db.close()

# ── Index — loaded once at startup ───────────────────────────────────────────

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        db = database.open_db(DB_PATH, mode="ro")
        _engine = search.Indexer.load(INDEX_NAME, db)
        db.close()
    return _engine

# ── Search endpoint ───────────────────────────────────────────────────────────

@app.route("/api")
def api():
    raw_query = request.args.get("s", "").strip()
    page      = int(request.args.get("page", 0))
    lang      = request.args.get("lang", "any")
    category  = request.args.get("category", "any")

    if not raw_query:
        return jsonify({"error": "empty query"}), 400

    # Sanitise and tokenise
    query  = html.unescape(raw_query)
    tokens = get_engine().tokenize_query(typography_undo(query))

    if not tokens:
        return jsonify({"error": "no recognisable keywords in query"}), 400

    # Build optional SQL filter
    clauses, params = [], []
    if lang != "any":
        clauses.append("lang = ?")
        params.append(lang)
    if category != "any":
        clauses.append("category = ?")
        params.append(category)

    sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    # Rank
    results = get_engine().rank(
        get_db(), tokens, search.search_methods.AI,
        n_results  = 500,
        sql_query  = sql,
        sql_params = params,
    )

    # Paginate
    per_page  = 20
    n_pages   = max(1, -(-len(results) // per_page))   # ceiling division
    page      = min(page, n_pages - 1)
    slice_    = results[page * per_page : (page + 1) * per_page]

    # Fetch display fields
    urls = [url for _, url, _ in slice_]
    if not urls:
        return jsonify({"error": "no results"}), 200

    ph   = ",".join(["?"] * len(urls))
    rows = get_db().execute(
        f"SELECT title, url, excerpt, datetime, category FROM pages WHERE url IN ({ph})",
        urls,
    ).fetchall()

    # Re-sort rows to match ranking order
    row_map = {row[1]: row for row in rows}
    items   = [
        {"title": row_map[u][0], "url": u,
         "excerpt": row_map[u][2], "date": str(row_map[u][3]),
         "category": row_map[u][4]}
        for u in urls if u in row_map
    ]

    return jsonify({
        "query":    query,
        "tokens":   tokens,
        "page":     page,
        "n_pages":  n_pages,
        "n_results": len(results),
        "results":  items,
    })

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
```

Run it with:

```bash
python search_app.py
# or in production:
gunicorn -w 2 -b 0.0.0.0:5000 search_app:app
```

Query it from the command line or a browser:

```
GET http://localhost:5000/api?s=exposure+compensation&lang=en
```

### Minimal HTML front-end

Drop a single `index.html` next to `search_app.py` and add a `render_template` call to the `/` route if you want a browser UI:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>My Search Engine</title>
  <style>
    body  { font-family: sans-serif; max-width: 800px; margin: 2rem auto; }
    input { width: 70%; padding: .4em; font-size: 1rem; }
    button { padding: .4em 1em; font-size: 1rem; }
    .result { margin: 1.5rem 0; }
    .result a { font-size: 1.1rem; font-weight: bold; }
    .result .meta { color: #666; font-size: .85rem; }
    .result p { margin: .3em 0; }
  </style>
</head>
<body>
  <h1>🔍 My Search Engine</h1>
  <div>
    <input id="q" type="text" placeholder="Type your query…" autofocus />
    <button onclick="search()">Search</button>
  </div>
  <p id="stats"></p>
  <div id="results"></div>

  <script>
    async function search(page = 0) {
      const q = document.getElementById('q').value.trim();
      if (!q) return;

      const res  = await fetch(`/api?s=${encodeURIComponent(q)}&page=${page}`);
      const data = await res.json();

      if (data.error) {
        document.getElementById('results').innerHTML = `<p style="color:red">${data.error}</p>`;
        return;
      }

      document.getElementById('stats').textContent =
        `${data.n_results} results — page ${data.page + 1} / ${data.n_pages}`;

      document.getElementById('results').innerHTML = data.results.map(r => `
        <div class="result">
          <a href="${r.url}" target="_blank">${r.title || r.url}</a>
          <div class="meta">${r.url} · ${r.date || ''} · ${r.category || ''}</div>
          <p>${r.excerpt || ''}</p>
        </div>
      `).join('');
    }

    document.getElementById('q').addEventListener('keydown', e => {
      if (e.key === 'Enter') search();
    });
  </script>
</body>
</html>
```

### Caching with Flask-Caching

For production, wrap expensive calls with a cache. The reference app uses Memcached, but the file-system backend is zero-dependency:

```python
from flask_caching import Cache

cache = Cache(app, config={
    "CACHE_TYPE":            "FileSystemCache",
    "CACHE_DIR":             "/tmp/my-engine-cache",
    "CACHE_DEFAULT_TIMEOUT": 3600 * 48,   # 48-hour TTL
})

@cache.memoize(3600 * 48)
def get_ranked_results(tokens, lang, category):
    ...   # expensive ranking call here
```

## Advanced Topics

### Incremental updates

Because `create_db` stores pages with `url` as the primary key, re-crawling a site and calling `database.populate_db` again will update existing rows in place and insert new ones — nothing is duplicated. The recommended update cycle is:

1. Crawl changed sources into a fresh `create_temp_db()`.
2. Run `batch_parse_web_page`, language-filter, and `Deduplicator` on the temp database.
3. Import into the permanent database with `import_pages` (existing URLs are updated, new ones inserted).
4. Re-run `batch_tokenize`, `batch_stem`, and `batch_vectorize` with `only_none=True` so only the new or changed rows are processed.
5. Rebuild the `Indexer`.

Steps 4–5 are fast when most of the corpus is unchanged; only the delta gets reprocessed.

### Domain-specific language support

The `Tokenizer` delegates stemming to [Snowball](https://snowballstemmer.readthedocs.io/en/latest/), which currently supports 26 languages. The `language` detection module uses `langdetect` under the hood. Adding a new language amounts to:

1. adding its stopwords file to the `models/` directory,
2. passing it in the `stopwords=` argument at `Tokenizer()` instantiation,
3. including it in the SQL `WHERE lang IN (...)` corpus queries.

### Result ranking transparency

The `search_methods.AI` mode blends two signals: 0.98 × semantic vector similarity + 0.02 × BM25+ keyword frequency. This blend can be tuned in `search.Indexer.rank()` to shift the balance between thematic relevance and exact-term matching, depending on your domain and typical query style.

### Exposing "related keywords"

The Indexer can suggest semantically related terms for query expansion or a "did you mean?" feature:

```python
related = model.get_related(tokens, n=20, k=5)
# e.g. for tokens=['exposur', 'histogram']:
# ['highlight', 'shadow', 'tone_curv', 'clipping', 'waveform', ...]
```

---

## What's Next

With a working search engine at home you now have a foundation to explore further:

- **RAG (Retrieval-Augmented Generation)** — feed the top results from `rank()` as context to a local LLM to answer questions in natural language, grounded in your own curated corpus.
- **Monitoring** — the `stats` table built by `Indexer` tracks word counts, page counts, domain distribution, and the most recent crawl date. Surface those in your UI to show users the freshness of the index.
- **Fine-tuned relevance** — the `category` and `lang` columns are indexed in SQLite. Adding `WHERE category = 'documentation'` to a query is essentially free in terms of latency, making faceted filtering a first-class citizen.
- **Private intranet search** — the crawler handles any URL including `http://localhost:*`, so you can index internal wikis, documentation servers, or local PDF archives with exactly the same pipeline.
