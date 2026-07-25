# Build Your Own Search Engine at Home

**Virtual Secretary** can be used to build domain-specific, semantics-aware search engines from scratch — without any external API, proprietary cloud service, or pre-packaged solution. You harvest the data yourself, train the language model yourself, and run everything on your own hardware.

This tutorial walks through the complete pipeline, from a crawled corpus to a search engine served in your browser. A recent CPU with 8+ cores and 16 GB of RAM is comfortable; less is possible but slower.

!!! tip "Live demo"
    A fully-operational instance of a photography-domain search engine built with this pipeline can be explored at **[chantal.aurelienpierre.com](https://chantal.aurelienpierre.com)**. That deployment's crawl jobs, orchestration, and app code are a private reference implementation; this tutorial reproduces the same patterns generically with the public library.

---

## Architecture: one database, stages you can split

Everything lives in **one corpus database** — the same one the [crawler](6-crawling-pages.md) writes into. That single database is *both* the language-model training corpus *and* the search corpus. A boolean `in_index` column flags the rows that belong to the search index; the rest is training-only or archival material.

The pipeline is a set of stages that each read and write that one database, so they can be **deferred to different machines** — crawl on a cheap always-on box, retrain the model occasionally on a powerful one — and reconciled with a timestamp delta ([`database.export_delta`][core.database.export_delta] / [`apply_delta`][core.database.apply_delta], see [Crawling Pages](6-crawling-pages.md)).

```
                          corpus.db  (the one database)
                                 │
  1. Crawl  ─────────────────────┤   parsed / content_hash / tokenized written at crawl time
                                 │
  2. Retrain language model ─────┤   Tokenizer (n-grams) + Word2Vec/FastText  → model files
       (occasional, heavy)       │   + batch_stem / batch_vectorize columns
                                 │
  3a. Mark the search subset ────┤   in_index = 1 on the curated sources
  3b. Build the search index ────┤   Indexer  → engine.joblib
       full  (K-means + PCA)      │
       or incremental (append)    │
                                 │
  4. Serve ──────────────────────┘   Flask app  →  /api  +  /
```

Each stage is an ordinary function call, so you orchestrate them however suits you — a shell script, a Makefile, a set of cron jobs. The natural split is: **crawl** and the light **incremental index update** run daily on the server; **retraining the model** and a **full index rebuild** run occasionally on a powerful machine; the two reconcile the database with a delta sync.

### The page schema

The central structure is `web_page`, a `TypedDict` whose keys map to columns of the `pages` table:

| Key | Type | Purpose |
|---|---|---|
| `url` | `str` | Canonical address (a **plain, non-unique** index — the same URL may appear several ways) |
| `title` / `excerpt` | `str` | Display fields |
| `content` | `str` | Raw human-readable text |
| `parsed` | `str` | Normalised content — written at crawl time |
| `content_hash` | `str` | `SHA-1(parsed)` — drives content-dedup and incremental reprocessing |
| `tokenized` | `list` | Non-destructive token lists — written at crawl time |
| `stemmed` | `list` | Stemmed + stopword-filtered + n-grammed tokens (needs the trained model) |
| `vectorized` | `np.ndarray` | Document centroid vector (needs the trained model) |
| `lang` / `datetime` / `category` / `dataset` | | Language, date, user label, provenance tag(s) |

Two more columns are added by the index stages, not by the crawler: **`in_index`** (search-subset flag) and **`search_rowid`** (position in the Indexer's arrays).

---

## Step 1 — Text processing

The NLP columns are filled by resumable batch passes that **only touch rows missing a value** (`only_none=True`, the default), so re-running after a crawl reprocesses just the delta.

`parsed`, `content_hash` and `tokenized` are already written **at crawl time**, so these two passes are near-no-ops on freshly crawled rows — they only mop up rows imported by other means:

```python
from core import batching, nlp, database

db = database.open_db("corpus.db")
tokenizer = nlp.Tokenizer()

batching.batch_parse_web_page(db, tokenizer)   # writes parsed + content_hash + lang (only_none)
batching.batch_tokenize(db, tokenizer)         # writes tokenized (only_none)
```

`stemmed` and `vectorized` need the *trained* model, so they come after Step 2.

---

## Step 2 — Training the language model

The language model is trained **directly from the corpus**, streamed one row at a time so the whole thing never sits in RAM, and filtered at query time to the languages the stemmer supports (here French + English).

### N-grams

Multi-word expressions ("signal-to-noise ratio", "colour science", "Aurélien Pierre") should be single vocabulary units. The tokenizer learns them from co-occurrence statistics, streaming the `tokenized` column with [`SQLitePageCorpus`][core.database.SQLitePageCorpus] — a lazy iterable that drives the SQL cursor one row at a time (no `.fetchall()`):

```python
from core import database, nlp

db = database.open_db("corpus.db", mode="ro")
tokenizer = nlp.Tokenizer()

for lang, stop in [("fr", " le la l' du de d' des les pour au aux en sur "),
                   ("en", " a an the for of with at from to in on by and or ")]:
    corpus = database.SQLitePageCorpus(
        db, f"SELECT tokenized FROM pages WHERE lang = '{lang}'", max_depth=1,
    )
    tokenizer.train_ngrams(corpus, stop)

tokenizer.save("my-tokenizer")
db.close()
```

Reload and sanity-check before the slow embedding step:

```python
tokenizer = nlp.Tokenizer.load("my-tokenizer")
print(tokenizer.tokenize_document_flat(
    tokenizer.normalize_text("Aurélien Pierre developed Ansel from the darktable codebase"),
    normalize=True, stem=True, remove_stopwords=True, n_grams=True))
# ['aurelien_pi', 'develop', 'ansel', 'darktabl', 'codebas']
```

### Stemming

With n-grams trained, produce the `stemmed` column the embedding consumes. Stemming is destructive (it drops suffixes/syntax to generalise semantics) and CPU-heavy, so it is a batch pass keyed on `only_none`:

```python
batching.batch_stem(db, tokenizer)     # writes stemmed (only_none)
```

### Embedding

Train Word2Vec (or FastText) on the `stemmed` column of the FR/EN corpus, again streamed:

```python
db = database.open_db("corpus.db", mode="ro")

corpus = database.SQLitePageCorpus(
    db, "SELECT stemmed FROM pages WHERE lang IN ('fr','en')", max_depth=0,
)
w2v = nlp.Word2Vec(
    corpus, "my-word2vec",
    vector_size=496, epochs=40, window=31, min_count=10,
    sample=1e-4, ns_exponent=-0.5, negative=5,
    tokenizer=nlp.Tokenizer.load("my-tokenizer"),
)
db.close()

print(w2v.wv.most_similar("luminanc"))   # [('brightn', .91), ('exposur', .88), ('lux', .86), …]
```

Finally, write the `vectorized` column (document centroids) with the trained model:

```python
db  = database.open_db("corpus.db")
w2v = nlp.Word2Vec.load_model("my-word2vec")
batching.batch_vectorize(db, w2v)        # writes vectorized (only_none)
db.close()
```

!!! note "Why this lives on the powerful machine"
    Retraining the model invalidates every `stemmed`/`vectorized` value, so it triggers a full reprocess of those columns across the whole corpus — heavy, and done only occasionally. Crawling and the light *incremental* index update (below) stay on the always-on server.

---

## Step 3a — Marking the search subset

Not every crawled page belongs in the search index: some sources are training-only, some need per-source content filters, and archived duplicates must be excluded. Instead of copying a filtered subset into a second database, flag the searchable rows in place with `in_index = 1`. Selection has three parts — source inclusion, per-source content rules, and a global keep predicate — expressed as plain SQL over the corpus, using [`dataset_rowids_clause`][core.database.dataset_rowids_clause] for fast provenance-tag lookups:

```python
db = database.open_db("corpus.db", mode="rw")

PLAIN_SOURCES = ["ansel", "github", "wikipedia", "poynton", "munsell", ...]      # searchable
FILTERED = {"pixls": ("content LIKE ? OR content LIKE ?", ("%darktable%", "%ansel%"))}
KEEP = "url NOT LIKE '%/blob/%' AND (lang IS NULL OR lang IN ('fr','en'))"

db.execute("ALTER TABLE pages ADD COLUMN in_index INTEGER DEFAULT 0")
database.rebuild_provenance_index(db)                     # index the dataset tags

for src in PLAIN_SOURCES:
    clause, params = database.dataset_rowids_clause(src)
    db.execute(f"UPDATE pages SET in_index = 1 WHERE ({clause}) AND ({KEEP})", params)
    db.commit()
# … plus FILTERED sources, then a one-row-per-URL pass.
```

The key invariant: `in_index = 1` must select **exactly one row per URL** (the most recently crawled), because the web app joins results back by URL and the corpus allows duplicate URLs. This marking step is idempotent — re-run it after each crawl to flag the new rows.

---

## Step 3b — Building the search index

The [`Indexer`][core.search.Indexer] builds, from the `in_index = 1` rows only, the in-memory structures that power ranking — a BM25+ sparse index for keyword matching, the matrix of page vectors for semantic matching, the principal component(s) subtracted to sharpen queries, and K-means topic clusters — then pickles them to a `.joblib` file. Each searchable row gets a contiguous `search_rowid` aligned with those arrays.

### Full build

```python
from core import search, database, nlp

db  = database.open_db("corpus.db")
w2v = nlp.Word2Vec.load_model("my-word2vec")

model = search.Indexer(db, "engine", w2v, principal_components=2)
```

This recomputes everything (BM25, vectors, PCA eigenvectors, K-means clusters, `search_rowid`) over the whole searchable subset. It is the **heavy** path — run it after a language-model retrain, or periodically to reclaim space. Test while it is live in RAM, then confirm the on-disk copy reloads:

```python
for q in ["install darktable on ubuntu", "difference between lightness and brightness"]:
    tokens = model.tokenize_query(q)
    for _, url, score in model.rank(db, tokens, search.search_methods.MIXED)[:5]:
        print(f"  {score:.3f}  {url}")

model = search.Indexer.load("engine", db)   # verifies integrity on load
```

### Incremental build

For the daily case you do not want to recompute K-means and PCA over hundreds of thousands of documents just to add a few hundred new pages. [`update_incremental`][core.search.Indexer.update_incremental] loads the existing engine and **appends only the diff** — the searchable rows that do not yet have a `search_rowid` — reusing the stored eigenvectors and cluster centroids:

```python
model = search.Indexer.load("engine", db)
model.update_incremental(db, "engine")   # append new docs, reuse PCA + clusters
```

New documents' BM25 postings are merged into the existing index in place, their vectors are projected with the *stored* principal components (PCA is not refit), and each is assigned to the nearest existing cluster centroid (K-means is not refit). Pages that left the subset or were re-crawled leave harmless "holes" that ranking skips; a periodic full rebuild reclaims them.

So the everyday cycle is light — crawl, re-mark, `batch_stem`/`batch_vectorize` the new rows (`only_none`), `update_incremental` — while a full rebuild and a model retrain are the occasional heavy counterpart.

---

## Step 4 — Querying the index

[`Indexer.rank`][core.search.Indexer.rank] supports three complementary modes:

```python
tokens = model.tokenize_query("how to manage exposure in darktable")

model.rank(db, tokens, search.search_methods.AI)      # semantic (dual embedding)
model.rank(db, tokens, search.search_methods.FUZZY)   # BM25+ keyword statistics
model.rank(db, tokens, search.search_methods.MIXED)   # weighted RRF fusion of both
```

Any call accepts an optional SQL `WHERE` clause that intersects with the ranking, for faceted search — evaluated only over the top candidates, so it stays fast:

```python
model.rank(db, tokens, search.search_methods.MIXED,
           n_results=500,
           sql_query="WHERE lang = ? AND category = ?", sql_params=["fr", "forum"])

# full-text and PCRE-regex filters work too (custom SQLite extension)
model.rank(db, tokens, search.search_methods.MIXED,
           sql_query="WHERE parsed REGEXP ?", sql_params=[r"colour\s+science"])
```

The result is a list of `(index, url, score)` tuples, best first. Fetch display fields with a follow-up query — restricting to the searchable subset so a duplicated URL resolves to a single live row:

```python
urls = [url for _, url, _ in results[:20]]
ph = ",".join("?" * len(urls))
rows = db.execute(
    f"SELECT title, url, excerpt, datetime FROM pages "
    f"WHERE url IN ({ph}) AND in_index = 1", urls).fetchall()
```

---

## Step 5 — The web interface

A minimal Flask app exposes the `Indexer` over HTTP. It loads the engine once at startup and opens one read-only DB connection per request.

```python
# search_app.py
import os
os.environ["OPENBLAS_NUM_THREADS"] = "2"

from flask import Flask, request, jsonify, g
from core import database, search
from core.utils import typography_undo
import html

app = Flask(__name__)
DB_PATH, INDEX_NAME = "corpus.db", "engine"

def get_db():
    if "_db" not in g:
        g._db = database.open_db(DB_PATH, mode="ro")
    return g._db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop("_db", None)
    if db: db.close()

_engine = None
def get_engine():
    global _engine
    if _engine is None:
        db = database.open_db(DB_PATH, mode="ro")
        _engine = search.Indexer.load(INDEX_NAME, db)
        db.close()
    return _engine

@app.route("/api")
def api():
    raw = request.args.get("s", "").strip()
    if not raw:
        return jsonify({"error": "empty query"}), 400
    tokens = get_engine().tokenize_query(typography_undo(html.unescape(raw)))
    if not tokens:
        return jsonify({"error": "no recognisable keywords"}), 400

    clauses, params = ["in_index = 1"], []
    for field in ("lang", "category"):
        v = request.args.get(field, "any")
        if v != "any":
            clauses.append(f"{field} = ?"); params.append(v)
    sql = "WHERE " + " AND ".join(clauses)

    results = get_engine().rank(get_db(), tokens, search.search_methods.MIXED,
                                n_results=500, sql_query=sql, sql_params=params)

    urls = [u for _, u, _ in results[:20]]
    if not urls:
        return jsonify({"error": "no results"}), 200
    ph = ",".join("?" * len(urls))
    rows = {r[1]: r for r in get_db().execute(
        f"SELECT title, url, excerpt, datetime, category FROM pages "
        f"WHERE url IN ({ph}) AND in_index = 1", urls).fetchall()}
    items = [{"title": rows[u][0], "url": u, "excerpt": rows[u][2],
              "date": str(rows[u][3]), "category": rows[u][4]}
             for u in urls if u in rows]
    return jsonify({"query": raw, "n_results": len(results), "results": items})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

```bash
python search_app.py                        # dev
gunicorn -w 2 -b 0.0.0.0:5000 search_app:app  # production
```

A minimal front-end (`index.html`) that calls `/api` with `fetch()` and renders the JSON is all you need for a browser UI; wrap `get_engine().rank(...)` in `flask_caching`'s `@cache.memoize` for production.

!!! tip "Slimming the deploy copy"
    The full corpus carries `content`/`tokenized`/`stemmed`/`vectorized`, which the live app never reads at query time (the ranker and vectors live in the pickled Indexer). For deployment, project a slim copy — the `in_index = 1` rows, display columns only — with [`import_pages`][core.database.import_pages] into a fresh DB, then [`compress_db`][core.database.compress_db]`(delete_columns=["content","tokenized","stemmed","vectorized"])`. Carry the `search_rowid` column so the app serves it identically to the full corpus. This ships far smaller.

---

## Advanced topics

### Result ranking transparency

`MIXED` fuses semantic and BM25+ rankings with weighted Reciprocal Rank Fusion; the AI vote weight is tunable in `Indexer.rank()` to shift between thematic relevance and exact-term matching.

### Related keywords and "did you mean?"

```python
related = model.get_related(tokens, n=20, k=5)
# for ['exposur','histogram'] → ['highlight','shadow','tone_curv','clipping','waveform', …]
```

Useful for query expansion. To map a stem back to its most probable original token for display, build a reverse lookup with [`nlp.StemTokenIndex`][core.nlp.StemTokenIndex], which writes a `stem_tokens` table you can query at request time.

### Monitoring

The `stats` table the Indexer builds tracks word/page counts, domain distribution, and the most recent crawl date — surface those to show index freshness.

---

## What's next

- **RAG** — feed the top `rank()` results as context to a local LLM, grounded in your curated corpus.
- **Private intranet search** — the crawler handles any URL including `http://localhost:*`, so internal wikis, docs servers, and local PDF archives use exactly the same pipeline.
