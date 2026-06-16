# TopConf Paper Search

A local-first academic search tool for accepted papers from top AI/ML/NLP/CV/system conferences.

This project is motivated by a simple problem: when doing literature review, we often do **not** want to search the entire web. We want to search accepted papers from a controlled set of venues such as **ICML, NeurIPS, ICLR, MLSys, CVPR, ICCV, ACL, EMNLP**, and later other venues such as INFOCOM, CHIL, etc.

## Why another academic search tool?

Existing tools are useful, but they do not fully match this workflow.

| Tool/source | Shortcoming for this use case |
|---|---|
| Google Scholar | **Good global coverage, but weak venue/year control**. It is hard to elegantly search only accepted papers from specific top conferences. Results may include arXiv, journals, workshop papers, citations, and unrelated versions. |
| Semantic Scholar | Has structured metadata and some venue filters, but **conference filtering is incomplete or limited for many venues/years**. It is not optimized for “search within this list of accepted conference papers.” |
| Paper Copilot | Useful for paper discovery and can search some conference pages, but **it still does not provide enough control over local data, custom conference sets, export, ranking experiments, or lab-specific workflows**. |
| Official conference websites | They usually publish accepted-paper lists, but the **websites are not designed as flexible search engines**. Search, export, semantic retrieval, and cross-conference comparison are limited. |

## What this project provides

TopConf Paper Search takes the accepted-paper lists from official conference pages and turns them into a local searchable dataset.

Main advantages:

- **Conference-focused**: search only the venues/years you care about.
- **Local and reproducible**: the source data is your saved HTML and generated CSV.
- **Simple normalized schema**: all conferences become one `papers.csv`.
- **Semantic search**: FAISS + SentenceTransformers/BGE embeddings.
- **Lexical search**: BM25 for exact technical terms.
- **Hybrid ranking**: Reciprocal Rank Fusion combines BM25 and semantic retrieval.
- **Export-friendly**: CSV/Excel export is built into the UI.
- **Easy to extend**: add new parsers, new conferences, better enrichers, or Zotero export later.

The current design intentionally avoids a full browser crawler. For now, you manually save official accepted-paper HTML pages, then parse/enrich/search them locally.

---

## Repository layout

```text
.
├── app/
│   └── streamlit_app.py             # Local search UI
├── configs/
│   └── conferences.yaml             # Conference parser notes
├── data/
│   ├── html/                        # Manually saved official HTML pages
│   ├── processed/                   # Generated papers.csv
│   ├── index/                       # Generated FAISS indices
│   └── cache/                       # Optional cache for fetched abstract pages
├── examples/                        # Tiny parser examples
├── src/paper_search/
│   ├── parsers.py                   # HTML parsers
│   ├── ingest.py                    # HTML -> CSV
│   ├── enrich_abstracts.py          # Visit paper URLs -> fill missing abstracts
│   ├── index.py                     # CSV -> FAISS indices
│   ├── bm25.py                      # BM25 lexical search
│   ├── fusion.py                    # RRF fusion utilities
│   ├── search.py                    # Search engine + CLI
│   └── export.py                    # CSV/Excel export helpers
└── tests/
    └── test_parsers.py
```

Recommended saved-HTML layout:

```text
data/html/
├── ACL/
│   ├── 2024/ACL 2024.html
│   └── 2025/ACL 2025.html
├── EMNLP/
│   ├── 2024/EMNLP 2024.html
│   └── 2025/EMNLP 2025.html
├── ICLR/
│   ├── 2024/ICLR 2024 Papers.html
│   ├── 2025/ICLR 2025 Papers.html
│   └── 2026/ICLR 2026 Papers.html
├── ICML/
│   ├── 2024/ICML 2024 Papers.html
│   └── 2025/ICML 2025 Papers.html
├── NeurIPS/
│   ├── 2024/NeurIPS 2024 Papers.html
│   └── 2025/NeurIPS 2025 Papers.html
├── CVPR/
├── ICCV/
└── MLSys/
```

The `CONF/YEAR/*.html` convention lets the parser infer conference and year automatically.

---

## Data schema

All conferences are converted to one normalized CSV.

| Column | Meaning |
|---|---|
| `paper_id` | Stable hash from conference/year/title/url |
| `conference` | Conference name, e.g., `ICML`, `NeurIPS`, `ICLR`, `ACL` |
| `year` | Conference year |
| `title` | Paper title |
| `abstract` | Paper abstract, if available from saved HTML or later enrichment |
| `authors` | Authors as one string |
| `url` | Paper/detail page URL |
| `pdf_url` | PDF URL, if available |
| `source_file` | Local HTML file path |
| `parser` | Parser family that extracted the row |

---

## Environment setup with `uv`

Python requirement: **Python 3.10+**.

Install `uv` if you do not already have it:

```bash
pip install uv
```
Or through the offical documents to install `uv` at https://docs.astral.sh/uv/getting-started/installation/.



The recommended setup is simply:

```bash
uv sync
```

`uv sync` creates the virtual environment if needed, resolves dependencies from `pyproject.toml`, installs this project in editable mode, and keeps the environment reproducible. You do not need to manually run `uv venv` or `uv pip install -e .` for normal use.

Core packages used by the project:

```text
beautifulsoup4
faiss-cpu
numpy
openpyxl
pandas
PyYAML
requests
sentence-transformers
streamlit
```

For GPU acceleration, install the PyTorch version that matches your CUDA environment separately. The default setup works on CPU.

## Quick start with the shared `papers.csv`
If you only want to **use the search tool** and do not want to parse official HTML pages yourself, start from a processed CSV shared by the project maintainer.

Expected file:

```text
/data/processed/papers.csv
```

This CSV already contains normalized paper metadata from the supported conferences, such as title, authors, URL, conference, year, and abstracts when available. With this file, you can skip the HTML parsing step and directly build the search index.

**You can jump to Step 3 if you already have `data/processed/papers.csv`**.

---

## Step 1: Parse saved HTML into `papers.csv`

If your saved HTML follows the recommended folder layout, you can parse all conferences at once:

```bash
python -m paper_search.ingest \
  --input data/html \
  --parser auto \
  --output data/processed/papers.csv
```

If you want to append one conference/year at a time:

```bash
python -m paper_search.ingest \
  --input data/html/ICLR/2025 \
  --conference ICLR \
  --year 2025 \
  --parser auto \
  --base-url https://iclr.cc/ \
  --output data/processed/papers.csv \
  --append
```

Useful parser names:

| Parser | Typical source |
|---|---|
| `auto` | Try to infer the best parser |
| `virtual` / `iclr_virtual` | ICLR/ICML/NeurIPS/MLSys virtual pages using paper cards/poster links |
| `acl` | ACL/EMNLP accepted-paper pages |
| `cvf` | CVPR/ICCV CVF openaccess pages |
| `pmlr` | PMLR proceedings pages |
| `openreview` | OpenReview-style pages |
| `generic` | Fallback parser |

Quickly inspect the parsed CSV:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv('data/processed/papers.csv')
print(df.shape)
print(df[['conference', 'year']].value_counts().head(20))
print(df[['title', 'url', 'abstract']].head())
PY
```

---

## Step 2: Enrich missing abstracts from paper URLs

Many accepted-paper list pages contain title and URL but not abstract. If `papers.csv` already has paper URLs, run the abstract enrichment command.

Test on a small sample first:

```bash
python -m paper_search.enrich_abstracts \
  --csv data/processed/papers.csv \
  --output data/processed/papers.enriched.sample.csv \
  --max-rows 30 \
  --min-wait 2.0 \
  --max-wait 6.0
```

If the sample looks good, update `papers.csv` directly:

```bash
python -m paper_search.enrich_abstracts \
  --csv data/processed/papers.csv \
  --inplace \
  --min-wait 3.0 \
  --max-wait 10.0 \
  --long-pause-every 80 \
  --long-pause-min 90 \
  --long-pause-max 240 \
  --save-every 25
```

What this does:

1. reads `data/processed/papers.csv`;
2. finds rows where `abstract` is empty;
3. visits the paper `url`;
4. extracts abstract text from metadata, abstract sections, OpenReview-like text, or embedded JSON;
5. writes the abstract back to the CSV;
6. saves progress periodically;
7. creates a backup before in-place updates.

Recommended conservative waiting behavior:

```text
--min-wait 3.0 --max-wait 10.0
--long-pause-every 80
--long-pause-min 90 --long-pause-max 240
```

This is intentionally slower but friendlier to official conference sites.

Check abstract coverage:

```bash
python - <<'PY'
import pandas as pd

df = pd.read_csv('data/processed/papers.csv')
abstract_len = df['abstract'].fillna('').astype(str).str.len()
print('Total papers:', len(df))
print('Papers with non-empty abstract:', (abstract_len > 0).sum())
print('Papers with meaningful abstract length > 50:', (abstract_len > 50).sum())
print(df.groupby(['conference', 'year']).apply(lambda x: (x['abstract'].fillna('').astype(str).str.len() > 50).sum()))
PY
```

---

## Step 3: Build FAISS semantic indices

You can choose a basic model or a stronger large model.

### Option A: basic model, faster and lighter

Good for quick tests and weaker machines:

```bash
python -m paper_search.index \
  --csv data/processed/papers.csv \
  --index-dir data/index_bge_base \
  --model-name BAAI/bge-base-en-v1.5 \
  --fields title abstract both
```

### Option B: large model, better quality but slower

Recommended for better matching quality:

```bash
python -m paper_search.index \
  --csv data/processed/papers.csv \
  --index-dir data/index_bge_large \
  --model-name BAAI/bge-large-en-v1.5 \
  --fields title abstract both
```

Notes:

- If abstracts are incomplete, `title` and `both` are still useful.
- After changing abstracts or changing the embedding model, rebuild the index.
- If the UI results look stale, restart Streamlit or clear its cache.

---

## Step 4: Start the local UI

Run:

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL printed in the terminal, usually:

```text
http://localhost:8501
```

Recommended UI settings:

```text
CSV path: data/processed/papers.csv
FAISS index directory: data/index_bge_large
Mode: Hybrid RRF
Search field: title + abstract
```

Try queries such as:

```text
speculative decoding
collaborative LLM inference
medical vision-language agent
domain generalization
retrieval augmented generation
multi-agent healthcare
```

---

## Command-line search

Hybrid RRF search:

```bash
python -m paper_search.search "collaborative LLM inference" \
  --csv data/processed/papers.csv \
  --index-dir data/index_bge_large \
  --mode hybrid_rrf \
  --field both \
  --top-k 30
```

Semantic-only search:

```bash
python -m paper_search.search "speculative decoding" \
  --csv data/processed/papers.csv \
  --index-dir data/index_bge_large \
  --mode semantic \
  --field both \
  --top-k 30
```

BM25-only search:

```bash
python -m paper_search.search "test time scaling" \
  --csv data/processed/papers.csv \
  --index-dir data/index_bge_large \
  --mode bm25 \
  --field both \
  --top-k 30
```

---

## Search modes

| Mode | What it does | Best for |
|---|---|---|
| `BM25` / `bm25` | Lexical ranking over paper text | Exact technical terms, acronyms, method names |
| `Semantic` / `semantic` | FAISS vector search using SentenceTransformers/BGE embeddings | Conceptual search, related work discovery, terminology variation |
| `Hybrid RRF` / `hybrid_rrf` | Retrieves from BM25 and FAISS, then combines rankings with Reciprocal Rank Fusion | Default mode; balanced search |
| `Regex keyword` / `keyword` | Simple keyword/regex-like matching | Quick debugging or exact term filtering |

Recommended default:

```text
Mode: Hybrid RRF
Field: title + abstract
Model: BAAI/bge-large-en-v1.5
```

---

## Understanding match scores

Different search modes produce different score types.

### `semantic_score`

This comes from FAISS similarity between the query embedding and paper embedding.

Higher usually means the paper is semantically closer to the query. The exact value depends on the embedding model and normalization, so it is best used for ranking within the same query/model, not as an absolute relevance probability.

### `bm25_score`

This comes from BM25 lexical matching.

Higher means stronger keyword overlap, especially for rare terms. BM25 is good at exact terms such as `speculative decoding`, `LoRA`, `RAG`, `diffusion`, or `vision-language model`.

### `semantic_rank` and `bm25_rank`

These show where the paper appeared in each individual retrieval list.

For example:

```text
semantic_rank = 3
bm25_rank = 42
```

means the paper was very strong semantically but weaker lexically.

### `rrf_score` / `score` in Hybrid RRF mode

Hybrid mode uses Reciprocal Rank Fusion:

```text
RRF score = 1 / (k + semantic_rank) + 1 / (k + bm25_rank)
```

A paper ranks well if it appears high in either semantic search or BM25 search. This avoids directly mixing raw FAISS and BM25 scores, which are not on the same scale.

Interpretation:

- high semantic rank + high BM25 rank = very strong result;
- high semantic rank only = conceptually related, even if words differ;
- high BM25 rank only = exact-term match, but may need manual judgment;
- low rank in both = likely weak result.

Do **not** interpret scores as probabilities. They are ranking signals.

---

## Recommended workflow

For each new conference/year:

1. Save the official accepted-paper HTML page manually.
2. Put it under `data/html/CONF/YEAR/`.
3. Run `paper_search.ingest` to update `papers.csv`.
4. Inspect parsed rows for obvious title/url errors.
5. Run `paper_search.enrich_abstracts` to fill missing abstracts.
6. Rebuild FAISS indices with `paper_search.index`.
7. Search in Streamlit using Hybrid RRF.
8. Export selected results to CSV/Excel.

---

## Adding a new conference

1. Save one official accepted-paper HTML page under `examples/`.
2. Try the automatic parser:

   ```bash
   python -m paper_search.ingest \
     --input examples/new_page.html \
     --conference NEWCONF \
     --year 2026 \
     --parser auto \
     --output /tmp/newconf.csv
   ```

3. If results are noisy, add a targeted parser in `src/paper_search/parsers.py`.
4. Add a small regression test in `tests/test_parsers.py`.
5. Add a note to `configs/conferences.yaml`.

---

## Future roadmap

Docker was part of the early prototype, but dependency installation can be slow because of PyTorch/SentenceTransformers. For now, local `uv` setup is the recommended workflow.

Planned future improvements:

- Docker image with prebuilt/cached ML dependencies.
- Better conference-specific abstract enrichment from official JSON files.
- Zotero/BibTeX/RIS export.
- Incremental indexing so adding one conference does not rebuild every vector.
- SQLite or DuckDB backend for larger corpora.
- Browser-agent crawler with per-conference adapters.
- Cross-encoder reranking for top retrieved candidates.
- Saved searches and lab-shared annotations.
- Deduplication across arXiv/OpenReview/proceedings versions.

---

## Current limitations

- Abstract coverage depends on what can be extracted from each paper URL.
- Some official pages load metadata dynamically from JavaScript/JSON.
- Ranking quality depends on abstract coverage and embedding model choice.
- The project is optimized for local research workflows, not a public multi-user search service.
