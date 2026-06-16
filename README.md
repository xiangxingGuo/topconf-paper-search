# TopConf Paper Search

A minimal, GitHub-ready MVP for searching accepted papers from selected top conferences.

The first version intentionally avoids brittle automated crawling. You save official accepted-paper HTML pages manually, then this tool:

1. parses paper metadata into one normalized CSV;
2. builds FAISS semantic indices over `title`, `abstract`, and `title + abstract`;
3. provides keyword, semantic, and hybrid search;
4. filters by conference and year;
5. exports current results or filtered conference/year subsets as CSV or Excel;
6. runs locally or in Docker with a Streamlit UI.

## MVP scope

Supported now:

- Manual HTML ingestion from conference official/proceedings pages.
- Parser hints for `iclr_virtual`, `pmlr`, `openreview`, `cvf`, `acl`, and `generic` pages.
- Normalized CSV schema for all conferences.
- FAISS cosine-similarity search using SentenceTransformers embeddings.
- Keyword search, semantic search, and hybrid search.
- Field selection: title, abstract, or title + abstract.
- Export to `.csv` and `.xlsx`.
- Docker Compose deployment.

Not included yet:

- A browser agent / crawler.
- Automatic detail-page fetching for missing abstracts.
- Zotero export.
- User accounts or shared database backend.

## Repository layout

```text
.
├── app/streamlit_app.py             # Search UI
├── configs/conferences.yaml         # Conference parser notes
├── data/
│   ├── html/                        # Put manually saved HTML here
│   ├── processed/                   # Generated papers.csv
│   └── index/                       # Generated FAISS indices
├── examples/                        # Tiny parser examples
├── src/paper_search/
│   ├── parsers.py                   # HTML parsers
│   ├── ingest.py                    # HTML -> CSV CLI
│   ├── index.py                     # CSV -> FAISS CLI
│   ├── search.py                    # Search engine + CLI
│   └── export.py                    # CSV/Excel bytes
└── tests/test_parsers.py
```

## Data schema

All conferences are converted to this CSV format:

| column | meaning |
|---|---|
| `paper_id` | stable hash from conference/year/title/url |
| `conference` | e.g. `ICML`, `NeurIPS`, `CVPR` |
| `year` | conference year |
| `title` | paper title |
| `abstract` | abstract if available in saved HTML |
| `authors` | authors as a semicolon/comma string |
| `url` | paper/detail URL if available |
| `pdf_url` | PDF URL if available |
| `source_file` | local HTML file path |
| `parser` | parser family that extracted the row |

## Quick start with Docker

### 1. Save HTML

Put manually saved official pages under a path like:

```text
data/html/ICML/2024/icml2024.html
data/html/CVPR/2024/cvpr2024_all.html
data/html/ACL/2024/main_conference_papers.html
```

The folder convention `CONF/YEAR/*.html` lets the CLI infer conference and year. You can also pass them explicitly.

### 2. Build the image

```bash
docker compose build
```

### 3. Ingest saved HTML into CSV

Examples:

```bash
# ICML / PMLR-style page
docker compose run --rm paper-search \
  python -m paper_search.ingest \
  --input data/html/ICML/2024 \
  --conference ICML \
  --year 2024 \
  --parser pmlr \
  --base-url https://proceedings.mlr.press/v235/ \
  --output data/processed/papers.csv \
  --append


# ICLR virtual page, e.g., https://iclr.cc/virtual/2025/papers.html
docker compose run --rm paper-search \
  python -m paper_search.ingest \
  --input data/html/ICLR/2025 \
  --conference ICLR \
  --year 2025 \
  --parser iclr_virtual \
  --base-url https://iclr.cc/ \
  --output data/processed/papers.csv \
  --append

# CVPR / CVF-style page
docker compose run --rm paper-search \
  python -m paper_search.ingest \
  --input data/html/CVPR/2024 \
  --conference CVPR \
  --year 2024 \
  --parser cvf \
  --base-url https://openaccess.thecvf.com/ \
  --output data/processed/papers.csv \
  --append

# ACL-style accepted paper page
docker compose run --rm paper-search \
  python -m paper_search.ingest \
  --input data/html/ACL/2024 \
  --conference ACL \
  --year 2024 \
  --parser acl \
  --base-url https://2024.aclweb.org/ \
  --output data/processed/papers.csv \
  --append
```

For unknown pages, start with `--parser auto`; if extraction is noisy, retry with a specific parser or adapt `src/paper_search/parsers.py`.

### 4. Build semantic index

```bash
docker compose run --rm paper-search \
  python -m paper_search.index \
  --csv data/processed/papers.csv \
  --index-dir data/index \
  --model-name sentence-transformers/all-MiniLM-L6-v2
```

The first run downloads the embedding model into the Docker `hf-cache` volume.

### 5. Start UI

```bash
docker compose up
```

Open the Streamlit URL printed by Docker, usually `http://localhost:8501`.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

python -m paper_search.ingest --input examples --conference TEST --year 2026 --output data/processed/papers.csv
python -m paper_search.index --csv data/processed/papers.csv --index-dir data/index
streamlit run app/streamlit_app.py
```

Run parser tests:

```bash
pytest
```

## Command-line search

```bash
python -m paper_search.search "retrieval augmented generation" \
  --csv data/processed/papers.csv \
  --index-dir data/index \
  --mode hybrid \
  --field both \
  --conference ICML \
  --year 2024 \
  --top-k 20
```

Modes:

- `keyword`: fast exact-ish term matching.
- `semantic`: embedding search with FAISS.
- `hybrid`: weighted blend of semantic and keyword scores.

Fields:

- `title`
- `abstract`
- `both`

## Recommended MVP workflow

For each conference/year:

1. Save accepted-paper page HTML manually.
2. Run `paper_search.ingest` with the best parser hint.
3. Quickly inspect `data/processed/papers.csv` for obvious parser errors.
4. Run `paper_search.index` once after all CSV updates.
5. Search/export from Streamlit.

## Adding a new conference

1. Save one official accepted-paper HTML page into `examples/`.
2. Run:

   ```bash
   python -m paper_search.ingest --input examples/new_page.html --conference NEWCONF --year 2026 --parser auto --output /tmp/newconf.csv
   ```

3. If results are bad, add a targeted parser function in `src/paper_search/parsers.py`.
4. Add a regression test in `tests/test_parsers.py`.
5. Add a note to `configs/conferences.yaml`.

## Future roadmap

Good next features after this MVP:

- Detail-page enrichment: when list pages only include title/authors, fetch each paper detail page and fill abstract/PDF.
- Zotero/BibTeX/RIS export.
- Incremental indexing so adding one conference does not rebuild every vector.
- SQLite or DuckDB storage for larger multi-year corpora.
- Scheduled crawler or browser agent with per-conference adapters.
- Reranking with a cross-encoder for top 100 semantic results.
- Saved searches and lab-shared annotations.
