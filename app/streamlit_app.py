from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Works both in Docker (/app/src) and local repo execution.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper_search.export import dataframe_to_csv_bytes, dataframe_to_excel_bytes
from paper_search.index import INDEX_FIELDS, build_indexes
from paper_search.search import PaperSearchEngine


DEFAULT_CSV = str(ROOT / "data" / "processed" / "papers.csv")
DEFAULT_INDEX = str(ROOT / "data" / "index")


st.set_page_config(page_title="TopConf Paper Search", layout="wide")
st.title("TopConf Paper Search")
st.caption("Conference-focused keyword, semantic, and hybrid search over accepted papers.")


@st.cache_resource(show_spinner=False)
def load_engine(csv_path: str, index_dir: str) -> PaperSearchEngine:
    return PaperSearchEngine(csv_path=csv_path, index_dir=index_dir)


@st.cache_data(show_spinner=False)
def cached_csv_bytes(df: pd.DataFrame) -> bytes:
    return dataframe_to_csv_bytes(df)


@st.cache_data(show_spinner=False)
def cached_excel_bytes(df: pd.DataFrame) -> bytes:
    return dataframe_to_excel_bytes(df)


with st.sidebar:
    st.header("Data")
    csv_path = st.text_input("Papers CSV", DEFAULT_CSV)
    index_dir = st.text_input("FAISS index directory", DEFAULT_INDEX)

    engine_error = None
    try:
        engine = load_engine(csv_path, index_dir)
    except Exception as exc:  # noqa: BLE001 - Streamlit should show actionable errors.
        engine = None
        engine_error = exc

    if st.button("Clear cache / reload data"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.header("Search")
    query = st.text_input("Query", placeholder="e.g., retrieval augmented generation, model editing, efficient inference")
    mode = st.radio("Mode", ["hybrid", "semantic", "keyword"], horizontal=True)
    field_label = st.selectbox("Search field", ["title + abstract", "title", "abstract"])
    field = {"title + abstract": "both", "title": "title", "abstract": "abstract"}[field_label]
    top_k = st.slider("Top K", 5, 200, 30, step=5)
    alpha = st.slider("Hybrid semantic weight", 0.0, 1.0, 1.0, step=0.05, disabled=(mode != "hybrid"))
    regex = st.checkbox("Use regex for keyword component", value=False)

    st.divider()
    st.header("Filters")
    if engine is not None and not engine.df.empty:
        selected_confs = st.multiselect("Conference", engine.conferences, default=[])
        selected_years = st.multiselect("Year", [str(y) for y in engine.years], default=[])
    else:
        selected_confs = []
        selected_years = []

    st.divider()
    with st.expander("Build / refresh semantic index"):
        st.write("Use after ingesting a new CSV. First run may download the embedding model.")
        model_name = st.text_input("Embedding model", "sentence-transformers/all-MiniLM-L6-v2")
        build_fields = st.multiselect("Index fields", INDEX_FIELDS, default=INDEX_FIELDS)
        if st.button("Build index"):
            try:
                stats = build_indexes(csv_path, index_dir, model_name=model_name, fields=build_fields)
                st.success(f"Built index for {stats['num_papers']} papers.")
                st.cache_resource.clear()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build index: {exc}")


if engine_error is not None:
    st.error(f"Could not load data: {engine_error}")
    st.stop()

if engine is None or engine.df.empty:
    st.warning("No papers loaded yet. Ingest saved HTML into data/processed/papers.csv first.")
    st.code(
        "python -m paper_search.ingest --input data/html/ICML/2024 --conference ICML --year 2024 --output data/processed/papers.csv --append",
        language="bash",
    )
    st.stop()

filtered_df = engine.filter_dataframe(conferences=selected_confs, years=selected_years)
metric_cols = st.columns(4)
metric_cols[0].metric("Total papers", f"{len(engine.df):,}")
metric_cols[1].metric("After filters", f"{len(filtered_df):,}")
metric_cols[2].metric("Conferences", f"{len(engine.conferences):,}")
metric_cols[3].metric("Years", f"{len(engine.years):,}")

try:
    if query.strip():
        if mode == "keyword":
            results = engine.keyword_search(
                query,
                field=field,
                top_k=top_k,
                conferences=selected_confs,
                years=selected_years,
                regex=regex,
            )
        elif mode == "semantic":
            results = engine.semantic_search(
                query,
                field=field,
                top_k=top_k,
                conferences=selected_confs,
                years=selected_years,
            )
        else:
            results = engine.hybrid_search(
                query,
                field=field,
                top_k=top_k,
                conferences=selected_confs,
                years=selected_years,
                alpha=alpha,
                regex=regex,
            )
    else:
        results = filtered_df.head(top_k).copy()
        results["score"] = 0.0
        results["match_type"] = "browse"
except Exception as exc:  # noqa: BLE001
    st.error(f"Search failed: {exc}")
    if mode in {"semantic", "hybrid"}:
        st.info("Build the semantic index first, or switch to keyword mode.")
    st.stop()

st.subheader(f"Results ({len(results):,})")

export_cols = st.columns(4)
export_cols[0].download_button(
    "Download results CSV",
    data=cached_csv_bytes(results),
    file_name="paper_search_results.csv",
    mime="text/csv",
)
export_cols[1].download_button(
    "Download results Excel",
    data=cached_excel_bytes(results),
    file_name="paper_search_results.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)
export_cols[2].download_button(
    "Download filtered CSV",
    data=cached_csv_bytes(filtered_df),
    file_name="filtered_papers.csv",
    mime="text/csv",
)
export_cols[3].download_button(
    "Download filtered Excel",
    data=cached_excel_bytes(filtered_df),
    file_name="filtered_papers.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

show_columns = [
    "score",
    "match_type",
    "conference",
    "year",
    "title",
    "authors",
    "url",
    "pdf_url",
    "abstract",
]
existing = [c for c in show_columns if c in results.columns]
st.dataframe(
    results[existing],
    width="stretch",
    hide_index=True
)

st.divider()
st.subheader("Paper cards")
for _, row in results.iterrows():
    title = str(row.get("title", "Untitled"))
    url = str(row.get("url", ""))
    pdf_url = str(row.get("pdf_url", ""))
    score = row.get("score", "")
    match_type = row.get("match_type", "")
    conference = row.get("conference", "")
    year = row.get("year", "")
    authors = str(row.get("authors", ""))
    abstract = str(row.get("abstract", ""))

    with st.container(border=True):
        if url:
            st.markdown(f"### [{title}]({url})")
        else:
            st.markdown(f"### {title}")
        st.caption(f"{conference} {year} · {match_type} · score={float(score):.4f}" if score != "" else f"{conference} {year}")
        if authors:
            st.write(authors)
        if pdf_url:
            st.markdown(f"[PDF]({pdf_url})")
        if abstract:
            with st.expander("Abstract"):
                st.write(abstract)
