from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import normalize_dataframe
from .schema import PAPER_COLUMNS
from .utils import clean_text


DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"
INDEX_FIELDS = ["title", "abstract", "both"]


def make_index_texts(df: pd.DataFrame, field: str) -> list[str]:
    """Create the passage/document text for each index field.

    Many rows may still miss abstracts, so every field falls back to title instead
    of producing empty vectors. The text is lightly labeled to help embedding
    models distinguish title/abstract content.
    """
    titles = df["title"].fillna("").astype(str).tolist()
    abstracts = df["abstract"].fillna("").astype(str).tolist()
    conferences = df["conference"].fillna("").astype(str).tolist() if "conference" in df.columns else [""] * len(df)
    years = df["year"].fillna("").astype(str).tolist() if "year" in df.columns else [""] * len(df)

    texts: list[str] = []
    for title, abstract, conference, year in zip(titles, abstracts, conferences, years):
        title = clean_text(title)
        abstract = clean_text(abstract)
        conf_year = clean_text(f"{conference} {year}")
        if field == "title":
            text = f"title: {title}"
        elif field == "abstract":
            text = f"abstract: {abstract}" if abstract else f"title: {title}"
        elif field == "both":
            parts = [f"title: {title}"]
            if abstract:
                parts.append(f"abstract: {abstract}")
            if conf_year:
                parts.append(f"venue: {conf_year}")
            text = "\n".join(parts)
        else:
            raise ValueError(f"Unknown index field: {field}")
        texts.append(clean_text(text))
    return texts


def format_documents_for_model(texts: list[str], model_name: str) -> list[str]:
    """Apply model-specific passage prefixes when useful.

    BGE uses a query prefix only, so documents are unchanged. E5 models benefit
    from a `passage:` prefix for corpus/document embeddings.
    """
    name = (model_name or "").lower()
    if "e5" in name:
        return [f"passage: {text}" for text in texts]
    return texts


def build_faiss_index(embeddings: np.ndarray):
    import faiss  # type: ignore

    vectors = np.asarray(embeddings, dtype="float32")
    if vectors.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def build_indexes(
    csv_path: str | Path,
    index_dir: str | Path,
    *,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 64,
    fields: list[str] | None = None,
) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import faiss  # type: ignore

    csv_path = Path(csv_path)
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_dataframe(pd.read_csv(csv_path))
    if df.empty:
        raise ValueError(f"No paper rows found in {csv_path}")

    metadata = df[PAPER_COLUMNS].copy()
    metadata.to_csv(index_dir / "metadata.csv", index=False)

    model = SentenceTransformer(model_name)
    fields = fields or INDEX_FIELDS
    stats: dict[str, object] = {
        "csv_path": str(csv_path),
        "model_name": model_name,
        "num_papers": int(len(df)),
        "fields": fields,
    }

    for field in fields:
        texts = format_documents_for_model(make_index_texts(df, field), model_name)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        index = build_faiss_index(embeddings)
        faiss.write_index(index, str(index_dir / f"{field}.faiss"))
        np.save(index_dir / f"{field}.npy", embeddings)
        stats[f"{field}_dim"] = int(embeddings.shape[1])

    (index_dir / "index_config.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build FAISS indices over paper title / abstract / both.")
    parser.add_argument("--csv", default="data/processed/papers.csv", help="Input normalized papers CSV")
    parser.add_argument("--index-dir", default="data/index", help="Directory where FAISS index files will be stored")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="SentenceTransformers model name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--fields",
        nargs="+",
        default=INDEX_FIELDS,
        choices=INDEX_FIELDS,
        help="Which text fields to index",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    stats = build_indexes(
        args.csv,
        args.index_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        fields=args.fields,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
