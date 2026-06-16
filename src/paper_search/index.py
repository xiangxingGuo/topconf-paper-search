from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .ingest import normalize_dataframe
from .schema import PAPER_COLUMNS
from .utils import clean_text


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_FIELDS = ["title", "abstract", "both"]


def make_index_texts(df: pd.DataFrame, field: str) -> list[str]:
    if field == "title":
        return [clean_text(x) for x in df["title"].fillna("").tolist()]
    if field == "abstract":
        texts = []
        for title, abstract in zip(df["title"].fillna(""), df["abstract"].fillna("")):
            # If an accepted list lacks abstract, keep it searchable by title rather than empty vector.
            texts.append(clean_text(abstract) or clean_text(title))
        return texts
    if field == "both":
        return [
            clean_text(f"title: {title}\nabstract: {abstract}")
            for title, abstract in zip(df["title"].fillna(""), df["abstract"].fillna(""))
        ]
    raise ValueError(f"Unknown index field: {field}")


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
        texts = make_index_texts(df, field)
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
