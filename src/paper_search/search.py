from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .index import DEFAULT_MODEL, INDEX_FIELDS
from .ingest import normalize_dataframe
from .utils import clean_text


FIELD_TO_COLUMNS = {
    "title": ["title"],
    "abstract": ["abstract"],
    "both": ["title", "abstract"],
}


def needs_query_prefix(model_name: str) -> bool:
    name = (model_name or "").lower()
    return "bge" in name or "e5" in name


def format_query_for_embedding(query: str, model_name: str) -> str:
    query = clean_text(query)
    name = (model_name or "").lower()
    if "e5" in name:
        return f"query: {query}"
    if "bge" in name:
        return f"Represent this sentence for searching relevant passages: {query}"
    return query


class PaperSearchEngine:
    def __init__(
        self,
        csv_path: str | Path = "data/processed/papers.csv",
        index_dir: str | Path = "data/index",
        *,
        model_name: str | None = None,
        lazy_model: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.index_dir = Path(index_dir)
        self.df = normalize_dataframe(pd.read_csv(self.csv_path)) if self.csv_path.exists() else pd.DataFrame()
        self.config = self._load_config()
        self.model_name = model_name or str(self.config.get("model_name", DEFAULT_MODEL))
        self._model = None
        self._faiss_indices: dict[str, object] = {}
        self._bm25_indices: dict[str, BM25Index] = {}
        if not lazy_model:
            self._load_model()

    def _load_config(self) -> dict:
        path = self.index_dir / "index_config.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {}

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load_faiss_index(self, field: str):
        if field not in INDEX_FIELDS:
            raise ValueError(f"Unknown semantic field: {field}")
        if field not in self._faiss_indices:
            import faiss  # type: ignore

            path = self.index_dir / f"{field}.faiss"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Build the index first with: python -m paper_search.index"
                )
            self._faiss_indices[field] = faiss.read_index(str(path))
        return self._faiss_indices[field]

    def _load_bm25_index(self, field: str) -> BM25Index:
        if field not in self._bm25_indices:
            columns = FIELD_TO_COLUMNS.get(field, FIELD_TO_COLUMNS["both"])
            texts = join_fields(self.df, columns)
            self._bm25_indices[field] = BM25Index(texts)
        return self._bm25_indices[field]

    @property
    def conferences(self) -> list[str]:
        if self.df.empty or "conference" not in self.df.columns:
            return []
        return sorted(x for x in self.df["conference"].dropna().astype(str).unique() if x)

    @property
    def years(self) -> list[int | str]:
        if self.df.empty or "year" not in self.df.columns:
            return []
        values = self.df["year"].dropna().unique().tolist()
        return sorted(values, key=lambda x: str(x), reverse=True)

    def filter_dataframe(
        self,
        *,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
    ) -> pd.DataFrame:
        df = self.df.copy()
        conferences = [str(x).upper() for x in (conferences or []) if str(x)]
        years = [str(x) for x in (years or []) if str(x)]
        if conferences:
            df = df[df["conference"].astype(str).str.upper().isin(conferences)]
        if years:
            df = df[df["year"].astype(str).isin(years)]
        return df

    def _allowed_positions(
        self,
        *,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
    ) -> list[int]:
        return [int(i) for i in self.filter_dataframe(conferences=conferences, years=years).index]

    def bm25_search(
        self,
        query: str,
        *,
        field: str = "both",
        top_k: int = 50,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
    ) -> pd.DataFrame:
        if self.df.empty:
            return with_search_columns(self.df)
        query = clean_text(query)
        filtered = self.filter_dataframe(conferences=conferences, years=years)
        if not query:
            return with_search_columns(filtered.head(top_k).copy(), score=0.0, mode="browse")
        if filtered.empty:
            return with_search_columns(filtered)

        bm25 = self._load_bm25_index(field)
        scores = bm25.score(query)
        allowed_positions = self._allowed_positions(conferences=conferences, years=years)
        if not allowed_positions:
            return with_search_columns(self.df.iloc[[]].copy())

        candidate_scores = [(pos, float(scores[pos])) for pos in allowed_positions if float(scores[pos]) > 0]
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        candidate_scores = candidate_scores[:top_k]
        if not candidate_scores:
            return with_search_columns(self.df.iloc[[]].copy())

        rows = []
        for pos, score in candidate_scores:
            row = self.df.iloc[pos].copy()
            row["score"] = score
            row["match_type"] = "bm25"
            rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)

    def keyword_search(
        self,
        query: str,
        *,
        field: str = "both",
        top_k: int = 50,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
        regex: bool = False,
    ) -> pd.DataFrame:
        # Keep the old exact/regex keyword search for debugging and comparison.
        if not regex:
            return self.bm25_search(
                query,
                field=field,
                top_k=top_k,
                conferences=conferences,
                years=years,
            )

        df = self.filter_dataframe(conferences=conferences, years=years)
        if df.empty:
            return with_search_columns(df)
        query = clean_text(query)
        if not query:
            return with_search_columns(df.head(top_k).copy(), score=0.0, mode="browse")

        columns = FIELD_TO_COLUMNS.get(field, FIELD_TO_COLUMNS["both"])
        texts = join_fields(df, columns)
        scores = keyword_scores(texts, query, regex=True)
        out = df.copy()
        out["score"] = scores
        out["match_type"] = "regex"
        out = out[out["score"] > 0].sort_values("score", ascending=False).head(top_k)
        return out.reset_index(drop=True)

    def semantic_search(
        self,
        query: str,
        *,
        field: str = "both",
        top_k: int = 50,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
        fetch_k: int | None = None,
    ) -> pd.DataFrame:
        if self.df.empty:
            return with_search_columns(self.df)
        query = clean_text(query)
        filtered = self.filter_dataframe(conferences=conferences, years=years)
        if not query:
            return with_search_columns(filtered.head(top_k), score=0.0, mode="browse")
        if filtered.empty:
            return with_search_columns(filtered)

        model = self._load_model()
        index = self._load_faiss_index(field)
        embed_query = format_query_for_embedding(query, self.model_name)
        query_vec = model.encode([embed_query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

        n = len(self.df)
        fetch_k = fetch_k or min(n, max(top_k * 10, 200))
        fetch_k = min(fetch_k, n)
        scores, positions = index.search(query_vec, fetch_k)
        candidates = []
        allowed_set = set(self._allowed_positions(conferences=conferences, years=years))
        for score, pos in zip(scores[0], positions[0]):
            pos = int(pos)
            if pos < 0 or pos >= n or pos not in allowed_set:
                continue
            row = self.df.iloc[pos].copy()
            row["score"] = float(score)
            row["match_type"] = "semantic"
            candidates.append(row)
            if len(candidates) >= top_k:
                break
        if not candidates:
            return with_search_columns(self.df.iloc[[]].copy())
        return pd.DataFrame(candidates).reset_index(drop=True)

    def hybrid_search(
        self,
        query: str,
        *,
        field: str = "both",
        top_k: int = 50,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
        alpha: float = 0.65,
        regex: bool = False,
    ) -> pd.DataFrame:
        # Backward compatible alias. Use RRF now; alpha/regex are ignored except for
        # the semantic-vs-BM25 weight approximation below.
        semantic_weight = float(alpha)
        bm25_weight = 1.0 - semantic_weight
        return self.hybrid_rrf_search(
            query,
            field=field,
            top_k=top_k,
            conferences=conferences,
            years=years,
            semantic_weight=max(semantic_weight, 0.0),
            bm25_weight=max(bm25_weight, 0.0),
        )

    def hybrid_rrf_search(
        self,
        query: str,
        *,
        field: str = "both",
        top_k: int = 50,
        conferences: Iterable[str] | None = None,
        years: Iterable[int | str] | None = None,
        candidate_k: int | None = None,
        rrf_k: int = 60,
        semantic_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> pd.DataFrame:
        if self.df.empty:
            return with_search_columns(self.df)
        query = clean_text(query)
        filtered = self.filter_dataframe(conferences=conferences, years=years)
        if not query:
            return with_search_columns(filtered.head(top_k).copy(), score=0.0, mode="browse")
        if filtered.empty:
            return with_search_columns(filtered)

        candidate_k = candidate_k or max(top_k * 10, 200)
        semantic = self.semantic_search(
            query,
            field=field,
            top_k=candidate_k,
            conferences=conferences,
            years=years,
            fetch_k=max(candidate_k * 3, 500),
        )
        bm25 = self.bm25_search(
            query,
            field=field,
            top_k=candidate_k,
            conferences=conferences,
            years=years,
        )
        if semantic.empty and bm25.empty:
            return with_search_columns(self.df.iloc[[]].copy())

        sem_ids = semantic["paper_id"].astype(str).tolist() if not semantic.empty else []
        bm25_ids = bm25["paper_id"].astype(str).tolist() if not bm25.empty else []
        fused = reciprocal_rank_fusion(
            [sem_ids, bm25_ids],
            k=rrf_k,
            weights=[semantic_weight, bm25_weight],
        )
        if not fused:
            return with_search_columns(self.df.iloc[[]].copy())

        # Use first occurrence of each paper ID from the full dataframe for stable metadata.
        by_id = self.df.copy()
        by_id["paper_id"] = by_id["paper_id"].astype(str)
        by_id = by_id.drop_duplicates(subset=["paper_id"]).set_index("paper_id", drop=False)

        sem_rank = {paper_id: i + 1 for i, paper_id in enumerate(sem_ids)}
        bm25_rank = {paper_id: i + 1 for i, paper_id in enumerate(bm25_ids)}
        rows = []
        for paper_id, score in sorted(fused.items(), key=lambda item: item[1], reverse=True):
            if paper_id not in by_id.index:
                continue
            row = by_id.loc[paper_id].copy()
            row["score"] = float(score)
            row["match_type"] = "hybrid_rrf"
            row["semantic_rank"] = sem_rank.get(paper_id, np.nan)
            row["bm25_rank"] = bm25_rank.get(paper_id, np.nan)
            rows.append(row)
            if len(rows) >= top_k:
                break
        if not rows:
            return with_search_columns(self.df.iloc[[]].copy())
        return pd.DataFrame(rows).reset_index(drop=True)


def with_search_columns(df: pd.DataFrame, *, score: float = 0.0, mode: str = "") -> pd.DataFrame:
    out = df.copy()
    if "score" not in out.columns:
        out["score"] = score
    if "match_type" not in out.columns:
        out["match_type"] = mode
    return out


def join_fields(df: pd.DataFrame, columns: list[str]) -> list[str]:
    parts = []
    for _, row in df.iterrows():
        values = [clean_text(row.get(column, "")) for column in columns]
        parts.append("\n".join(v for v in values if v))
    return parts


def keyword_scores(texts: list[str], query: str, *, regex: bool = False) -> np.ndarray:
    scores = np.zeros(len(texts), dtype="float32")
    if regex:
        try:
            pattern = re.compile(query, flags=re.I)
        except re.error:
            return scores
        for i, text in enumerate(texts):
            matches = pattern.findall(text or "")
            scores[i] = float(len(matches))
        return scores

    query_lower = query.lower()
    tokens = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]+", query_lower) if len(t) >= 2]
    if not tokens:
        return scores

    for i, text in enumerate(texts):
        text_lower = (text or "").lower()
        score = 0.0
        if query_lower in text_lower:
            score += 3.0
        for token in tokens:
            count = len(re.findall(rf"\b{re.escape(token)}\b", text_lower))
            if count:
                score += 1.0 + math.log(count)
        if tokens and all(re.search(rf"\b{re.escape(token)}\b", text_lower) for token in tokens):
            score += 2.0
        scores[i] = float(score)
    return scores


def normalize_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return series
    max_value = float(series.max())
    min_value = float(series.min())
    if max_value == min_value:
        return pd.Series(1.0, index=series.index)
    return (series - min_value) / (max_value - min_value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search conference papers from the command line.")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--csv", default="data/processed/papers.csv")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--mode", choices=["bm25", "keyword", "semantic", "hybrid", "hybrid_rrf"], default="hybrid_rrf")
    parser.add_argument("--field", choices=INDEX_FIELDS, default="both")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--conference", action="append", default=[])
    parser.add_argument("--year", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=0.5, help="Legacy hybrid semantic weight")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--candidate-k", type=int, default=200)
    parser.add_argument("--regex", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    engine = PaperSearchEngine(args.csv, args.index_dir)
    if args.mode == "bm25":
        results = engine.bm25_search(
            args.query,
            field=args.field,
            top_k=args.top_k,
            conferences=args.conference,
            years=args.year,
        )
    elif args.mode == "keyword":
        results = engine.keyword_search(
            args.query,
            field=args.field,
            top_k=args.top_k,
            conferences=args.conference,
            years=args.year,
            regex=args.regex,
        )
    elif args.mode == "semantic":
        results = engine.semantic_search(
            args.query,
            field=args.field,
            top_k=args.top_k,
            conferences=args.conference,
            years=args.year,
        )
    elif args.mode == "hybrid":
        results = engine.hybrid_search(
            args.query,
            field=args.field,
            top_k=args.top_k,
            conferences=args.conference,
            years=args.year,
            alpha=args.alpha,
            regex=args.regex,
        )
    else:
        results = engine.hybrid_rrf_search(
            args.query,
            field=args.field,
            top_k=args.top_k,
            conferences=args.conference,
            years=args.year,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
        )
    columns = ["score", "match_type", "semantic_rank", "bm25_rank", "conference", "year", "title", "authors", "url"]
    print(results[[c for c in columns if c in results.columns]].to_string(index=False, max_colwidth=90))


if __name__ == "__main__":
    main()
