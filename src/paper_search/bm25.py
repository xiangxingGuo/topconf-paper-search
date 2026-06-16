from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]+")


def tokenize(text: str) -> list[str]:
    """Small, dependency-free tokenizer for academic paper titles/abstracts."""
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(str(text)) if len(t) >= 2]


@dataclass
class BM25Index:
    """A simple in-memory BM25 index.

    This avoids an extra dependency such as rank_bm25 and is fast enough for tens of
    thousands of conference papers. It is rebuilt in memory when the search engine
    loads, so there is no separate BM25 build step.
    """

    documents: list[str]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.doc_tokens: list[list[str]] = [tokenize(doc) for doc in self.documents]
        self.doc_lens = np.array([len(tokens) for tokens in self.doc_tokens], dtype="float32")
        self.avgdl = float(self.doc_lens.mean()) if len(self.doc_lens) else 0.0
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.doc_freq.update(set(tokens))
        self.n_docs = len(self.doc_tokens)
        self.term_freqs: list[Counter[str]] = [Counter(tokens) for tokens in self.doc_tokens]

    def idf(self, term: str) -> float:
        # Robertson/Sparck Jones IDF with a +1 shift to keep values positive.
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score(self, query: str | Iterable[str]) -> np.ndarray:
        if isinstance(query, str):
            query_terms = tokenize(query)
        else:
            query_terms = [str(t).lower() for t in query if str(t)]
        scores = np.zeros(self.n_docs, dtype="float32")
        if not query_terms or self.n_docs == 0:
            return scores

        unique_terms = list(dict.fromkeys(query_terms))
        avgdl = self.avgdl or 1.0
        for term in unique_terms:
            idf = self.idf(term)
            if idf <= 0:
                continue
            for i, tf_counter in enumerate(self.term_freqs):
                tf = tf_counter.get(term, 0)
                if tf == 0:
                    continue
                denom = tf + self.k1 * (1.0 - self.b + self.b * float(self.doc_lens[i]) / avgdl)
                scores[i] += float(idf * (tf * (self.k1 + 1.0)) / denom)
        return scores
