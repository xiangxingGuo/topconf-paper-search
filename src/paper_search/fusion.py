from __future__ import annotations

from collections.abc import Hashable


def reciprocal_rank_fusion(
    ranked_lists: list[list[Hashable]],
    *,
    k: int = 60,
    weights: list[float] | None = None,
) -> dict[Hashable, float]:
    """Fuse multiple ranked lists by Reciprocal Rank Fusion (RRF).

    RRF uses ranks instead of raw scores, so it is safer than linearly mixing BM25
    scores with embedding cosine/IP scores.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    fused: dict[Hashable, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranked, start=1):
            if item in seen:
                continue
            seen.add(item)
            fused[item] = fused.get(item, 0.0) + float(weight) / float(k + rank)
    return fused
