from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def reciprocal_rank(ranked: Sequence[int], qrels: Mapping[int, int], k: int = 10) -> float:
    for i, doc in enumerate(ranked[:k], start=1):
        if qrels.get(doc, 0) > 0:
            return 1.0 / i
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum((2.0 ** rel - 1.0) / math.log2(i + 1) for i, rel in enumerate(gains, start=1))


def dcg_at_k(ranked: Sequence[int], qrels: Mapping[int, int], k: int = 10) -> float:
    return dcg([qrels.get(doc, 0) for doc in ranked[:k]])


def ndcg_at_k(ranked: Sequence[int], qrels: Mapping[int, int], k: int = 10) -> float:
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    return dcg_at_k(ranked, qrels, k) / ideal if ideal > 0 else 0.0


def mean_over_queries(runs: Sequence[tuple[Sequence[int], Mapping[int, int]]], k: int = 10) -> dict[str, float]:
    if not runs:
        return {"mrr": 0.0, "ndcg": 0.0}
    n = len(runs)
    return {
        "mrr": sum(reciprocal_rank(r, q, k) for r, q in runs) / n,
        "ndcg": sum(ndcg_at_k(r, q, k) for r, q in runs) / n,
    }


def _self_check() -> None:
    print("MRR")
    runs = [([7, 1, 2], {7: 1}), ([1, 7, 2], {7: 1}), ([1, 2, 3, 7], {7: 1}), ([1, 2, 3], {7: 1})]
    m = mean_over_queries(runs)["mrr"]
    assert abs(m - (1 + 1 / 2 + 1 / 4 + 0) / 4) < 1e-12
    assert reciprocal_rank([1, 2, 3, 7], {7: 1}, k=3) == 0.0
    print(f"  ok  ranks 1, 2, 4, miss -> MRR = {m:.4f}; cutoff k respected")

    print("DCG / NDCG")
    qrels = {1: 3, 2: 2, 3: 0}
    assert abs(dcg_at_k([1, 2, 3], qrels) - (7 + 3 / math.log2(3))) < 1e-12
    assert ndcg_at_k([1, 2, 3], qrels) == 1.0
    swapped = ndcg_at_k([2, 1, 3], qrels)
    assert abs(swapped - (3 + 7 / math.log2(3)) / (7 + 3 / math.log2(3))) < 1e-12
    assert ndcg_at_k([9, 8], qrels) == 0.0 and ndcg_at_k([1], {}) == 0.0
    print(f"  ok  ideal order = 1.0, swapped top-2 = {swapped:.4f}, no relevant = 0")
    print("core.metrics: all checks passed")


if __name__ == "__main__":
    _self_check()
