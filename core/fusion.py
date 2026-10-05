from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from core import config


@dataclass(slots=True, frozen=True)
class FusedHit:
    id: int
    score: float
    ranks: tuple[int | None, ...]


def rrf(rankings: Sequence[Sequence[int]], k: int = config.RRF_K, top_n: int | None = None) -> list[FusedHit]:
    ranks: dict[int, list[int | None]] = {}
    for m, ranking in enumerate(rankings):
        for pos, doc in enumerate(ranking, start=1):
            slot = ranks.setdefault(int(doc), [None] * len(rankings))
            if slot[m] is None:
                slot[m] = pos
    hits = [
        FusedHit(doc, sum(1.0 / (k + r) for r in rs if r is not None), tuple(rs))
        for doc, rs in ranks.items()
    ]
    hits.sort(key=lambda h: (-h.score, min(r for r in h.ranks if r is not None), h.id))
    return hits[:top_n] if top_n is not None else hits


def _self_check() -> None:
    print("formula")
    hits = rrf([[10, 20, 30], [20, 40]])
    by_id = {h.id: h for h in hits}
    assert abs(by_id[20].score - (1 / 62 + 1 / 61)) < 1e-15
    assert abs(by_id[10].score - 1 / 61) < 1e-15 and by_id[10].ranks == (1, None)
    assert [h.id for h in hits] == [20, 10, 40, 30]
    print(f"  ok  RRF(20) = 1/62 + 1/61 = {by_id[20].score:.6f}; missing lists contribute 0")

    print("consensus beats a single-retriever outlier")
    hits = rrf([[1, 2, 3, 4], [5, 2, 6, 7]])
    assert hits[0].id == 2, hits
    print(f"  ok  doc #2-in-both ({hits[0].score:.5f}) > doc #1-in-one ({hits[1].score:.5f})")

    print("k controls how much the top rank dominates")
    lo = rrf([[1, 2, 3, 4], [5, 2, 6, 7]], k=0)
    assert lo[0].id in (1, 5), "with k = 0 a single #1 (1/1) beats #2 twice (1/2 + 1/2 = 1, tie broken by best rank)"
    print(f"  ok  k=0 -> top {lo[0].id} (score {lo[0].score}); k=60 -> top {hits[0].id}")

    assert rrf([], top_n=5) == [] and len(rrf([[1, 2, 3]], top_n=2)) == 2
    print("core.fusion: all checks passed")


if __name__ == "__main__":
    _self_check()
