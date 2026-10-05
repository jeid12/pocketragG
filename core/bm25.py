from __future__ import annotations

import math
import os
import pickle
import re
import sys
from array import array
from collections import Counter
from pathlib import Path

import numpy as np

from core import config
from core.dense import top_k

_TOKEN = re.compile(r"\w+")
STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i if in into is it its of on or "
    "our she so that the their them then there these they this to was we were what when which "
    "who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in (m.group().lower() for m in _TOKEN.finditer(text)) if t not in STOPWORDS]


class BM25Index:
    def __init__(self, k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.k1 = k1
        self.b = b
        self._postings: dict[str, tuple[array, array]] = {}
        self._doc_len = array("I")
        self._total_len = 0

    def add(self, text: str) -> int:
        doc_id = len(self._doc_len)
        counts = Counter(tokenize(text))
        for term, tf in counts.items():
            ids, tfs = self._postings.setdefault(term, (array("I"), array("I")))
            ids.append(doc_id)
            tfs.append(tf)
        length = sum(counts.values())
        self._doc_len.append(length)
        self._total_len += length
        return doc_id

    @property
    def n_docs(self) -> int:
        return len(self._doc_len)

    @property
    def vocab_size(self) -> int:
        return len(self._postings)

    @property
    def avgdl(self) -> float:
        return self._total_len / self.n_docs if self.n_docs else 0.0

    def nbytes(self) -> int:
        total = sys.getsizeof(self._postings) + sys.getsizeof(self._doc_len)
        for term, (ids, tfs) in self._postings.items():
            total += sys.getsizeof(term) + sys.getsizeof(ids) + sys.getsizeof(tfs) + 56
        return total

    def df(self, term: str) -> int:
        p = self._postings.get(term)
        return len(p[0]) if p else 0

    def idf(self, term: str) -> float:
        n = self.df(term)
        return math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        if not self.n_docs:
            return out
        dl = np.frombuffer(self._doc_len, dtype=np.uint32)
        avgdl = self.avgdl or 1.0
        for term in dict.fromkeys(tokenize(query)):
            p = self._postings.get(term)
            if p is None:
                continue
            ids = np.frombuffer(p[0], dtype=np.uint32)
            tf = np.frombuffer(p[1], dtype=np.uint32).astype(np.float64)
            norm = self.k1 * (1.0 - self.b + self.b * dl[ids] / avgdl)
            out[ids] += self.idf(term) * tf * (self.k1 + 1.0) / (tf + norm)
        return out

    def search(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        ids = top_k(s, k)
        ids = ids[s[ids] > 0]
        return ids, s[ids]

    def save(self, path: str | os.PathLike) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | os.PathLike) -> BM25Index:
        with open(path, "rb") as f:
            return pickle.load(f)


def _self_check() -> None:
    import tempfile
    import time

    print("hand-computed example")
    idx = BM25Index()
    for text in ["cat cat dog", "dog bird", "fish"]:
        idx.add(text)
    assert idx.n_docs == 3 and idx.avgdl == 2.0 and idx.vocab_size == 4
    idf_cat = math.log((3 - 1 + 0.5) / (1 + 0.5) + 1)
    expected_d0 = idf_cat * (2 * 2.5) / (2 + 1.5 * (0.25 + 0.75 * 3 / 2))
    idf_dog = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
    expected_dog = [idf_dog * 2.5 / (1 + 1.5 * (0.25 + 0.75 * L / 2)) for L in (3, 2)]
    got = idx.scores("the Cat")
    assert abs(got[0] - expected_d0) < 1e-9 and got[1] == got[2] == 0, got
    got = idx.scores("dog")
    assert np.allclose(got[:2], expected_dog, atol=1e-9) and got[2] == 0
    assert idx.scores("cat cat")[0] == idx.scores("cat")[0]
    print(f"  ok  score(d0, 'cat') = {expected_d0:.9f}; shorter doc wins on equal tf for 'dog'; "
          f"stopwords + duplicate terms ignored")

    print("exact identifier lookup")
    corpus = [
        "The build system produces binaries for several processor families.",
        "Use the cross compiler to target embedded ARM boards and phones.",
        "Release wheels are published for x86_64 and aarch64 Linux.",
        "Processor architecture choices affect binary size and speed.",
    ]
    idx = BM25Index()
    for t in corpus:
        idx.add(t)
    ids, scores = idx.search("which wheels exist for x86_64?", k=3)
    assert ids[0] == 2, (ids, scores)
    assert len(idx.search("quantum chromodynamics", k=3)[0]) == 0
    print(f"  ok  'x86_64' -> chunk {ids[0]} (score {scores[0]:.3f}); unmatched query returns nothing")

    print("scale: 50k synthetic chunks x 150 tokens")
    rng = np.random.default_rng(0)
    vocab = np.array([f"t{i}" for i in range(30_000)])
    probs = 1 / np.arange(1, len(vocab) + 1)
    probs /= probs.sum()
    idx = BM25Index()
    t = time.perf_counter()
    for _ in range(50):
        block = rng.choice(vocab, size=(1_000, 150), p=probs)
        for row in block:
            idx.add(" ".join(row))
    build = time.perf_counter() - t
    size = idx.nbytes()
    t = time.perf_counter()
    for _ in range(20):
        idx.search("t3 t250 t9000", k=10)
    q_ms = (time.perf_counter() - t) / 20 * 1e3
    with tempfile.TemporaryDirectory() as tmp:
        idx.save(Path(tmp) / "bm25.pkl")
        again = BM25Index.load(Path(tmp) / "bm25.pkl")
        assert np.array_equal(again.scores("t3 t250"), idx.scores("t3 t250"))
    print(f"  ok  build {build:.1f}s, index ~{size / 2**20:.0f} MB, vocab {idx.vocab_size}, "
          f"{q_ms:.1f} ms/query, save/load round-trips")
    print("core.bm25: all checks passed")


if __name__ == "__main__":
    _self_check()
