from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from itertools import batched
from pathlib import Path

import numpy as np

from core import config


def normalize(X: np.ndarray, eps: float = config.EPS) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    n = scores.shape[0]
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k] if k < n else np.arange(n)
    return idx[np.argsort(-scores[idx], kind="stable")]


class Embedder:

    def __init__(self, model_name: str = config.EMBED_MODEL, batch_size: int = config.EMBED_BATCH,
                 cache_dir: str | os.PathLike = config.MODEL_CACHE):
        from fastembed import TextEmbedding

        self.batch_size = batch_size
        self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        for batch in batched(texts, self.batch_size):
            E = np.stack(list(self._model.embed(list(batch), batch_size=len(batch))))
            yield normalize(E)

    def embed_query(self, text: str) -> np.ndarray:
        return next(self.embed([text]))[0]


class DenseIndex:

    def __init__(self, path: str | os.PathLike, dim: int = config.EMBED_DIM):
        self.path = Path(path)
        self.dim = dim
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        size = self.path.stat().st_size
        row_bytes = dim * np.dtype(np.float32).itemsize
        if size % row_bytes:
            raise ValueError(f"{self.path} is not a whole number of {dim}-d float32 rows")
        self._n = size // row_bytes
        self._mm: np.memmap | None = None

    def __len__(self) -> int:
        return self._n

    def add(self, X: np.ndarray) -> range:
        X = np.ascontiguousarray(normalize(np.atleast_2d(X)), dtype=np.float32)
        if X.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {X.shape[1]}")
        with open(self.path, "ab") as f:
            f.write(X.tobytes())
        first = self._n
        self._n += X.shape[0]
        self._mm = None
        return range(first, self._n)

    def clear(self) -> None:
        self._mm = None
        self.path.write_bytes(b"")
        self._n = 0

    def matrix(self) -> np.ndarray:
        if self._n == 0:
            return np.empty((0, self.dim), dtype=np.float32)
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype=np.float32, mode="r", shape=(self._n, self.dim))
        return self._mm

    def iter_blocks(self, rows: int = config.SCORE_BLOCK_ROWS) -> Iterator[tuple[int, np.ndarray]]:
        D = self.matrix()
        for start in range(0, self._n, rows):
            yield start, np.asarray(D[start:start + rows])

    def search(self, q: np.ndarray, k: int, rows: int = config.SCORE_BLOCK_ROWS) -> tuple[np.ndarray, np.ndarray]:
        q = normalize(q)
        best_ids = np.empty(0, dtype=np.int64)
        best_scores = np.empty(0, dtype=np.float32)
        for start, B in self.iter_blocks(rows):
            s = B @ q
            local = top_k(s, k)
            ids = np.concatenate([best_ids, local + start])
            scores = np.concatenate([best_scores, s[local]])
            keep = top_k(scores, k)
            best_ids, best_scores = ids[keep], scores[keep]
        return best_ids, best_scores


def _self_check() -> None:
    import tempfile
    import time

    rng = np.random.default_rng(0)
    print("normalisation")
    X = rng.normal(size=(20_000, config.EMBED_DIM)).astype(np.float32) * rng.uniform(0.1, 50, size=(20_000, 1))
    Xn = normalize(X)
    assert np.allclose(np.linalg.norm(Xn, axis=1), 1, atol=1e-6)
    assert np.all(normalize(np.zeros((2, 4))) == 0), "eps must keep zero vectors finite"
    print("  ok  row norms = 1 +/- 1e-6, zero vector stays finite")

    print("blockwise top-k vs full sort")
    with tempfile.TemporaryDirectory() as tmp:
        index = DenseIndex(Path(tmp) / "dense.f32")
        for block in np.array_split(X, 7):
            index.add(block)
        assert len(DenseIndex(index.path)) == 20_000, "row count must survive reopen"
        for trial in range(5):
            q = rng.normal(size=config.EMBED_DIM)
            ids, scores = index.search(q, k=10, rows=3_000)
            full = np.argsort(-(Xn @ normalize(q)), kind="stable")[:10]
            assert np.array_equal(ids, full), (ids, full)
            assert np.all(np.diff(scores) <= 0)
        t = time.perf_counter()
        for _ in range(20):
            index.search(q, k=10)
        print(f"  ok  matches argsort on 5 queries; {(time.perf_counter() - t) / 20 * 1e3:.1f} ms/query over 20k rows")
        assert DenseIndex(index.path).search(q, 50_000)[0].shape == (20_000,), "k > N must return all rows"

    print("semantic sanity (MiniLM)")
    try:
        emb = Embedder()
    except Exception as e:
        print(f"  skip  embedder unavailable: {e}")
    else:
        docs = ["The cat curled up and slept on the warm windowsill.",
                "Quarterly revenue grew four percent on strong bond sales.",
                "Compile the kernel for x86_64 and ARM64 targets."]
        D = np.vstack(list(emb.embed(docs)))
        s = D @ emb.embed_query("a kitten napping in the sun")
        print(f"  cos = {np.round(s, 3)}")
        assert s.argmax() == 0
        print("  ok  cat passage ranks first")
    print("core.dense: all checks passed")


if __name__ == "__main__":
    _self_check()
