from __future__ import annotations

import json
import os
import resource
import sys
import threading
import time
from array import array
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from core import config
from core.bm25 import BM25Index
from core.dense import DenseIndex, Embedder
from core.fusion import rrf
from core.svd_map import SpectralMap


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 2**20 if sys.platform == "darwin" else rss / 2**10


@dataclass(slots=True, frozen=True)
class Passage:
    rank: int
    chunk_id: int
    doc: str
    page: int | None
    text: str
    rrf: float
    dense: float
    bm25: float
    dense_rank: int | None
    bm25_rank: int | None


class Engine:
    def __init__(self, index_dir: str | os.PathLike = config.INDEX_DIR, embedder: Embedder | None = None):
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self._embedder = embedder
        self.dense = DenseIndex(self.dir / "dense.f32")
        self.chunks_path = self.dir / "chunks.jsonl"
        self.chunks_path.touch(exist_ok=True)
        self.offsets = array("Q")
        self.bm25 = BM25Index()
        self.svd: SpectralMap | None = None
        self.docs: list[dict] = []
        self._load()

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    def __len__(self) -> int:
        return len(self.offsets)

    def _load(self) -> None:
        meta = self.dir / "meta.json"
        if not meta.exists():
            self.reset()
            return
        state = json.loads(meta.read_text())
        self.docs = state["docs"]
        self.offsets = array("Q", state["offsets"])
        self.bm25 = BM25Index.load(self.dir / "bm25.pkl")
        self.svd = SpectralMap.from_dict(state["svd"]) if state.get("svd") else None
        n = len(self.offsets)
        if self.bm25.n_docs != n or len(self.dense) < n:
            self.reset()
        elif len(self.dense) > n:
            self._truncate(n)

    def _truncate(self, n: int) -> None:
        with open(self.dense.path, "r+b") as f:
            f.truncate(n * self.dense.dim * 4)
        self.dense = DenseIndex(self.dense.path, self.dense.dim)
        end = 0
        if n:
            with open(self.chunks_path, "rb") as f:
                f.seek(self.offsets[n - 1])
                f.readline()
                end = f.tell()
        with open(self.chunks_path, "r+b") as f:
            f.truncate(end)

    def _save(self) -> None:
        self.bm25.save(self.dir / "bm25.pkl")
        state = {"docs": self.docs, "offsets": self.offsets.tolist(),
                 "svd": self.svd.to_dict() if self.svd else None}
        (self.dir / "meta.json").write_text(json.dumps(state))

    def reset(self) -> None:
        with self.lock:
            self.dense.clear()
            self.chunks_path.write_bytes(b"")
            self.offsets = array("Q")
            self.bm25 = BM25Index()
            self.svd = None
            self.docs = []
            for name in ("bm25.pkl", "meta.json"):
                (self.dir / name).unlink(missing_ok=True)

    def chunk(self, chunk_id: int) -> dict:
        with open(self.chunks_path, "rb") as f:
            f.seek(self.offsets[chunk_id])
            return json.loads(f.readline())

    def chunks(self, offset: int = 0, limit: int = 20) -> list[dict]:
        ids = range(offset, min(offset + limit, len(self)))
        return [self.chunk(i) for i in ids]

    def ingest(self, name: str, chunks, sha256: str = "", size: int = 0) -> dict:
        with self.lock:
            if sha256 and any(d["sha256"] == sha256 for d in self.docs):
                raise ValueError(f"{name} is already indexed")
            try:
                return self._ingest(name, chunks, sha256, size)
            except BaseException:
                self._load()
                raise

    def _ingest(self, name: str, chunks, sha256: str, size: int) -> dict:
        t0 = time.perf_counter()
        first = len(self)
        batch: list[str] = []
        with open(self.chunks_path, "ab") as out:
            for c in chunks:
                if len(self) >= config.MAX_CHUNKS:
                    break
                self.offsets.append(out.tell())
                record = asdict(c) | {"id": len(self.offsets) - 1, "doc": name}
                out.write(json.dumps(record, ensure_ascii=False).encode() + b"\n")
                self.bm25.add(c.text)
                batch.append(c.text)
                if len(batch) == self.embedder.batch_size:
                    self._embed(batch)
                    batch = []
            if batch:
                self._embed(batch)
        added = len(self) - first
        if added:
            self.svd = SpectralMap().fit_blocks(B for _, B in self.dense.iter_blocks())
        doc = {"name": name, "sha256": sha256, "bytes": size, "first_chunk": first, "n_chunks": added,
               "seconds": round(time.perf_counter() - t0, 2)}
        self.docs.append(doc)
        self._save()
        return doc | {"total_chunks": len(self), "capped": len(self) >= config.MAX_CHUNKS}

    def _embed(self, texts: list[str]) -> None:
        for E in self.embedder.embed(texts):
            self.dense.add(E)

    def search(self, q: str, k: int = 5, depth: int = 50) -> tuple[list[Passage], np.ndarray]:
        with self.lock:
            qv = self.embedder.embed_query(q)
            if not len(self):
                return [], qv
            d_ids, _ = self.dense.search(qv, depth)
            s_ids, _ = self.bm25.search(q, depth)
            hits = rrf([d_ids.tolist(), s_ids.tolist()], top_n=k)
            ids = np.array([h.id for h in hits], dtype=np.int64)
            dense_scores = np.asarray(self.dense.matrix()[ids]) @ qv if len(ids) else np.empty(0)
            bm25_scores = self.bm25.scores(q)
            passages = []
            for rank, (h, ds) in enumerate(zip(hits, dense_scores), start=1):
                c = self.chunk(h.id)
                passages.append(Passage(rank, h.id, c["doc"], c["page"], c["text"], h.score, float(ds),
                                        float(bm25_scores[h.id]), h.ranks[0], h.ranks[1]))
            return passages, qv

    def ranking(self, q: str, mode: str = "hybrid", k: int = 10, depth: int = 50) -> list[int]:
        with self.lock:
            if mode == "bm25":
                return self.bm25.search(q, k)[0].tolist()
            qv = self.embedder.embed_query(q)
            d_ids = self.dense.search(qv, depth if mode == "hybrid" else k)[0].tolist()
            if mode == "dense":
                return d_ids
            s_ids = self.bm25.search(q, depth)[0].tolist()
            return [h.id for h in rrf([d_ids, s_ids], top_n=k)]

    def map(self, q: str | None = None, k: int = 5, max_points: int = 1500) -> dict:
        with self.lock:
            if not len(self) or self.svd is None:
                return {"points": [], "query": None, "hits": [], "eta": 0.0, "n_total": 0}
            n = len(self)
            hits: list[int] = []
            query = None
            if q:
                passages, qv = self.search(q, k)
                hits = [p.chunk_id for p in passages]
                x, y = self.svd.project(qv)
                query = {"x": float(x), "y": float(y)}
            ids = np.arange(n) if n <= max_points else np.unique(np.r_[
                np.random.default_rng(0).choice(n, max_points, replace=False), hits]).astype(np.int64)
            X2 = self.svd.project(np.asarray(self.dense.matrix()[ids]))
            points = [{"id": int(i), "x": float(a), "y": float(b)} for i, (a, b) in zip(ids, X2)]
            return {"points": points, "query": query, "hits": hits, "eta": self.svd.eta, "n_total": n}

    def stats(self) -> dict:
        return {
            "n_chunks": len(self),
            "n_docs": len(self.docs),
            "docs": self.docs,
            "vocab_size": self.bm25.vocab_size,
            "avgdl": round(self.bm25.avgdl, 2),
            "eta": self.svd.eta if self.svd else 0.0,
            "bm25_mb": round(self.bm25.nbytes() / 2**20, 2),
            "dense_mb_on_disk": round(len(self) * config.EMBED_DIM * 4 / 2**20, 2),
            "peak_rss_mb": round(peak_rss_mb(), 1),
            "max_chunks": config.MAX_CHUNKS,
        }

