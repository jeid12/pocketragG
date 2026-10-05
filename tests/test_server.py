import hashlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

import server
from core.engine import Engine
from core.synthesizer import extractive
from tests.test_core import make_pdf


class HashEmbedder:
    batch_size = 4

    def _vec(self, text):
        v = np.zeros(384, dtype=np.float32)
        for w in text.lower().split():
            v[int(hashlib.md5(w.strip(".,?").encode()).hexdigest(), 16) % 384] += 1
        return v / max(np.linalg.norm(v), 1e-12)

    def embed(self, texts):
        texts = list(texts)
        yield np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server.config, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(server, "synthesize", lambda q, ps: extractive(q, ps, "offline test"))
    monkeypatch.setattr(server, "Engine", lambda: Engine(tmp_path / "index", HashEmbedder()))
    with TestClient(server.app) as c:
        yield c


def upload(client, name, data):
    return client.post("/ingest", files={"file": (name, data)})


def test_ingest_query_map_stats(client):
    text = ("Frogs and newts are absent from oceanic islands because sea water kills their spawn. " * 20 +
            "Pigeons descend from the rock pigeon according to breeders. " * 20)
    r = upload(client, "notes.txt", text.encode())
    assert r.status_code == 200 and r.json()["n_chunks"] > 1

    r = client.post("/query", json={"q": "why no frogs on islands", "k": 3})
    body = r.json()
    assert r.status_code == 200 and body["mode"] == "extractive"
    assert "Frogs" in body["passages"][0]["text"] and body["passages"][0]["rank"] == 1
    assert {"dense", "bm25", "rrf", "dense_rank", "bm25_rank"} <= body["passages"][0].keys()

    m = client.get("/map", params={"q": "rock pigeon"}).json()
    assert len(m["points"]) == client.get("/stats").json()["n_chunks"] and m["query"] and 0 < m["eta"] <= 1

    assert client.get("/chunks", params={"limit": 2}).json()["chunks"][1]["id"] == 1


def test_pdf_upload_keeps_pages(client):
    pages = [" ".join(f"page{p} word{j}" for j in range(100)) for p in range(1, 6)]
    r = upload(client, "doc.pdf", make_pdf(pages))
    assert r.status_code == 200
    chunks = client.get("/chunks", params={"limit": 50}).json()["chunks"]
    assert chunks[0]["page"] == 1 and chunks[-1]["page"] >= 4


@pytest.mark.parametrize("name,data,status", [
    ("x.exe", b"MZ\x90\x00", 415),
    ("x.pdf", b"not a pdf", 415),
])
def test_rejections(client, name, data, status):
    assert upload(client, name, data).status_code == status


def test_corrupt_pdf_is_rejected_and_index_survives(client):
    assert upload(client, "good.txt", b"alpha beta gamma delta " * 100).status_code == 200
    before = client.get("/health").json()["chunks"]
    r = upload(client, "bad.pdf", b"%PDF-1.4\n" + b"\x8f garbage " * 500)
    assert r.status_code == 422, r.text
    assert client.get("/health").json()["chunks"] == before
    assert client.post("/query", json={"q": "gamma"}).json()["passages"]


def test_duplicate_and_reset(client):
    first = upload(client, "a.txt", b"alpha beta gamma " * 100)
    assert first.status_code == 200
    assert upload(client, "a.txt", b"alpha beta gamma " * 100).status_code == 409
    deleted = client.delete(f"/documents/{first.json()['sha256']}")
    assert deleted.status_code == 200 and deleted.json()["remaining_docs"] == 0
    assert client.post("/reset").json() == {"ok": True}
    assert client.get("/health").json()["chunks"] == 0
    assert client.post("/query", json={"q": "alpha"}).json()["passages"] == []


def test_engine_reopens_from_disk(tmp_path):
    e = Engine(tmp_path, HashEmbedder())
    from core.ingest import iter_chunks
    e.ingest("w.txt", iter_chunks(((f"w{i}", None) for i in range(500))))
    again = Engine(tmp_path, HashEmbedder())
    assert len(again) == len(e) and again.svd is not None
    assert again.chunk(2)["text"] == e.chunk(2)["text"]


def test_interrupted_ingest_rolls_back_to_last_save(tmp_path):
    from core.ingest import iter_chunks
    e = Engine(tmp_path, HashEmbedder())
    e.ingest("a.txt", iter_chunks(((f"a{i}", None) for i in range(500))))
    saved = len(e)

    def crash():
        yield from iter_chunks(((f"b{i}", None) for i in range(500)))
        raise RuntimeError("power cut")

    with pytest.raises(RuntimeError):
        e.ingest("b.txt", crash())
    again = Engine(tmp_path, HashEmbedder())
    assert len(again) == len(again.dense) == again.bm25.n_docs == saved
    assert [d["name"] for d in again.docs] == ["a.txt"]
    assert again.chunks_path.read_bytes().count(b"\n") == saved
