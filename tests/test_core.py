import io
import hashlib
import math

import numpy as np
import pytest

from core import config
from core.bm25 import BM25Index, tokenize
from core.dense import DenseIndex, normalize, top_k
from core.fusion import rrf
from core.ingest import (InvalidContent, PayloadTooLarge, SpooledUpload, UnsupportedMedia,
                         chunk_file, iter_chunks)
from core.metrics import mean_over_queries, ndcg_at_k
from core.svd_map import SpectralMap


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


def make_pdf(pages: list[str]) -> bytes:
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    for i, text in enumerate(pages):
        stream = f"BT /F1 6 Tf 10 700 Td ({text}) Tj ET".encode()
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    f"/Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode())
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        out += b"%010d 00000 n \n" % off
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    return bytes(out)


@pytest.mark.parametrize("stream,name,declared,max_bytes,exc", [
    (io.BytesIO(b""), "a.pdf", 600 * 2**20, config.MAX_UPLOAD_BYTES, PayloadTooLarge),
    (io.BytesIO(b"x" * 5000), "a.txt", None, 4096, PayloadTooLarge),
    (io.BytesIO(b"MZ\x90"), "a.exe", None, config.MAX_UPLOAD_BYTES, UnsupportedMedia),
    (io.BytesIO(b"MZ\x90\x00"), "a.pdf", None, config.MAX_UPLOAD_BYTES, UnsupportedMedia),
    (io.BytesIO(b""), "a.txt", None, config.MAX_UPLOAD_BYTES, InvalidContent),
])
def test_gates_reject_and_clean_up(tmp_path, stream, name, declared, max_bytes, exc):
    with pytest.raises(exc):
        with SpooledUpload(stream, name, declared, max_bytes=max_bytes, dir=tmp_path):
            pass
    assert list(tmp_path.iterdir()) == []


def test_pdf_100_pages_chunked_with_overlap_and_pages(tmp_path):
    pages = [" ".join(f"p{p}w{j}" for j in range(40)) for p in range(1, 101)]
    with SpooledUpload(io.BytesIO(make_pdf(pages)), "doc.pdf", dir=tmp_path) as up:
        assert up.size > 0 and len(up.sha256) == 64 and up.encoding is None
        chunks = list(up.chunks())
        spooled = up.path
    assert not spooled.exists()
    assert chunks[-1].end_word == 4000
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_word - b.start_word == 30
        assert a.text.split()[-30:] == b.text.split()[:30]
    for c in chunks:
        first = c.text.split()[0]
        assert c.page == int(first[1:first.index("w")])


def test_latin1_text_falls_back(tmp_path):
    f = tmp_path / "old.txt"
    f.write_bytes("caf\xe9 cr\xe8me br\xfbl\xe9e".encode("latin-1"))
    assert next(chunk_file(f)).text == "café crème brûlée"


def test_utf16_text_detected(tmp_path):
    import codecs

    f = tmp_path / "wide.txt"
    f.write_bytes(codecs.BOM_UTF16_LE + "hello unicode world".encode("utf-16-le"))
    assert "hello" in next(chunk_file(f)).text


def test_max_chunks_cap():
    words = ((f"w{i}", None) for i in range(10_000))
    assert len(list(iter_chunks(words, max_chunks=5))) == 5


def test_blockwise_search_matches_argsort(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5_000, config.EMBED_DIM))
    index = DenseIndex(tmp_path / "d.f32")
    index.add(X[:2_000])
    index.add(X[2_000:])
    q = rng.normal(size=config.EMBED_DIM)
    ids, _ = index.search(q, 25, rows=777)
    assert np.array_equal(ids, np.argsort(-(normalize(X) @ normalize(q)), kind="stable")[:25])


def test_top_k_edges():
    assert top_k(np.array([]), 3).size == 0
    assert list(top_k(np.array([0.1, 0.9, 0.5]), 10)) == [1, 2, 0]


def test_bm25_hand_computed():
    idx = BM25Index()
    for t in ["cat cat dog", "dog bird", "fish"]:
        idx.add(t)
    expected = math.log(8 / 3) * 5 / (2 + 1.5 * 1.375)
    assert idx.scores("cat")[0] == pytest.approx(expected, abs=1e-12)


def test_tokenize_keeps_identifiers():
    assert tokenize("The x86_64 build of np.argpartition") == ["x86_64", "build", "np", "argpartition"]


def test_rrf_consensus():
    hits = rrf([[1, 2, 3], [4, 2, 5]])
    assert hits[0].id == 2 and hits[0].ranks == (2, 2)


def test_metrics():
    assert mean_over_queries([([3, 1], {1: 1}), ([1], {1: 1})])["mrr"] == 0.75
    assert ndcg_at_k([1, 2], {1: 2, 2: 1}) == 1.0


def test_svd_matches_numpy():
    rng = np.random.default_rng(2)
    D = normalize(rng.normal(size=(500, config.EMBED_DIM)) * np.r_[6.0, 3.0, np.ones(382)])
    m = SpectralMap().fit(D, rows=64)
    _, S, _ = np.linalg.svd(D.astype(np.float64), full_matrices=False)
    assert m.eta == pytest.approx((S[:2] ** 2).sum() / (S ** 2).sum(), abs=1e-9)


@pytest.fixture(scope="module")
def embedder():
    try:
        from core.dense import Embedder
        return Embedder()
    except Exception as e:
        pytest.skip(f"embedder unavailable: {e}")


def test_pipeline_end_to_end(tmp_path, embedder):
    topics = {
        "cats": "Domestic cats sleep up to sixteen hours a day and groom their fur with rough tongues. ",
        "finance": "The central bank raised interest rates to curb inflation and stabilise the currency. ",
        "build": "Release wheels are compiled for the x86_64 and aarch64 targets using a cross toolchain. ",
    }
    doc = tmp_path / "doc.txt"
    doc.write_text("".join(text * 12 for text in topics.values()))
    chunks = list(chunk_file(doc, size=60, overlap=10))

    dense, bm25 = DenseIndex(tmp_path / "d.f32"), BM25Index()
    for c in chunks:
        bm25.add(c.text)
    for batch in embedder.embed(c.text for c in chunks):
        dense.add(batch)
    assert len(dense) == bm25.n_docs == len(chunks)

    def search(q):
        d_ids, _ = dense.search(embedder.embed_query(q), 10)
        s_ids, _ = bm25.search(q, 10)
        return rrf([d_ids.tolist(), s_ids.tolist()])

    assert "x86_64" in chunks[search("x86_64")[0].id].text
    assert "cats" in chunks[search("how long do kittens nap")[0].id].text
    assert "bank" in chunks[search("monetary policy")[0].id].text

    m = SpectralMap().fit_blocks(B for _, B in dense.iter_blocks())
    X2 = m.project(dense.matrix())
    q2 = m.project(embedder.embed_query("interest rates and inflation"))
    nearest = int(np.argmin(np.linalg.norm(X2 - q2, axis=1)))
    assert 0 < m.eta <= 1 and X2.shape == (len(chunks), 2)
    assert "bank" in chunks[nearest].text


def test_delete_doc_rebuilds_index(tmp_path):
    from core.engine import Engine
    from core.ingest import iter_chunks

    e = Engine(tmp_path, HashEmbedder())
    first = e.ingest("a.txt", iter_chunks(((f"alpha {i}", None) for i in range(500))), sha256="aaa", size=1)
    e.ingest("b.txt", iter_chunks(((f"beta {i}", None) for i in range(500))), sha256="bbb", size=1)
    out = e.delete_doc(first["sha256"])
    assert out["ok"] is True and out["remaining_docs"] == 1
    assert len(e.docs) == 1 and e.docs[0]["name"] == "b.txt"
    assert all(c["doc"] == "b.txt" for c in e.chunks(0, 10))
