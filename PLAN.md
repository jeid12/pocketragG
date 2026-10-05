# PocketRAG — Build Plan

Source of truth: `projectrag.pdf` (spec). This file records the decisions the spec leaves
open, the risks, and the day-by-day checklist with a concrete "done" test for each day.

---

## 0. Decisions the spec leaves open

| # | Question | Decision | Why |
|---|----------|----------|-----|
| D1 | Where do the 384-d embeddings come from? The spec says "pure NumPy" but never names a model. | `sentence-transformers/all-MiniLM-L6-v2` (384-d) run through **ONNX Runtime** (via `fastembed`) on the **backend only**. The client never embeds. | PyTorch alone would blow the 350 MB envelope. ONNX MiniLM is about 90 MB on disk and has a small RSS. The "zero black-box" rule is about *retrieval* (indexing, scoring, fusion). The encoder is a fixed feature map, so say this plainly in the oral exam. |
| D2 | Port 8550 is both the backend port and Flet's default web port. | Backend stays on **8550** (as in the spec). Run the Flet web client with `--port 8551`. | Otherwise both servers collide when you run them locally. |
| D3 | How to do the SVD without holding U (N×384) in memory? | Stream-accumulate the Gram matrix **G = DᵀD** (384×384) block by block, then `np.linalg.eigh(G)`. The eigenvalues are σ², and the eigenvectors are V. | Same V and η as `np.linalg.svd(D)` (Eckart–Young still applies), but memory is O(d²) instead of O(N·d). This is a good defence talking point. |
| D4 | Centre D before the SVD? | **No** (the spec uses raw D). If the plot looks like one blob, try mean-centring (which turns it into PCA) and compare. | The first singular vector of uncentred unit vectors is usually the corpus "mean direction". |
| D5 | Where to store the dense matrix | `np.memmap` float32 file under `data/index/`, scored in blocks of 8192 rows with a running top-k merge. | See risk R1. |
| D6 | Tokeniser for BM25 | Lowercase, `re.finditer(r"\w+")`, no stemming, and a small stop-word list. Keep technical tokens like `x86_64` whole. | The spec says BM25 exists "to resolve precise technical identifiers". |
| D7 | How the Claude API is called | Official `anthropic` SDK (or raw `requests`, as the spec lists), model set by the `CLAUDE_MODEL` env var. Prompt: "answer ONLY from these numbered passages; cite [n]; else say not found". | Presentation-layer isolation invariant. |
| D8 | State between requests | One in-process index per server (single-user demo). Persist to `data/index/` so a restart doesn't re-embed. | Keeps scope within 20 h. |

## 1. Risks (read before Day 1)

**R1: The 500 MB / 350 MB numbers don't fit together unless you scope them.**
500 MB of plain text is about 85 M words. With a 150-word window and stride 120 that's roughly 700k chunks.
- Dense matrix: 700k × 384 × 4 B ≈ **1.1 GB** (float16: 540 MB). This can't be held in RAM.
- Python-dict BM25 postings for 85 M tokens: **several GB**.
- MiniLM on CPU embeds about 200–500 chunks/s, so **30–60 min** to ingest.

Plan: honour the 500 MB limit on **upload and streaming** (Gates 1–3 plus the chunker never load the file
into memory). Put a configurable **`MAX_CHUNKS` cap** (default 50 000) on what gets indexed. Use memmap
(D5) and compact postings (`array('I')` rather than lists of ints). In the defence, say the RSS
bound you actually measured and the chunk cap. Don't claim 500 MB of text is fully indexed in 350 MB.
Note that real 500 MB PDFs are mostly images and fonts, so their extracted text is far smaller.

**R2: pypdf text extraction is page-granular.** Stream page by page (`reader.pages` is lazy) and feed
the words into the chunker generator. Never build `"".join(all_pages)`.

**R3: `tracemalloc` doesn't see everything** (for example ONNX Runtime's native allocations).
For the envelope, measure **RSS** with `resource.getrusage(RUSAGE_SELF).ru_maxrss` (bytes on
macOS, KiB on Linux) as well as tracemalloc.

**R4: The Python 3.14 venv.** `flet build apk` and some wheels (onnxruntime) can lag behind new
Pythons. If an install fails, recreate the venv with Python 3.12. The Docker image pins 3.12
either way.

**R5: The Android app must reach the backend.** On the device, the backend URL isn't `localhost`.
Put `BACKEND_URL` in the client settings (LAN IP of the laptop, e.g. `http://192.168.x.x:8550`)
and add `android:usesCleartextTraffic` or use HTTPS. Test this early (Day 7), not on Day 9.

**Measured on setup day (2026-09-28, Intel Mac, Python 3.12, fastembed 0.8.1 / onnxruntime 1.23.2):**
- MiniLM loads and embeds fine: dim 384, and the output is **already L2-normalised**. Still apply `normalize()`
  (it's idempotent and it's the spec's Eq. 2), but don't count on it to fix anything.
- **Peak RSS is about 287 MB with just the model loaded**, so only about 60 MB is left under the 350 MB envelope.
  Consequences: embed in small batches (≤ 32); never keep the chunk *texts* in RAM (store byte offsets
  and read them back from disk); keep the dense matrix in memmap. If the budget still fails, either run the
  encoder in a separate worker process and measure the *retrieval* process, or state the envelope
  as "excluding the fixed 90 MB model weights". Decide this with real numbers on Day 2.
- Throughput was about 270 embeddings/s on short sentences. 150-word chunks will be slower, so budget roughly
  **10–15 min for 50k chunks**. Use a small document for the demo.

**R4 (resolved):** onnxruntime has no Python 3.14 wheel for Intel macOS, so `env/` was recreated on
**Python 3.12** (same as Docker).

**R6: Flet is at 1.0.x.** Most tutorials and Stack Overflow answers are written for the 0.2x API, and
entry points and some controls were renamed. Use the 1.0 docs only.

## 2. API contract (freeze on Day 6; the client codes against this)

```
POST /ingest   multipart file  -> {doc_id, n_chunks, vocab_size, avgdl, sha256, bytes}
                                  413 if > 500 MB, 415 if bad extension/magic, 422 if null bytes
POST /query    {q, k=5}        -> {answer, passages:[{rank, chunk_id, text, dense, bm25, rrf,
                                    dense_rank, bm25_rank}], latency_ms}
GET  /map      ?q=...          -> {points:[{chunk_id, x, y}], query:{x,y}|null, eta}
GET  /stats                    -> {n_chunks, vocab_size, avgdl, mrr, ndcg}
GET  /health                   -> {ok: true}
```

## 3. Sprint checklist (2 h/day)

Every `python -m core.X` needs an `if __name__ == "__main__":` block that runs a small self-check
and exits non-zero on failure. That's the verification target in the spec.

### Day 1: `core/ingest.py`
- [x] `SpooledUpload` context manager (`__enter__`/`__exit__`): copies the stream to a temp file in 8 MB
      reads, counts bytes, raises `PayloadTooLarge` past 524 288 000, and always unlinks the temp file.
- [x] `sniff(path, ext)`: PDF must start with `%PDF-`; txt/md must decode as UTF-8 and contain no `0x00`.
- [x] `iter_words(path)`: generator (pdf page by page, text line by line, `re.finditer`).
- [x] `iter_chunks(words, size=150, overlap=30)`: yields `Chunk(slots=True, frozen=True)` with
      `id, text, start_word, end_word, page`.
- **Done when:** a 600 MB dummy file gives a 413, a renamed .exe gives a 415, and a 100-page PDF yields
  chunks whose consecutive word offsets overlap by exactly 30.

### Day 2: `core/dense.py`
- [x] `normalize(X)` with `np.linalg.norm(..., axis=1, keepdims=True)` and `np.maximum(·, 1e-12)`.
- [x] `Embedder` wrapper (fastembed MiniLM, batch 64, returns float32).
- [x] `DenseIndex`: memmap writer (append), `search(q, k)` in blocks with `argpartition` + merge.
- **Done when:** all row norms equal 1 ± 1e-6, the top-k matches a full `argsort` on random data, and the
  query "cat" ranks a cat sentence above a finance sentence.

### Day 3: `core/bm25.py`
- [x] Inverted index `dict[str, array('I')]` of chunk ids plus a parallel tf array; doc lengths; avgdl.
- [x] IDF per Eq. 4 (the `+1` form, always ≥ 0) and score per Eq. 5, k1 = 1.5, b = 0.75.
- **Done when:** a hand-computed 3-doc example matches to 1e-9, and an exact identifier query (`x86_64`)
  ranks its chunk #1.

### Day 4: `core/fusion.py` + `core/metrics.py`
- [x] `rrf(rankings: list[list[int]], k=60) -> list[(id, score)]`. Documents missing from a list get 0
      from that list.
- [x] `mrr_at_k`, `dcg_at_k`, `ndcg_at_k` (Eq. 10, graded rel).
- **Done when:** textbook examples match, and a doc that is #1 in only one list loses to a doc that is #2 in both.

### Day 5: `core/svd_map.py`
- [x] `fit(D_memmap)`: blockwise `G += B.T @ B`, then `eigh`, sort descending, V2 and η (Eq. 9).
- [x] `project(X)` gives `X @ V2`.
- **Done when:** V2 and η match `np.linalg.svd` on a 2 000×384 random matrix (up to sign), and η is in (0, 1].

### Day 6: `core/synthesizer.py` + `server.py`
- [x] Synthesizer: numbered passages go in, the answer comes out with `[n]` citations. If there's no API key,
      it falls back to returning the top passages verbatim, so the demo never hard-fails.
- [x] FastAPI routes per §2, with CORS enabled for the Flet web client.
- **Done when:** `curl -F file=@sample.pdf :8550/ingest`, then `/query`, returns cited passages.

### Day 7: `client/app.py`
- [x] Centred container with `max-width 750`, `NavigationBar` with 4 destinations, and a `BACKEND_URL` setting.
- [x] Docs view (file picker, upload, stats) and Chat view (answer plus expandable cited passages).
- **Done when:** `flet run client/app.py` ingests and answers against the local backend. Also run it
  once on a phone (`flet run --android` / Flet app) to catch R5 early.

### Day 8: `client/views/geometry_view.py` + `theory_view.py`
- [x] `flet.canvas` scatter: chunks in grey, top-k highlighted, query as a star, η in the corner.
- [x] Theory view: formulas plus the live MRR/NDCG from `/stats`.
- **Done when:** the query marker lands near its retrieved chunks with `flet run --web --port 8551 client/app.py`.

### Day 9: Docker + APK
- [x] `docker compose up --build` gives a healthy `/health` on :8550. The ONNX model is baked into the
      image at build time (no download at runtime).
- [x] `flet build apk`, `adb install build/apk/*.apk`, query from the phone to the laptop backend (tested on the Android 16 emulator via `10.0.2.2`).
- **Done when:** the APK is under 20 MB and answers a query over Wi-Fi.

### Day 10: Verification + video
- [x] `eval.py`: about 20 hand-labelled queries in `data/eval/qrels.json`. Report MRR@10 and NDCG@10 for
      dense only, BM25 only, and RRF (the ablation table is strong defence material).
- [x] Memory run: ingest the largest test file and record peak RSS and tracemalloc peak.
- [ ] Record the 60 s demo (phone + browser side by side), following spec §7.1.

## 4. Progress log

**Days 1–5 done (2026-09-29).** All `python -m core.{ingest,dense,bm25,fusion,metrics,svd_map}`
self-checks pass, and so does `pytest` (17 tests, including an end-to-end ingest → dense + BM25 → RRF → SVD run).
Measured on the Intel Mac:
| What | Result |
|---|---|
| Ingest 20 MB text | 28k chunks in 3.9 s (about 100 s per 500 MB); tracemalloc peak 16 MB, set by the 8 MB block size and independent of file size |
| Dense search, 20k rows | 3.2 ms/query (blockwise argpartition, exact match with argsort) |
| BM25, 50k chunks × 150 tokens | 20 s build, **about 53 MB** of index, 1.5 ms/query |
| SVD via Gram eigh | V2, σ and η equal `np.linalg.svd` to 1e-6 / 1e-9 |

Memory budget at 50k chunks: model (about 287 MB) + BM25 (about 53 MB) is already about 340 MB of the 350 MB limit. The dense
matrix is on disk and costs only one 12 MB block. **Decision for Day 6:** either lower `MAX_CHUNKS` to about 20k for the
demo, or run the encoder in a child process during ingest so the serving process holds only the indexes.
Choose after measuring real RSS in `server.py`.

Implementation notes for the defence:
- tracemalloc slows pure-Python loops by about 10×, so time runs untraced and measure memory in a separate pass.
- MiniLM already returns unit vectors, and `normalize()` (Eq. 2) is kept on purpose and is idempotent.
- The BM25 tokenizer is `\w+` lowercased, so identifiers like `x86_64` stay whole tokens.
- The SVD sign convention (largest loading positive) keeps the 2D plot from flipping between re-fits.

**Days 6–10 done (2026-09-29)**, apart from installing the APK on a phone and recording the video (see below).

| What | Result |
|---|---|
| Backend peak RSS (macOS, model + FastAPI + 1.3k chunks) | **315 MB** |
| Backend peak RSS (Docker, Linux) | 365 MB → **337 MB** with `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=131072` (in the Dockerfile) |
| Embedding batch size | 32 → **4**: same speed (about 25 chunks/s on 150-word chunks), peak 687 MB → 253 MB |
| Ingest *Origin of Species* (970 KB, 1,322 chunks) | 51 s on the Mac, 112 s in Docker |
| Query latency | retrieval 8–16 ms, total about 0.7 s including the Claude call |
| `MAX_CHUNKS` | lowered to **20k** so BM25 (about 1 MB per 1k chunks) keeps the total under 350 MB |
| Tests | 26 passing (core + FastAPI routes + crash-rollback) |

Evaluation (`python eval.py`, 24 paraphrased questions, labels = chunks containing a gold phrase, table of contents and licence excluded):

| retriever | MRR@10 | NDCG@10 | median ms |
|---|---|---|---|
| dense | 0.739 | 0.508 | 6.7 |
| BM25 | 0.670 | 0.422 | 1.9 |
| hybrid (RRF) | 0.733 | 0.505 | 10.8 |

What to say about this in the defence: the questions are deliberately paraphrased, which favours embeddings. Hybrid fixes
BM25's misses (e.g. "sterile castes in ant colonies": dense #3 → hybrid #1). When BM25 finds nothing relevant in its top 50, the fused
list slightly dilutes dense, so hybrid ties dense instead of beating it. The BM25 failures are vocabulary mismatches such as
"freshwater" vs "fresh-water", where the tokenizer splits on the hyphen. Adding keyword-style queries, or a hyphen-joining
tokenizer rule, would show where BM25 earns its place.

Other decisions made while building:
- The Claude model is `claude-opus-5` with effort `low` (it only formats) and the server-side refusal fallback enabled. Every `[n]` citation is
  checked against the passages; an answer with an invalid citation is replaced by the extractive fallback.
- The extractive fallback picks the sentence with the most query-term overlap from each passage, not the first sentences.
- A failed ingest (e.g. a corrupt PDF) rolls the index back to the last saved state. Nothing already indexed is lost.
- The client bundles DejaVu Sans so the math symbols (₁ ₂ q̂ Σ ‖) render on web and Android without network fonts.
- Android blocks plain HTTP by default. `client/pyproject.toml` sets `usesCleartextTraffic` through
  `[tool.flet.android.manifest_application]`.

APK: builds in about 2 min once cached. Installed on the emulator, it connects over plain HTTP, ingests, answers with citations and draws the map.
**Size is 53 MB (arm64) and does not meet the spec's < 20 MB.** Our code is 0.8 MB. The rest is Flet's fixed runtime: Flutter engine 11 MB, Flet
Dart app 14 MB, and an embedded Python 3.14 with its stdlib (about 22 MB). Even an empty Flet app is about 45 MB, so < 20 MB would need a
non-Python client (e.g. plain Flutter/Dart) or serving the web build as a PWA. Say this honestly in the defence.

Still to do by hand: install the APK on a real phone (`adb install`), point it at the laptop's LAN IP, and record the 60 s video.

## 5. Buffer / cut list if behind
Cut in this order: theory_view polish, then persisted index, then APK (demo the web build on a phone
browser instead), then Docker (run uvicorn directly). Never cut: the core engines, eval.py, and the memory measurement.
