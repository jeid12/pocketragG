# PocketRAG

First-principles hybrid document retrieval: dense cosine search on the unit hypersphere, Okapi BM25,
Reciprocal Rank Fusion and a truncated-SVD map. The backend is Python + NumPy (FastAPI). The client is
Flet (web and Android). Gemini only formats answers from passages that were already retrieved.

Spec: `projectrag.pdf`. Design decisions, measurements and progress: [PLAN.md](PLAN.md).

## Setup

```bash
source env/bin/activate                     # Python 3.12
pip install -r requirements-dev.txt
brew install tesseract                      # OCR for images and scanned PDFs (optional but recommended)
cp .env.example .env                        # optional: GEMINI_API_KEY
```

Without a working API key, answers fall back to extractive mode: the best-matching sentence from each
retrieved passage, with citations. The app shows which mode it used and why.

Set `GEMINI_API_KEY` in `.env` to enable model-generated answers.

## Run

```bash
uvicorn server:app --host 0.0.0.0 --port 8550      # backend
flet run --web --port 8551 client/app.py            # web client → http://localhost:8551
```

Deep links: `/ask?q=...` asks a question, and `/map?q=...` projects a query onto the map.

With Docker (backend only):

```bash
docker compose up --build                           # http://localhost:8550/health
```

## Android

```bash
cd client
flet build apk --split-per-abi --arch arm64-v8a x86_64 -o ../build/apk
adb install ../build/apk/app-arm64-v8a-release.apk
```

The phone and the laptop must be on the same Wi-Fi network. Start the backend with `--host 0.0.0.0`,
then in the app's Docs tab set the backend URL to `http://<laptop LAN IP>:8550`. Android no longer
defaults to localhost, so this must point at the laptop backend. The app remembers the URL once you
set it. Find the IP with `ipconfig getifaddr en0`.

## Verify

```bash
python -m core.ingest        # also: core.dense, core.bm25, core.fusion, core.metrics, core.svd_map
pytest -q                    # 26 tests
python eval.py               # MRR@10 / NDCG@10 for dense, BM25 and hybrid, plus memory
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/ingest` | multipart `file` (PDF, Office, images via OCR, text/markup ≤ 500 MB) → 413 / 415 / 422 / 409 |
| POST | `/query` | `{"q": "...", "k": 5}` → answer, mode, citations, passages with RRF/cosine/BM25 scores and ranks |
| GET | `/map?q=` | 2D SVD coordinates of chunks, the query point, retrieved ids, η |
| GET | `/chunks?offset=&limit=` | chunk inspection |
| GET | `/stats` | corpus stats, memory, latest eval results |
| POST | `/reset` | clear the index |
| GET | `/health` | liveness |
