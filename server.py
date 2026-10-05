from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from core import config
from core.engine import Engine
from core.ingest import IngestError, SpooledUpload
from core.synthesizer import synthesize

engine: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = Engine()
    yield


app = FastAPI(title="PocketRAG", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryIn(BaseModel):
    q: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    synthesize: bool = True


@app.get("/health")
def health() -> dict:
    return {"ok": True, "chunks": len(engine)}


@app.post("/ingest")
async def ingest(request: Request, file: UploadFile = File(...)) -> dict:
    declared = file.size
    if declared is None and request.headers.get("content-length"):
        declared = int(request.headers["content-length"])

    def work() -> dict:
        with SpooledUpload(file.file, file.filename or "upload", declared, dir=config.UPLOAD_DIR) as up:
            doc = engine.ingest(file.filename, up.chunks(max_chunks=config.MAX_CHUNKS), up.sha256, up.size)
        return doc | {"vocab_size": engine.bm25.vocab_size, "avgdl": round(engine.bm25.avgdl, 2)}

    config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        return await run_in_threadpool(work)
    except IngestError as e:
        raise HTTPException(e.status, str(e))
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.post("/query")
def query(body: QueryIn) -> dict:
    t0 = time.perf_counter()
    passages, _ = engine.search(body.q, body.k)
    retrieval_ms = (time.perf_counter() - t0) * 1e3
    result = synthesize(body.q, [p.text for p in passages]) if body.synthesize else \
        {"answer": "", "mode": "none", "citations": [], "reason": ""}
    return {
        **result,
        "passages": [asdict(p) for p in passages],
        "retrieval_ms": round(retrieval_ms, 1),
        "total_ms": round((time.perf_counter() - t0) * 1e3, 1),
    }


@app.get("/map")
def map_(q: str | None = None, k: int = Query(5, ge=1, le=20), max_points: int = Query(1500, ge=10, le=10_000)) -> dict:
    return engine.map(q, k, max_points)


@app.get("/chunks")
def chunks(offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=100)) -> dict:
    return {"total": len(engine), "chunks": engine.chunks(offset, limit)}


@app.get("/stats")
def stats() -> dict:
    out = engine.stats()
    results = config.DATA_DIR / "eval" / "results.json"
    out["eval"] = json.loads(results.read_text()) if results.exists() else None
    return out


@app.post("/reset")
def reset() -> dict:
    engine.reset()
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=config.HOST, port=config.PORT)
