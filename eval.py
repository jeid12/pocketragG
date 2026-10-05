from __future__ import annotations

import json
import re
import statistics
import sys
import time
import tracemalloc

from core import config
from core.engine import Engine, peak_rss_mb
from core.ingest import chunk_file
from core.metrics import mean_over_queries

K = 10
MODES = ("dense", "bm25", "hybrid")
QRELS = config.DATA_DIR / "eval" / "qrels.json"
RESULTS = config.DATA_DIR / "eval" / "results.json"
INDEX = config.DATA_DIR / "eval" / "index"


def build_qrels(engine: Engine, spec: dict) -> list[dict]:
    lo, hi = spec["body_chunks"]
    texts = {i: engine.chunk(i)["text"] for i in range(lo, min(hi, len(engine)))}
    out = []
    for item in spec["queries"]:
        rel = {}
        for grade, key in ((1, "secondary"), (2, "primary")):
            for phrase in item.get(key, []):
                pattern = re.compile(re.escape(phrase), re.I)
                for i, t in texts.items():
                    if pattern.search(t):
                        rel[i] = grade
        out.append({"q": item["q"], "qrels": rel})
    return out


def main() -> int:
    spec = json.loads(QRELS.read_text())
    engine = Engine(INDEX)
    corpus = config.ROOT / spec["corpus"]
    if not len(engine):
        print(f"indexing {corpus.name} (one-off, about 1 min)…")
        doc = engine.ingest(corpus.name, chunk_file(corpus))
        print(f"  {doc['n_chunks']} chunks in {doc['seconds']}s")
    labelled = build_qrels(engine, spec)
    empty = [x["q"] for x in labelled if not x["qrels"]]
    if empty:
        print("queries with no relevant chunk:", empty)
        return 1

    engine.embedder.embed_query("warm up")
    tracemalloc.start()
    runs: dict[str, list] = {m: [] for m in MODES}
    latency: dict[str, list] = {m: [] for m in MODES}
    per_query = []
    for x in labelled:
        row = {"q": x["q"], "n_relevant": len(x["qrels"])}
        for m in MODES:
            t = time.perf_counter()
            ranked = engine.ranking(x["q"], m, K)
            latency[m].append((time.perf_counter() - t) * 1e3)
            runs[m].append((ranked, x["qrels"]))
            row[m] = next((i + 1 for i, d in enumerate(ranked) if x["qrels"].get(d, 0) > 0), None)
        per_query.append(row)
    query_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    results = {m: mean_over_queries(runs[m], K) for m in MODES}
    report = {
        "corpus": spec["source"],
        "n_chunks": len(engine),
        "n_queries": len(labelled),
        "k": K,
        "results": results,
        "latency_ms": {m: round(statistics.median(latency[m]), 2) for m in MODES},
        "memory": {"peak_rss_mb": round(peak_rss_mb(), 1),
                   "query_tracemalloc_peak_mb": round(query_peak / 2**20, 2),
                   "bm25_index_mb": round(engine.bm25.nbytes() / 2**20, 2)},
        "per_query": per_query,
    }
    RESULTS.write_text(json.dumps(report, indent=2))

    print(f"\n{spec['source']}: {len(engine)} chunks, {len(labelled)} labelled queries\n")
    print(f"{'retriever':<10}{'MRR@' + str(K):>10}{'NDCG@' + str(K):>10}{'median ms':>12}")
    for m in MODES:
        print(f"{m:<10}{results[m]['mrr']:>10.3f}{results[m]['ndcg']:>10.3f}{report['latency_ms'][m]:>12.1f}")
    print(f"\nfirst relevant rank per query (dense / bm25 / hybrid):")
    for row in per_query:
        ranks = " / ".join(str(row[m] or "–") for m in MODES)
        print(f"  {ranks:<14} {row['q']}")
    mem = report["memory"]
    print(f"\npeak RSS {mem['peak_rss_mb']} MB · tracemalloc peak while querying {mem['query_tracemalloc_peak_mb']} MB"
          f" · BM25 index {mem['bm25_index_mb']} MB")
    print(f"written to {RESULTS.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
