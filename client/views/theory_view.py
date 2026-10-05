from __future__ import annotations

import flet as ft

from api import ApiError

SECTIONS = [
    ("Unit hypersphere (dense retrieval)",
     "d̂ᵢ = dᵢ / max(‖dᵢ‖₂, ε),  ε = 10⁻¹²\ns = D q̂ ∈ [−1, 1]ᴺ\n"
     "Top-k by argpartition: O(N + k log k) instead of O(N log N)",
     "Each 150-word chunk is embedded into ℝ³⁸⁴ and normalised, so one matrix–vector product gives every "
     "cosine similarity at once. The matrix lives on disk and is scored in blocks."),
    ("Okapi BM25 (lexical retrieval)",
     "IDF(q) = ln((N − n(q) + 0.5) / (n(q) + 0.5) + 1)\n"
     "score = Σ IDF(q) · f·(k₁+1) / (f + k₁(1 − b + b·|D|/avgdl))\nk₁ = 1.5,  b = 0.75",
     "Catches exact identifiers and rare terms that embeddings blur. Term frequency saturates through k₁; "
     "b penalises long chunks."),
    ("Reciprocal Rank Fusion",
     "RRF(d) = Σₘ 1 / (k + πₘ(d)),  k = 60",
     "Cosine scores and BM25 scores live on different scales, so only their ranks are combined. "
     "A chunk ranked well by both retrievers beats one ranked first by only one."),
    ("Truncated SVD (the map)",
     "D = U Σ Vᵀ,  X₂D = D V₂,  q₂D = q̂ V₂\nη = (σ₁² + σ₂²) / Σ σⱼ²",
     "By Eckart–Young–Mirsky, V₂ gives the best rank-2 approximation in Frobenius norm. V₂ is computed "
     "from the 384×384 Gram matrix DᵀD, so memory does not grow with N."),
    ("Evaluation",
     "MRR = (1/|Q|) Σ 1 / rankᵢ\nDCG@K = Σ (2^relᵢ − 1) / log₂(i + 1),  NDCG = DCG / IDCG",
     "Run python eval.py on the backend to fill the table below."),
]


def formula_card(title: str, formula: str, note: str) -> ft.Card:
    return ft.Card(content=ft.Container(padding=14, content=ft.Column([
        ft.Text(title, size=16, weight=ft.FontWeight.W_600),
        ft.Container(ft.Text(formula, font_family="DejaVu", size=13, selectable=True), padding=10,
                     border_radius=8, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST),
        ft.Text(note, size=13, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT),
    ], spacing=8)))


class TheoryView(ft.Column):
    def __init__(self, app):
        super().__init__(scroll=ft.ScrollMode.AUTO, spacing=12)
        self.app = app
        self.live = ft.Column(spacing=8)
        self.controls = [ft.Text("Theory", size=24, weight=ft.FontWeight.BOLD)] + \
            [formula_card(*s) for s in SECTIONS] + [ft.Text("Live measurements", size=18,
                                                            weight=ft.FontWeight.W_600), self.live]

    async def refresh(self):
        try:
            s = await self.app.api.stats()
        except ApiError as err:
            self.live.controls = [ft.Text(str(err), color=ft.Colors.ERROR)]
            self.update()
            return
        rows = [
            ("chunks indexed", f"{s['n_chunks']:,} (cap {s['max_chunks']:,})"),
            ("BM25 index in RAM", f"{s['bm25_mb']} MB"),
            ("dense matrix on disk", f"{s['dense_mb_on_disk']} MB"),
            ("backend peak RSS", f"{s['peak_rss_mb']} MB (budget 350 MB)"),
            ("explained energy of the 2D map", f"{s['eta']:.2%}"),
        ]
        controls = [ft.Row([ft.Text(k, expand=True, size=13), ft.Text(v, size=13, weight=ft.FontWeight.W_600)])
                    for k, v in rows]
        ev = s.get("eval")
        if ev:
            table = ft.DataTable(
                columns=[ft.DataColumn(ft.Text("retriever")), ft.DataColumn(ft.Text(f"MRR@{ev['k']}")),
                         ft.DataColumn(ft.Text(f"NDCG@{ev['k']}"))],
                rows=[ft.DataRow(cells=[ft.DataCell(ft.Text(m)), ft.DataCell(ft.Text(f"{v['mrr']:.3f}")),
                                        ft.DataCell(ft.Text(f"{v['ndcg']:.3f}"))])
                      for m, v in ev["results"].items()],
            )
            controls += [ft.Text(f"{ev['n_queries']} labelled queries on {ev['corpus']}", size=12,
                                 color=ft.Colors.ON_SURFACE_VARIANT), table]
        else:
            controls.append(ft.Text("No evaluation results yet: run python eval.py.", size=12,
                                    color=ft.Colors.ON_SURFACE_VARIANT))
        self.live.controls = controls
        self.update()
