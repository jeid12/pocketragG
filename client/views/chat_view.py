from __future__ import annotations

import flet as ft

from api import ApiError


def rank_text(r):
    return f"#{r}" if r else "–"


def passage_card(p: dict) -> ft.Container:
    page = f" · p.{p['page']}" if p.get("page") else ""
    return ft.Container(
        padding=10, border_radius=10, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
        content=ft.Column([
            ft.Text(f"[{p['rank']}] chunk {p['chunk_id']} · {p['doc']}{page}",
                    size=12, weight=ft.FontWeight.W_600, color=ft.Colors.PRIMARY),
            ft.Text(f"RRF {p['rrf']:.4f}   cosine {p['dense']:.3f} ({rank_text(p['dense_rank'])})   "
                    f"BM25 {p['bm25']:.2f} ({rank_text(p['bm25_rank'])})",
                    size=11, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Text(p["text"], size=13, selectable=True),
        ], spacing=4),
    )


class ChatView(ft.Column):
    def __init__(self, app):
        super().__init__(spacing=10)
        self.app = app
        self.feed = ft.ListView(expand=True, spacing=12, auto_scroll=True)
        self.input = ft.TextField(hint_text="Ask about your documents…", expand=True, dense=True,
                                  on_submit=self.on_send, shift_enter=True, min_lines=1, max_lines=4)
        self.send_btn = ft.IconButton(ft.Icons.SEND, on_click=self.on_send, tooltip="Ask")
        self.controls = [
            ft.Text("Ask", size=24, weight=ft.FontWeight.BOLD),
            self.feed,
            ft.Row([self.input, self.send_btn]),
        ]
        self.feed.controls.append(ft.Text(
            "Answers are built only from retrieved passages. Each passage shows its RRF score and its rank "
            "in the dense (cosine) and lexical (BM25) retrievers.", size=12, color=ft.Colors.ON_SURFACE_VARIANT))

    async def refresh(self):
        pass

    async def on_send(self, e):
        q = (self.input.value or "").strip()
        if q:
            self.input.value = ""
            await self.ask(q)

    async def ask(self, q: str):
        self.send_btn.disabled = True
        self.feed.controls.append(ft.Container(
            content=ft.Text(q, color=ft.Colors.ON_PRIMARY_CONTAINER), padding=12, border_radius=14,
            bgcolor=ft.Colors.PRIMARY_CONTAINER, margin=ft.Margin.only(left=60)))
        thinking = ft.Row([ft.ProgressRing(width=16, height=16, stroke_width=2), ft.Text("Retrieving…", size=12)])
        self.feed.controls.append(thinking)
        self.update()
        try:
            r = await self.app.api.query(q)
            self.feed.controls[-1] = self.answer_card(q, r)
            self.app.add_history(q, r)
        except ApiError as err:
            self.feed.controls[-1] = ft.Text(str(err), color=ft.Colors.ERROR)
        self.send_btn.disabled = False
        self.update()

    def answer_card(self, q: str, r: dict) -> ft.Control:
        if not r["passages"]:
            return ft.Text("No documents indexed yet. Upload one in the Docs tab.", color=ft.Colors.ERROR)
        if r["mode"] == "gemini":
            badge, color = f"Gemini · {r.get('model', '')}", ft.Colors.GREEN
        elif r["mode"] == "claude":
            badge, color = f"Claude · {r.get('model', '')}", ft.Colors.GREEN
        else:
            badge, color = "Extractive", ft.Colors.AMBER
        meta = [ft.Container(ft.Text(badge, size=11, color=ft.Colors.BLACK), bgcolor=color,
                             padding=ft.Padding.symmetric(horizontal=8, vertical=2), border_radius=8),
                ft.Text(f"retrieval {r['retrieval_ms']:.0f} ms · total {r['total_ms']:.0f} ms", size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT)]
        body = [ft.Markdown(r["answer"], selectable=True, extension_set=ft.MarkdownExtensionSet.GITHUB_WEB),
                ft.Row(meta, wrap=True)]
        if r.get("reason"):
            body.append(ft.Text(r["reason"], size=11, italic=True, color=ft.Colors.ON_SURFACE_VARIANT))
        body.append(ft.ExpansionTile(
            title=ft.Text(f"{len(r['passages'])} verified source passages", size=13),
            controls=[passage_card(p) for p in r["passages"]], controls_padding=8))
        return ft.Card(content=ft.Container(padding=14, content=ft.Column(body, spacing=8)))
