from __future__ import annotations

import flet as ft


class HistoryView(ft.Column):
    def __init__(self, app):
        super().__init__(scroll=ft.ScrollMode.AUTO, spacing=12)
        self.app = app
        self.clear_btn = ft.OutlinedButton("Clear history", icon=ft.Icons.DELETE_OUTLINE,
                                           on_click=self.on_clear)
        self.feed = ft.Column(spacing=10)
        self.controls = [
            ft.Text("History", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Recent questions and answers from this session.", size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            self.clear_btn,
            self.feed,
        ]

    async def refresh(self):
        entries = list(reversed(self.app.history))
        if not entries:
            self.feed.controls = [ft.Text("No history yet. Ask a question to populate this tab.",
                                          color=ft.Colors.ON_SURFACE_VARIANT)]
            try:
                self.update()
            except RuntimeError:
                pass
            return
        cards = []
        for item in entries:
            cards.append(ft.Card(content=ft.Container(padding=14, content=ft.Column([
                ft.Text(item["question"], size=14, weight=ft.FontWeight.W_600),
                ft.Text(item["answer"], size=13, selectable=True, max_lines=5, overflow=ft.TextOverflow.ELLIPSIS),
                ft.Text(f"{item['mode'].title()} · {item['timestamp']}", size=11,
                        color=ft.Colors.ON_SURFACE_VARIANT),
            ], spacing=6))))
        self.feed.controls = cards
        try:
            self.update()
        except RuntimeError:
            pass

    async def on_clear(self, e):
        self.app.history.clear()
        await self.refresh()