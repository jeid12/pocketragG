from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import flet as ft

from api import DEFAULT_BACKEND, Api
from views.chat_view import ChatView
from views.docs_view import DocsView
from views.geometry_view import GeometryView
from views.theory_view import TheoryView

MAX_WIDTH = 750
PADDING = 16
PREF_KEY = "pocketrag.backend"
ROUTES = ["/", "/ask", "/map", "/theory"]
MATH_FONT = "DejaVu"


class PocketRAG:
    def __init__(self, page: ft.Page):
        self.page = page
        self.api = Api(DEFAULT_BACKEND)
        self.prefs = ft.SharedPreferences()
        self.file_picker = ft.FilePicker()
        self.views = [DocsView(self), ChatView(self), GeometryView(self), TheoryView(self)]
        self.shell = ft.Container(content=self.views[0], width=self.shell_width(), padding=PADDING)
        self.nav = ft.NavigationBar(
            selected_index=0, on_change=self.on_nav,
            destinations=[
                ft.NavigationBarDestination(icon=ft.Icons.DESCRIPTION_OUTLINED, selected_icon=ft.Icons.DESCRIPTION,
                                            label="Docs"),
                ft.NavigationBarDestination(icon=ft.Icons.CHAT_OUTLINED, selected_icon=ft.Icons.CHAT, label="Ask"),
                ft.NavigationBarDestination(icon=ft.Icons.SCATTER_PLOT_OUTLINED, selected_icon=ft.Icons.SCATTER_PLOT,
                                            label="Map"),
                ft.NavigationBarDestination(icon=ft.Icons.FUNCTIONS, label="Theory"),
            ],
        )

    def shell_width(self) -> float:
        return min(MAX_WIDTH, self.page.width or MAX_WIDTH)

    def content_width(self) -> float:
        return self.shell_width() - 2 * PADDING

    async def start(self):
        page = self.page
        page.title = "PocketRAG"
        page.theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
        page.dark_theme = ft.Theme(color_scheme_seed=ft.Colors.INDIGO)
        page.padding = 0
        page.fonts = {MATH_FONT: "fonts/DejaVuSans.ttf"}
        page.navigation_bar = self.nav
        page.on_resize = self.on_resize
        page.add(ft.SafeArea(ft.Row([self.shell], alignment=ft.MainAxisAlignment.CENTER,
                                    vertical_alignment=ft.CrossAxisAlignment.STRETCH), expand=True))
        try:
            saved = await self.prefs.get(PREF_KEY)
            if saved:
                self.api.set_base(saved)
                self.views[0].backend.value = saved
        except Exception:
            pass
        url = urlsplit(page.route or "/")
        path = url.path.rstrip("/") or "/"
        await self.select(ROUTES.index(path) if path in ROUTES else 0)
        q = parse_qs(url.query).get("q", [""])[0].strip()
        if q and path == "/ask":
            await self.views[1].ask(q)
        elif q and path == "/map":
            await self.views[2].project(q)

    async def set_backend(self, url: str):
        self.api.set_base(url)
        try:
            await self.prefs.set(PREF_KEY, self.api.base)
        except Exception:
            pass

    def on_resize(self, e):
        self.shell.width = self.shell_width()
        if isinstance(self.shell.content, GeometryView):
            self.shell.content.draw()
        self.page.update()

    async def select(self, index: int):
        self.nav.selected_index = index
        self.shell.content = self.views[index]
        self.page.update()
        await self.views[index].refresh()

    async def on_nav(self, e):
        await self.select(e.control.selected_index)

    async def show_on_map(self, q: str):
        self.nav.selected_index = 2
        self.shell.content = self.views[2]
        self.page.update()
        await self.views[2].project(q)

    def toast(self, message: str):
        self.page.show_dialog(ft.SnackBar(ft.Text(message)))


async def main(page: ft.Page):
    await PocketRAG(page).start()


if __name__ == "__main__":
    ft.run(main)
