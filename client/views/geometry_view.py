from __future__ import annotations

import math

import flet as ft
import flet.canvas as cv

from api import ApiError

HEIGHT = 420
PAD = 24


def star(cx: float, cy: float, r: float) -> list:
    pts = []
    for i in range(10):
        a = -math.pi / 2 + i * math.pi / 5
        rad = r if i % 2 == 0 else r * 0.45
        pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    elements = [cv.Path.MoveTo(*pts[0])] + [cv.Path.LineTo(x, y) for x, y in pts[1:]] + [cv.Path.Close()]
    return elements


class GeometryView(ft.Column):
    def __init__(self, app):
        super().__init__(scroll=ft.ScrollMode.AUTO, spacing=12)
        self.app = app
        self.map_data: dict | None = None
        self.width_px = app.content_width()
        self.screen: list[tuple[int, float, float]] = []
        self.query = ft.TextField(hint_text="Project a query onto the map…", dense=True, expand=True,
                                  on_submit=self.on_project)
        self.caption = ft.Text("", size=12, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT)
        self.canvas = cv.Canvas(shapes=[], width=self.width_px, height=HEIGHT, on_resize=self.on_resize)
        self.detail = ft.Container()
        self.controls = [
            ft.Text("Semantic map", size=24, weight=ft.FontWeight.BOLD),
            ft.Text("Every chunk's 384-d embedding projected onto the top two right singular vectors "
                    "(X₂D = D V₂). Orange points are the retrieved passages; the star is the query (q̂ V₂).",
                    size=12, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT),
            ft.Row([self.query, ft.IconButton(ft.Icons.SEARCH, on_click=self.on_project)]),
            ft.Container(
                content=ft.GestureDetector(content=self.canvas, on_tap_down=self.on_tap),
                height=HEIGHT, border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST,
            ),
            self.caption,
            self.detail,
        ]

    async def refresh(self):
        await self.load(self.query.value or None)

    async def load(self, q: str | None):
        try:
            self.map_data = await self.app.api.map(q)
        except ApiError as err:
            self.caption.value = str(err)
            self.update()
            return
        self.draw()

    async def project(self, q: str):
        self.query.value = q
        await self.load(q)

    async def on_project(self, e):
        await self.load((self.query.value or "").strip() or None)

    def on_resize(self, e: cv.CanvasResizeEvent):
        self.width_px = e.width
        self.draw()

    def draw(self):
        d = self.map_data
        self.width_px = self.app.content_width()
        self.canvas.width = self.width_px
        if not d or not d["points"]:
            self.canvas.shapes = [cv.Text(PAD, PAD, "No documents indexed yet.")]
            self.caption.value = ""
            self.update()
            return
        xs = [p["x"] for p in d["points"]] + ([d["query"]["x"]] if d["query"] else [])
        ys = [p["y"] for p in d["points"]] + ([d["query"]["y"]] if d["query"] else [])
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        w, h = max(self.width_px, 100), HEIGHT
        sx = (w - 2 * PAD) / ((x1 - x0) or 1)
        sy = (h - 2 * PAD) / ((y1 - y0) or 1)

        def to_screen(x, y):
            return PAD + (x - x0) * sx, h - PAD - (y - y0) * sy

        hits = set(d["hits"])
        muted = ft.Paint(color=ft.Colors.with_opacity(0.35, ft.Colors.BLUE_GREY_400))
        hot = ft.Paint(color=ft.Colors.ORANGE_700)
        shapes, top = [], []
        self.screen = []
        for p in d["points"]:
            X, Y = to_screen(p["x"], p["y"])
            self.screen.append((p["id"], X, Y))
            if p["id"] in hits:
                top.append(cv.Circle(X, Y, 7, hot))
            else:
                shapes.append(cv.Circle(X, Y, 2.5, muted))
        shapes += top
        if d["query"]:
            qx, qy = to_screen(d["query"]["x"], d["query"]["y"])
            shapes.append(cv.Path(star(qx, qy, 12), paint=ft.Paint(color=ft.Colors.RED_600)))
            shapes.append(cv.Text(qx + 14, qy - 8, "query", style=ft.TextStyle(size=11, color=ft.Colors.RED_600)))
        shapes.append(cv.Text(w - PAD - 60, h - 18, "σ₁ axis →",
                              style=ft.TextStyle(size=10, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT)))
        shapes.append(cv.Text(6, 6, "↑ σ₂ axis", style=ft.TextStyle(size=10, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT)))
        self.canvas.shapes = shapes
        shown = len(d["points"])
        sample = f"{shown:,} of {d['n_total']:,} chunks (sampled)" if shown < d["n_total"] else f"{shown:,} chunks"
        self.caption.value = (f"{sample} · explained energy η = (σ₁² + σ₂²) / Σσⱼ² = {d['eta']:.1%} · "
                              "tap a point to read it")
        self.update()

    async def on_tap(self, e: ft.TapEvent):
        if not self.screen or e.local_position is None:
            return
        tx, ty = e.local_position.x, e.local_position.y
        cid, X, Y = min(self.screen, key=lambda s: (s[1] - tx) ** 2 + (s[2] - ty) ** 2)
        if (X - tx) ** 2 + (Y - ty) ** 2 > 20 ** 2:
            return
        try:
            c = (await self.app.api.chunks(cid, 1))["chunks"][0]
        except (ApiError, IndexError):
            return
        page = f" · page {c['page']}" if c.get("page") else ""
        self.detail.content = ft.Card(content=ft.Container(padding=12, content=ft.Column([
            ft.Text(f"chunk {cid} · {c['doc']}{page}", size=12, color=ft.Colors.PRIMARY),
            ft.Text(c["text"], size=13, selectable=True),
        ], spacing=4)))
        self.update()
