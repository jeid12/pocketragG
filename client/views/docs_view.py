from __future__ import annotations

import flet as ft

from api import DEFAULT_BACKEND, LOCAL_BACKEND, ApiError


def stat_tile(label: str, value: str) -> ft.Container:
    return ft.Container(
        content=ft.Column([ft.Text(value, size=20, weight=ft.FontWeight.BOLD),
                           ft.Text(label, size=12, font_family="DejaVu", color=ft.Colors.ON_SURFACE_VARIANT)], spacing=2),
        padding=12, border_radius=12, bgcolor=ft.Colors.SURFACE_CONTAINER_HIGHEST, expand=True,
    )


class DocsView(ft.Column):
    def __init__(self, app):
        super().__init__(scroll=ft.ScrollMode.AUTO, spacing=14)
        self.app = app
        self.cursor = 0
        default_backend = DEFAULT_BACKEND if not app.page.web else LOCAL_BACKEND
        self.backend = ft.TextField(label="Backend URL", value=app.api.base or default_backend,
                        hint_text=default_backend,
                                    dense=True, expand=True, on_submit=self.on_connect)
        self.status = ft.Text("", size=12)
        self.progress = ft.ProgressBar(visible=False)
        self.stats_row = ft.Row(spacing=8)
        self.docs_list = ft.Column(spacing=4)
        self.chunk_list = ft.Column(spacing=8)
        self.more = ft.TextButton("Load more chunks", on_click=self.on_more, visible=False)
        self.upload_btn = ft.FilledButton("Upload document", icon=ft.Icons.UPLOAD_FILE, on_click=self.on_upload)
        self.controls = [
            ft.Text("Documents", size=24, weight=ft.FontWeight.BOLD),
            ft.Row([self.backend, ft.IconButton(ft.Icons.LINK, tooltip="Connect", on_click=self.on_connect)]),
            self.status,
            ft.Row([self.upload_btn,
                    ft.OutlinedButton("Clear index", icon=ft.Icons.DELETE_OUTLINE, on_click=self.on_reset)],
                   wrap=True),
            ft.Text("PDF, Office, images (OCR), text and markup — up to 500 MB. "
                    "150-word chunks, 30-word overlap.",
                    size=12, color=ft.Colors.ON_SURFACE_VARIANT),
            self.progress,
            self.stats_row,
            self.docs_list,
            ft.Text("Chunks", size=18, weight=ft.FontWeight.W_600),
            self.chunk_list,
            self.more,
        ]

    async def on_connect(self, e=None):
        backend = (self.backend.value or "").strip()
        if not backend:
            self.status.value = "Enter the laptop backend URL, for example http://192.168.1.20:8550"
            self.status.color = ft.Colors.ERROR
            self.update()
            return
        await self.app.set_backend(backend)
        await self.refresh()

    async def refresh(self):
        if not self.app.api.base:
            self.status.value = "Set the backend URL to connect to your laptop backend."
            self.status.color = ft.Colors.ON_SURFACE_VARIANT
            self.stats_row.controls = []
            self.docs_list.controls = []
            self.chunk_list.controls = []
            self.more.visible = False
            self.update()
            return
        try:
            s = await self.app.api.stats()
        except ApiError as err:
            self.status.value = str(err)
            self.status.color = ft.Colors.ERROR
            self.update()
            return
        self.status.value = f"Connected to {self.app.api.base}"
        self.status.color = ft.Colors.GREEN
        self.stats_row.controls = [
            stat_tile("chunks", f"{s['n_chunks']:,}"),
            stat_tile("vocab", f"{s['vocab_size']:,}"),
            stat_tile("avg words", f"{s['avgdl']:.0f}"),
            stat_tile("2D η", f"{s['eta']:.1%}"),
        ]
        self.docs_list.controls = [
            ft.ListTile(
                leading=ft.Icon(ft.Icons.DESCRIPTION),
                title=ft.Text(d["name"]),
                subtitle=ft.Text(f"{d['n_chunks']:,} chunks · {d['bytes'] / 2**20:.1f} MB · indexed in {d['seconds']}s"),
                trailing=ft.IconButton(
                    ft.Icons.DELETE_OUTLINE,
                    tooltip="Delete document",
                    on_click=lambda e, sha=d["sha256"], name=d["name"]: self.on_delete_doc(sha, name),
                ),
                dense=True,
            )
            for d in s["docs"]
        ]
        self.cursor = 0
        self.chunk_list.controls = []
        await self.load_chunks()

    async def load_chunks(self):
        data = await self.app.api.chunks(self.cursor, 10)
        for c in data["chunks"]:
            page = f" · page {c['page']}" if c.get("page") else ""
            self.chunk_list.controls.append(ft.Card(content=ft.Container(padding=12, content=ft.Column([
                ft.Text(f"#{c['id']} · {c['doc']}{page} · words {c['start_word']}–{c['end_word']}",
                        size=12, color=ft.Colors.PRIMARY),
                ft.Text(c["text"], size=13, max_lines=4, overflow=ft.TextOverflow.ELLIPSIS),
            ], spacing=4))))
        self.cursor += len(data["chunks"])
        self.more.visible = self.cursor < data["total"]
        self.update()

    async def on_more(self, e):
        await self.load_chunks()

    async def on_upload(self, e):
        files = await self.app.file_picker.pick_files(
            dialog_title="Choose a document", file_type=ft.FilePickerFileType.ANY,
            with_data=self.app.page.web)
        if not files:
            return
        f = files[0]
        if f.size and f.size > 524_288_000:
            self.app.toast("File is larger than 500 MB")
            return
        self.progress.visible = True
        self.upload_btn.disabled = True
        self.status.value = f"Indexing {f.name}… (embedding runs at about 25 chunks/s)"
        self.status.color = None
        self.update()
        try:
            doc = await self.app.api.ingest(f.name, path=f.path, data=f.bytes)
            capped = " (chunk cap reached)" if doc.get("capped") else ""
            self.app.toast(f"Indexed {doc['n_chunks']:,} chunks in {doc['seconds']}s{capped}")
        except ApiError as err:
            self.app.toast(str(err))
        finally:
            self.progress.visible = False
            self.upload_btn.disabled = False
            self.update()
        await self.refresh()

    async def on_reset(self, e):
        try:
            await self.app.api.reset()
            self.app.toast("Index cleared")
        except ApiError as err:
            self.app.toast(str(err))
        await self.refresh()

    async def on_delete_doc(self, sha256: str, name: str):
        try:
            await self.app.api.delete_doc(sha256)
            self.app.toast(f"Deleted {name}")
        except ApiError as err:
            self.app.toast(str(err))
        await self.refresh()
