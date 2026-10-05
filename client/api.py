from __future__ import annotations

import asyncio
import os

import requests

DEFAULT_BACKEND = os.getenv("POCKETRAG_BACKEND", "http://127.0.0.1:8550")


class ApiError(Exception):
    pass


class Api:
    def __init__(self, base: str = DEFAULT_BACKEND):
        self.base = base.rstrip("/")

    def set_base(self, base: str) -> None:
        self.base = base.strip().rstrip("/")

    def _call(self, method: str, path: str, timeout: float = 30, **kw):
        try:
            r = requests.request(method, f"{self.base}{path}", timeout=timeout, **kw)
        except requests.RequestException as e:
            raise ApiError(f"cannot reach backend at {self.base} ({type(e).__name__})") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise ApiError(f"{r.status_code}: {detail}")
        return r.json()

    async def call(self, method: str, path: str, **kw):
        return await asyncio.to_thread(self._call, method, path, **kw)

    async def health(self):
        return await self.call("GET", "/health", timeout=5)

    async def stats(self):
        return await self.call("GET", "/stats")

    async def chunks(self, offset: int = 0, limit: int = 20):
        return await self.call("GET", "/chunks", params={"offset": offset, "limit": limit})

    async def query(self, q: str, k: int = 5):
        return await self.call("POST", "/query", json={"q": q, "k": k}, timeout=120)

    async def map(self, q: str | None = None, k: int = 5):
        params = {"k": k}
        if q:
            params["q"] = q
        return await self.call("GET", "/map", params=params, timeout=60)

    async def reset(self):
        return await self.call("POST", "/reset")

    async def ingest(self, name: str, path: str | None = None, data: bytes | None = None):
        def send():
            if path:
                with open(path, "rb") as f:
                    return self._call("POST", "/ingest", files={"file": (name, f)}, timeout=3600)
            return self._call("POST", "/ingest", files={"file": (name, data)}, timeout=3600)

        return await asyncio.to_thread(send)
