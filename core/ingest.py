from __future__ import annotations

import codecs
import hashlib
import os
import re
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from core import config

_WORD = re.compile(r"\S+")
_TEXT_BLOCK_CHARS = 1 << 20
_MAX_TOKEN_CHARS = 1 << 16


class IngestError(Exception):
    status = 400


class PayloadTooLarge(IngestError):
    status = 413


class UnsupportedMedia(IngestError):
    status = 415


class InvalidContent(IngestError):
    status = 422


@dataclass(slots=True, frozen=True)
class Chunk:
    id: int
    text: str
    start_word: int
    end_word: int
    page: int | None


def check_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_EXT:
        raise UnsupportedMedia(f"extension {ext or '(none)'} not in {sorted(config.ALLOWED_EXT)}")
    return ext


def check_declared_size(size: int | None, max_bytes: int = config.MAX_UPLOAD_BYTES) -> None:
    if size is not None and size > max_bytes:
        raise PayloadTooLarge(f"declared size {size} > {max_bytes} bytes")


def inspect_file(path: str | os.PathLike, ext: str) -> str | None:
    with open(path, "rb") as f:
        if ext == ".pdf":
            if f.read(5) != b"%PDF-":
                raise UnsupportedMedia("missing %PDF- header")
            return None

        head = f.read(3)
        encoding = "utf-8-sig" if head == codecs.BOM_UTF8 else "utf-8"
        decoder = codecs.getincrementaldecoder("utf-8")()
        f.seek(0)
        while block := f.read(config.READ_BLOCK_BYTES):
            if b"\x00" in block:
                raise InvalidContent("null byte found: binary data in a text file")
            if encoding != "latin-1":
                try:
                    decoder.decode(block)
                except UnicodeDecodeError:
                    encoding = "latin-1"
        if encoding != "latin-1":
            try:
                decoder.decode(b"", final=True)
            except UnicodeDecodeError:
                encoding = "latin-1"
        return encoding


class SpooledUpload:

    def __init__(
        self,
        stream: BinaryIO,
        filename: str,
        declared_size: int | None = None,
        *,
        max_bytes: int = config.MAX_UPLOAD_BYTES,
        dir: str | os.PathLike | None = None,
    ):
        self.stream = stream
        self.filename = filename
        self.declared_size = declared_size
        self.max_bytes = max_bytes
        self.dir = dir
        self.path: Path | None = None
        self.size = 0
        self.sha256 = ""
        self.ext = ""
        self.encoding: str | None = None

    def __enter__(self) -> SpooledUpload:
        self.ext = check_extension(self.filename)
        check_declared_size(self.declared_size, self.max_bytes)
        fd, name = tempfile.mkstemp(prefix="pocketrag-", suffix=self.ext, dir=self.dir)
        self.path = Path(name)
        try:
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as out:
                while block := self.stream.read(config.READ_BLOCK_BYTES):
                    self.size += len(block)
                    if self.size > self.max_bytes:
                        raise PayloadTooLarge(f"stream exceeded {self.max_bytes} bytes")
                    digest.update(block)
                    out.write(block)
            if self.size == 0:
                raise InvalidContent("empty file")
            self.sha256 = digest.hexdigest()
            self.encoding = inspect_file(self.path, self.ext)
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(self, *exc) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self.path is not None:
            self.path.unlink(missing_ok=True)

    def words(self) -> Iterator[tuple[str, int | None]]:
        return iter_words(self.path, self.ext, self.encoding)

    def chunks(self, **kw) -> Iterator[Chunk]:
        return iter_chunks(self.words(), **kw)


def _iter_text_words(path: str | os.PathLike, encoding: str) -> Iterator[tuple[str, None]]:
    tail = ""
    with open(path, encoding=encoding, errors="replace") as f:
        while block := f.read(_TEXT_BLOCK_CHARS):
            block = tail + block
            tail = ""
            for m in _WORD.finditer(block):
                if m.end() == len(block) and len(block) - m.start() < _MAX_TOKEN_CHARS:
                    tail = m.group()
                else:
                    yield m.group(), None
    if tail:
        yield tail, None


def _iter_pdf_words(path: str | os.PathLike) -> Iterator[tuple[str, int]]:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(path)
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for m in _WORD.finditer(text):
                yield m.group(), page_no
    except (PyPdfError, ValueError, KeyError, TypeError) as e:
        raise InvalidContent(f"unreadable PDF: {type(e).__name__}") from e


def iter_words(path: str | os.PathLike, ext: str, encoding: str | None = None) -> Iterator[tuple[str, int | None]]:
    if ext == ".pdf":
        return _iter_pdf_words(path)
    return _iter_text_words(path, encoding or "utf-8")


def iter_chunks(
    words: Iterable[tuple[str, int | None]],
    size: int = config.CHUNK_WORDS,
    overlap: int = config.CHUNK_OVERLAP,
    max_chunks: int = config.MAX_CHUNKS,
) -> Iterator[Chunk]:
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    stride = size - overlap
    window: deque[tuple[str, int | None]] = deque()
    start = fresh = cid = 0

    def emit() -> Chunk:
        return Chunk(cid, " ".join(w for w, _ in window), start, start + len(window), window[0][1])

    for item in words:
        window.append(item)
        fresh += 1
        if len(window) == size:
            yield emit()
            cid += 1
            if cid >= max_chunks:
                return
            for _ in range(stride):
                window.popleft()
            start += stride
            fresh = 0
    if fresh:
        yield emit()


def chunk_file(path: str | os.PathLike, **kw) -> Iterator[Chunk]:
    path = Path(path)
    ext = check_extension(path.name)
    check_declared_size(path.stat().st_size)
    encoding = inspect_file(path, ext)
    return iter_chunks(iter_words(path, ext, encoding), **kw)


def _self_check() -> None:
    import io
    import time
    import tracemalloc

    class Zeros(io.RawIOBase):
        def __init__(self, n):
            self.left = n

        def read(self, n=-1):
            n = self.left if n < 0 else min(n, self.left)
            self.left -= n
            return b"a" * n

    with tempfile.TemporaryDirectory() as tmp:
        def expect(exc, fn):
            try:
                fn()
            except exc as e:
                print(f"  ok  {exc.__name__} ({e.status}): {e}")
                return
            raise AssertionError(f"expected {exc.__name__}")

        def spool(stream, name, declared=None, **kw):
            with SpooledUpload(stream, name, declared, dir=tmp, **kw):
                pass

        print("gates")
        expect(PayloadTooLarge, lambda: spool(Zeros(0), "big.pdf", 600 * 2**20))
        expect(PayloadTooLarge, lambda: spool(Zeros(3 * 2**20), "big.txt", max_bytes=2**20))
        expect(UnsupportedMedia, lambda: spool(io.BytesIO(b"MZ\x90\x00"), "setup.exe"))
        expect(UnsupportedMedia, lambda: spool(io.BytesIO(b"MZ\x90\x00\x03"), "renamed.pdf"))
        expect(InvalidContent, lambda: spool(io.BytesIO(b"hello\x00world"), "notes.txt"))
        assert not os.listdir(tmp), "temp files leaked after aborted uploads"
        print("  ok  no temp files left behind")

        print("chunker")
        n_words = 1000
        words = [(f"w{i}", None) for i in range(n_words)]
        chunks = list(iter_chunks(words))
        for a, b in zip(chunks, chunks[1:]):
            assert b.start_word - a.start_word == 120
            assert a.end_word - b.start_word == 30
            assert a.text.split()[-30:] == b.text.split()[:30]
        assert chunks[-1].end_word == n_words and all(len(c.text.split()) <= 150 for c in chunks)
        print(f"  ok  {len(chunks)} chunks, stride 120, overlap 30, tail covered")

        print("streaming memory")
        src = Path(tmp) / "big.txt"
        line = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do " * 8 + "\n").encode()
        with open(src, "wb") as f:
            for _ in range(20 * 2**20 // len(line)):
                f.write(line)
        t = time.perf_counter()
        n = sum(1 for _ in chunk_file(src, max_chunks=10**9))
        elapsed = time.perf_counter() - t
        tracemalloc.start()
        sum(1 for _ in chunk_file(src, max_chunks=10**9))
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        bound = 2 * config.READ_BLOCK_BYTES + 4 * 2**20
        print(f"  ok  {src.stat().st_size / 2**20:.0f} MB -> {n} chunks in {elapsed:.1f}s, "
              f"tracemalloc peak {peak / 2**20:.1f} MB (bound {bound / 2**20:.0f} MB, independent of file size)")
        assert peak < bound
    print("core.ingest: all checks passed")


if __name__ == "__main__":
    _self_check()
