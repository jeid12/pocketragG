import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("POCKETRAG_DATA", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

MAX_UPLOAD_BYTES = 524_288_000
READ_BLOCK_BYTES = 8 * 1024 * 1024
ALLOWED_EXT = frozenset({
    ".pdf",
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".log",
    ".yaml", ".yml", ".ini", ".tex", ".rst",
    ".html", ".htm", ".rtf",
    ".docx", ".odt", ".pptx", ".xlsx", ".epub",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
})

TEXT_EXT = frozenset({
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml", ".log",
    ".yaml", ".yml", ".ini", ".tex", ".rst",
})

IMAGE_EXT = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"})

ZIP_DOC_EXT = frozenset({".docx", ".pptx", ".xlsx", ".odt", ".epub"})
CHUNK_WORDS = 150
CHUNK_OVERLAP = 30
MAX_CHUNKS = int(os.getenv("POCKETRAG_MAX_CHUNKS", 20_000))

EMBED_MODEL = os.getenv("POCKETRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
EMBED_BATCH = int(os.getenv("POCKETRAG_EMBED_BATCH", 4))
MODEL_CACHE = Path(os.getenv("FASTEMBED_CACHE_PATH", ROOT / ".models"))
EPS = 1e-12
SCORE_BLOCK_ROWS = 8192

BM25_K1 = 1.5
BM25_B = 0.75

RRF_K = 60

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta")

HOST = os.getenv("POCKETRAG_HOST", "0.0.0.0")
PORT = int(os.getenv("POCKETRAG_PORT", 8550))
