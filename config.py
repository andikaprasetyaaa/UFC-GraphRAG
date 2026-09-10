"""
Konfigurasi terpusat untuk sistem GraphRAG Hybrid Retrieval.
Semua path penyimpanan, nama model, dan hyperparameter retrieval
diatur di sini supaya module lain tinggal `import config`.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # baca file .env di root project (jika ada)

# ---------------------------------------------------------------------------
# Path dasar
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = Path(os.getenv("CSV_PATH", DATA_DIR / "master.csv"))

# ---------------------------------------------------------------------------
# Kredensial Gemini untuk chat generation dan model embedding lokal
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")

# ---------------------------------------------------------------------------
# Lokasi index/artifact hasil build_index.py
# ---------------------------------------------------------------------------
FAISS_INDEX_DIR = STORAGE_DIR / "faiss_index"
BM25_INDEX_PATH = STORAGE_DIR / "bm25_retriever.pkl"
GRAPH_PATH = STORAGE_DIR / "knowledge_graph.pkl"
DOCS_PATH = STORAGE_DIR / "documents.pkl"

# ---------------------------------------------------------------------------
# Lapisan cache ("canggih" / multi-tier)
#   L1 - Semantic query cache : lewati retrieval + LLM sepenuhnya jika ada
#        pertanyaan lama yang mirip secara makna.
#   L2 - LLM exact-prompt cache (SQLite) : lewati pemanggilan Gemini jika
#        prompt akhir persis sama seperti sebelumnya.
#   L3 - Embedding cache (SQLite) : jangan hitung ulang embedding lokal untuk
#        teks yang sudah pernah di-embed (baik saat indexing maupun query).
# ---------------------------------------------------------------------------
EMBEDDING_CACHE_PATH = STORAGE_DIR / "embedding_cache.sqlite"
LLM_CACHE_PATH = STORAGE_DIR / "llm_cache.sqlite"
SEMANTIC_CACHE_PATH = STORAGE_DIR / "semantic_cache.pkl"

SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.94"))
SEMANTIC_CACHE_MAX_ITEMS = int(os.getenv("SEMANTIC_CACHE_MAX_ITEMS", "1000"))

# ---------------------------------------------------------------------------
# Hyperparameter retrieval
# ---------------------------------------------------------------------------
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "8"))
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "8"))
HYBRID_TOP_N = int(os.getenv("HYBRID_TOP_N", "6"))
GRAPH_EXPAND_PER_FIGHTER = int(os.getenv("GRAPH_EXPAND_PER_FIGHTER", "2"))
RRF_K = int(os.getenv("RRF_K", "60"))  # konstanta Reciprocal Rank Fusion

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence_transformers").strip().lower()
SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)


def require_api_key() -> None:
    """Panggil di titik masuk (entrypoint) untuk gagal cepat & jelas
    jika API key belum di-set, daripada error kriptik dari Google SDK."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY belum di-set. Salin .env.example menjadi .env "
            "lalu isi API key Gemini kamu (https://aistudio.google.com/apikey)."
        )
