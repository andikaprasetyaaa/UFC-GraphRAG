"""
caching.py
==========
Implementasi cache berlapis ("canggih") supaya sistem hemat biaya & latensi
saat berulang kali dipanggil dengan pertanyaan yang sama/mirip:

  L1 - SemanticCache        : cache jawaban akhir, key = kemiripan MAKNA
                               pertanyaan (cosine similarity embedding).
                               Cache hit di sini = skip retrieval + skip
                               pemanggilan LLM sama sekali.
  L2 - SQLiteCache (LLM)    : cache exact-match untuk pemanggilan Gemini
                               (prompt akhir persis sama -> jawaban lama
                               dipakai lagi, tanpa hit API).
  L3 - PersistentCachedEmbeddings : cache exact-match untuk embedding per
                               potongan teks, dipakai baik saat build index
                               (ribuan dokumen) maupun saat runtime (query
                               berulang) supaya tidak menghitung ulang
                               vektor yang sama.

Semua cache disimpan di disk (SQLite / pickle) di folder storage/ sehingga
tetap ada walau proses di-restart.
"""
from __future__ import annotations

import hashlib
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------------
# L2 - LLM exact-prompt cache
# ---------------------------------------------------------------------------
def init_llm_cache(path: Path) -> None:
    """Aktifkan cache global LangChain untuk semua pemanggilan chat model.
    Prompt akhir yang identik tidak akan memanggil Gemini API lagi."""
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(SQLiteCache(database_path=str(path)))


# ---------------------------------------------------------------------------
# L3 - Embedding cache (exact-match, disk-backed via SQLite)
# ---------------------------------------------------------------------------
class PersistentCachedEmbeddings(Embeddings):
    """Bungkus Embeddings apa pun (mis. GoogleGenerativeAIEmbeddings) dengan
    cache SQLite lokal. Key = sha256(model_name + teks). Ini mengurangi
    biaya & latensi baik untuk indexing ulang dokumen yang tidak berubah
    maupun query yang berulang."""

    def __init__(self, underlying: Embeddings, cache_path: Path, namespace: str = "default"):
        self._underlying = underlying
        self._namespace = namespace
        self._conn = sqlite3.connect(str(cache_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache "
            "(key TEXT PRIMARY KEY, vector BLOB, created_at REAL)"
        )
        self._conn.commit()

    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self._namespace}::{text}".encode("utf-8")).hexdigest()
        return h

    def _get_many(self, texts: List[str]) -> Dict[str, List[float]]:
        keys = [self._key(t) for t in texts]
        placeholders = ",".join("?" * len(keys))
        if not keys:
            return {}
        rows = self._conn.execute(
            f"SELECT key, vector FROM embedding_cache WHERE key IN ({placeholders})",
            keys,
        ).fetchall()
        return {k: pickle.loads(v) for k, v in rows}

    def _put_many(self, items: Dict[str, List[float]]) -> None:
        now = time.time()
        self._conn.executemany(
            "INSERT OR REPLACE INTO embedding_cache (key, vector, created_at) VALUES (?, ?, ?)",
            [(k, pickle.dumps(v), now) for k, v in items.items()],
        )
        self._conn.commit()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        keys = [self._key(t) for t in texts]
        cached = self._get_many(texts)
        missing_idx = [i for i, k in enumerate(keys) if k not in cached]

        if missing_idx:
            missing_texts = [texts[i] for i in missing_idx]
            fresh_vectors = self._underlying.embed_documents(missing_texts)
            new_items = {keys[i]: vec for i, vec in zip(missing_idx, fresh_vectors)}
            self._put_many(new_items)
            cached.update(new_items)

        return [cached[k] for k in keys]

    def embed_query(self, text: str) -> List[float]:
        key = self._key(text)
        hit = self._get_many([text])
        if key in hit:
            return hit[key]
        vec = self._underlying.embed_query(text)
        self._put_many({key: vec})
        return vec

    def stats(self) -> Dict[str, int]:
        (count,) = self._conn.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()
        return {"cached_vectors": count}


def get_cached_embeddings(underlying: Embeddings, cache_path: Path, namespace: str) -> PersistentCachedEmbeddings:
    return PersistentCachedEmbeddings(underlying, cache_path, namespace=namespace)


# ---------------------------------------------------------------------------
# L1 - Semantic query cache
# ---------------------------------------------------------------------------
class SemanticCache:
    """Cache jawaban akhir berdasarkan KEMIRIPAN MAKNA pertanyaan, bukan
    exact string match. Jika pertanyaan baru cukup mirip (cosine similarity
    >= threshold) dengan pertanyaan yang pernah dijawab, jawaban lama
    langsung dipakai ulang -- tanpa retrieval, tanpa panggilan LLM."""

    def __init__(self, path: Path, threshold: float = 0.94, max_items: int = 1000):
        self.path = Path(path)
        self.threshold = threshold
        self.max_items = max_items
        self.entries: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self.entries = pickle.load(f)
            except (pickle.PickleError, EOFError, ValueError):
                self.entries = []

    def _save(self) -> None:
        with open(self.path, "wb") as f:
            pickle.dump(self.entries, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return float(np.dot(a, b) / denom)

    def lookup_by_vector(self, query_vector: List[float]) -> Optional[Dict[str, Any]]:
        if not self.entries:
            return None
        qv = np.asarray(query_vector, dtype=np.float32)
        best_entry, best_score = None, -1.0
        for entry in self.entries:
            score = self._cosine(entry["vector"], qv)
            if score > best_score:
                best_entry, best_score = entry, score
        if best_entry is not None and best_score >= self.threshold:
            return {**best_entry, "similarity": best_score}
        return None

    def store(self, query: str, query_vector: List[float], answer: str, sources: Optional[List[Dict]] = None) -> None:
        entry = {
            "query": query,
            "vector": np.asarray(query_vector, dtype=np.float32),
            "answer": answer,
            "sources": sources or [],
            "created_at": time.time(),
        }
        self.entries.append(entry)
        if len(self.entries) > self.max_items:
            # buang entry paling lama (FIFO) supaya cache tidak membengkak
            self.entries = self.entries[-self.max_items :]
        self._save()

    def stats(self) -> Dict[str, int]:
        return {"cached_answers": len(self.entries)}
