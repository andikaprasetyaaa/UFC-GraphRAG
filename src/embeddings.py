"""
embeddings.py
=============
Factory untuk model embedding sentence-transformers,
dibungkus dengan cache L3 (`PersistentCachedEmbeddings`) supaya teks yang sama
tidak di-embed berulang kali -- baik saat build_index (ribuan dokumen) maupun
saat runtime (pertanyaan yang sering diulang pengguna).
"""
from __future__ import annotations

import config
from src.caching import get_cached_embeddings


def resolve_embedding_backend(value: str | None) -> str:
    """Normalisasi alias backend embedding agar mudah dipilih via `.env`."""
    normalized = (value or config.EMBEDDING_BACKEND or "sentence_transformers").strip().lower().replace("-", "_")
    aliases = {
        "sentence_transformer": "sentence_transformers",
        "sentence_transformers": "sentence_transformers",
        "huggingface": "sentence_transformers",
        "hf": "sentence_transformers",
    }
    return aliases.get(normalized, "sentence_transformers")


def build_embeddings():
    """Kembalikan embeddings yang sudah dibungkus cache sesuai backend yang dipilih."""
    backend = resolve_embedding_backend(config.EMBEDDING_BACKEND)

    from langchain_community.embeddings import HuggingFaceEmbeddings

    underlying = HuggingFaceEmbeddings(model_name=config.SENTENCE_TRANSFORMER_MODEL)
    namespace = "sentence_transformers"

    return get_cached_embeddings(
        underlying,
        config.EMBEDDING_CACHE_PATH,
        namespace=namespace,
    )
