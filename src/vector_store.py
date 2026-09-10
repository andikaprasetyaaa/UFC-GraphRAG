"""
vector_store.py
================
Wrapper tipis di atas FAISS (via langchain_community.vectorstores.FAISS)
untuk retrieval SEMANTIK/vektor. Index disimpan ke disk (`save_local`)
supaya build_index.py hanya perlu dijalankan sekali; sesi tanya-jawab
berikutnya tinggal `load_local`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def build_faiss_index(documents: List[Document], embeddings: Embeddings, path: Path) -> FAISS:
    vs = FAISS.from_documents(documents, embeddings)
    path.parent.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(path))
    return vs


def load_faiss_index(path: Path, embeddings: Embeddings) -> FAISS:
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,  # aman: index dibuat sendiri oleh build_index.py
    )
