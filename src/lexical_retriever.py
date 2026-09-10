"""
lexical_retriever.py
=====================
Retriever LEKSIKAL berbasis BM25, dibangun langsung di atas library
`rank_bm25` (bukan lewat modul `langchain.retrievers`/`langchain_community`
yang sedang banyak berubah struktur di ekosistem LangChain v1.x). Ini
membuat komponen BM25 stabil lintas versi dan sepenuhnya bisa di-pickle
untuk dipersist ke disk.

BM25 unggul untuk query berbasis KATA KUNCI/EKSAK: nama fighter, nama
event, istilah teknik ("armbar", "guillotine"), metode ("KO/TKO") -- hal-hal
yang kadang kurang "menonjol" secara similarity vektor tapi sangat relevan
secara leksikal.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25LexicalRetriever(BaseRetriever):
    """Retriever BM25 yang serializable & LangChain-compatible."""

    model_config = {"arbitrary_types_allowed": True}

    documents: Any
    bm25: Any
    k: int = 8

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Document] = []
        for i in ranked_idx[: self.k]:
            if scores[i] <= 0:
                continue
            results.append(self.documents[i])
        return results

    @classmethod
    def from_documents(cls, documents: List[Document], k: int = 8) -> "BM25LexicalRetriever":
        tokenized_corpus = [tokenize(d.page_content) for d in documents]
        bm25 = BM25Okapi(tokenized_corpus)
        return cls(documents=documents, bm25=bm25, k=k)


def build_bm25_retriever(documents: List[Document], k: int, path: Path) -> BM25LexicalRetriever:
    retriever = BM25LexicalRetriever.from_documents(documents, k=k)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(retriever, f, protocol=pickle.HIGHEST_PROTOCOL)
    return retriever


def load_bm25_retriever(path: Path) -> BM25LexicalRetriever:
    with open(path, "rb") as f:
        return pickle.load(f)
