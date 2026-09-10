"""
hybrid_retriever.py
====================
Inti dari sistem: GraphRAG Hybrid Retriever.

Alur `_get_relevant_documents`:
    1. Jalankan retrieval VEKTOR (FAISS/sentence-transformers) -> top-k semantik.
  2. Jalankan retrieval LEKSIKAL (BM25) -> top-k kata kunci.
  3. Gabungkan keduanya dengan **Reciprocal Rank Fusion (RRF)** -- metode
     fusion yang lebih robust daripada rata-rata skor mentah, karena skor
     BM25 & cosine similarity punya skala yang tidak sepadan.
  4. **Graph expansion**: dari dokumen top hasil fusion, tarik entitas
     fighter yang disebut, lalu telusuri knowledge graph untuk menambah
     N laga terbaru masing-masing fighter sebagai konteks tambahan --
     inilah bagian "Graph" pada GraphRAG, membantu LLM menjawab
     pertanyaan yang butuh konteks relasional (bukan cuma satu dokumen
     yang paling mirip).
  5. Dedup & kembalikan.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from . import graph_builder as gb


def _fight_attrs_to_text(attrs: Dict[str, Any]) -> str:
    winner = (
        attrs.get("self_name")
        if attrs.get("self_result") == "menang"
        else attrs.get("opponent_name")
        if attrs.get("self_result") == "kalah"
        else None
    )
    hasil = f"{winner} menang" if winner else "hasil seri/tidak ada pemenang"
    return (
        f"[Graph-expansion] Pada {attrs.get('date')}, {attrs.get('self_name')} vs "
        f"{attrs.get('opponent_name')} di {attrs.get('event_name')} "
        f"({attrs.get('weight_class')}): {hasil} via {attrs.get('method')}."
    )


class GraphRAGHybridRetriever(BaseRetriever):
    """Hybrid retriever: FAISS (vektor) + BM25 (leksikal) via Reciprocal
    Rank Fusion, diperkaya konteks knowledge graph (relasi fighter <-> fight)."""

    model_config = {"arbitrary_types_allowed": True}

    vector_retriever: Any
    bm25_retriever: Any
    graph: Any
    rrf_k: int = 60
    top_n: int = 6
    graph_expand_per_fighter: int = 2

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        fused = self._reciprocal_rank_fusion([vector_docs, bm25_docs])
        top_docs = fused[: self.top_n]

        graph_docs = self._graph_expand(top_docs)
        return self._dedupe(top_docs + graph_docs)

    # -- RRF -----------------------------------------------------------
    def _reciprocal_rank_fusion(self, doc_lists: List[List[Document]]) -> List[Document]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}
        for docs in doc_lists:
            for rank, doc in enumerate(docs):
                key = self._doc_key(doc)
                doc_map[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [doc_map[k] for k in ranked_keys]

    @staticmethod
    def _doc_key(doc: Document) -> str:
        return str(doc.metadata.get("doc_id") or hash(doc.page_content))

    # -- Graph expansion -------------------------------------------------
    def _graph_expand(self, docs: List[Document]) -> List[Document]:
        fighter_ids: Set[str] = set()
        exclude_fight_ids: Set[str] = set()

        for d in docs:
            doc_type = d.metadata.get("doc_type")
            if doc_type == "fight":
                if d.metadata.get("fight_id"):
                    exclude_fight_ids.add(d.metadata["fight_id"])
                for k in ("r_fighter_id", "b_fighter_id"):
                    if d.metadata.get(k):
                        fighter_ids.add(d.metadata[k])
            elif doc_type == "fighter_profile":
                if d.metadata.get("fighter_id"):
                    fighter_ids.add(d.metadata["fighter_id"])

        extra_docs: List[Document] = []
        for fid in fighter_ids:
            recent = gb.get_recent_fights(self.graph, fid, n=self.graph_expand_per_fighter)
            for attrs in recent:
                fight_id = attrs.get("fight_id")
                if not fight_id or fight_id in exclude_fight_ids:
                    continue
                exclude_fight_ids.add(fight_id)
                extra_docs.append(
                    Document(
                        page_content=_fight_attrs_to_text(attrs),
                        metadata={
                            "doc_id": f"graph:{fight_id}:{fid}",
                            "doc_type": "graph_expansion",
                            "fight_id": fight_id,
                            "source": "knowledge_graph",
                        },
                    )
                )
        return extra_docs

    @staticmethod
    def _dedupe(docs: List[Document]) -> List[Document]:
        seen: Set[str] = set()
        out: List[Document] = []
        for d in docs:
            key = GraphRAGHybridRetriever._doc_key(d)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
        return out


def format_docs_for_prompt(docs: List[Document]) -> str:
    """Format dokumen jadi blok teks konteks yang rapi untuk prompt LLM,
    dengan penanda sumber supaya jawaban bisa disitasi."""
    blocks = []
    for i, d in enumerate(docs, start=1):
        tag = d.metadata.get("doc_type", "dokumen")
        ref = d.metadata.get("fight_id") or d.metadata.get("fighter_id") or ""
        blocks.append(f"[{i}] ({tag} | ref:{ref})\n{d.page_content}")
    return "\n\n".join(blocks)
