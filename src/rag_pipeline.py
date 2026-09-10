"""
rag_pipeline.py
================
Titik integrasi akhir: memuat semua artifact hasil `build_index.py`
(FAISS index, BM25 retriever, knowledge graph) + menyiapkan cache
berlapis + memanggil Gemini (lewat LangChain) untuk menjawab pertanyaan
bahasa natural tentang data UFC.

Urutan proses `answer()`:
  1. Embed pertanyaan sekali (dipakai ulang untuk semantic cache & FAISS).
  2. Cek SemanticCache (L1) -- kalau mirip pertanyaan lama, langsung balas.
  3. Kalau miss: jalankan GraphRAGHybridRetriever (FAISS + BM25 + graph).
  4. Cek apakah query menyebut 2 fighter (head-to-head) -> tambahkan fakta
     EKSAK dari graph sebagai grounding context.
  5. Panggil Gemini lewat LCEL chain (prompt | llm | parser). SQLiteCache
     (L2) otomatis aktif secara global lewat `init_llm_cache`.
  6. Simpan hasil ke SemanticCache untuk pertanyaan berikutnya.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import config
from src import caching, graph_builder, graph_qa, vector_store
from src.embeddings import build_embeddings
from src.hybrid_retriever import GraphRAGHybridRetriever, format_docs_for_prompt
from src.lexical_retriever import load_bm25_retriever

SYSTEM_PROMPT = """\
Kamu adalah asisten analis UFC yang menjawab HANYA berdasarkan konteks yang \
diberikan. Tujuanmu adalah memberi jawaban yang akurat, ringkas, dan dapat \
ditelusuri dari sumber data UFC.

Aturan prioritas:
1. Hanya berdasarkan konteks yang diberikan. Jangan memakai pengetahuan umum, \
   asumsi, atau mengarang fakta yang tidak ada di konteks.
2. Jika konteks memuat "[Fakta graph EKSAK]", anggap angka di sana sebagai \
   kebenaran mutlak dan jangan menghitung ulang, membantah, atau mengubah \
   angka yang sudah disebut secara eksplisit.
3. Jika informasi tidak cukup atau tidak ada di konteks, katakan dengan jujur: \
   "Tidak ada informasi yang cukup di konteks untuk menjawab secara pasti." \
   dan jangan mengarang (no hallucination).
4. Jawab dalam Bahasa Indonesia yang jelas dan ringkas, kecuali user meminta \
   bahasa lain.
5. Saat relevan, sertakan referensi seperti [1], [2], atau [3] sesuai nomor \
   dokumen konteks yang mendukung jawaban.
6. Bersikap objektif, berbasis data, dan seperti analis statistik olahraga.
7. Fokus pada fakta yang dapat diverifikasi dan hindari klaim yang terlalu umum \
   tanpa bukti di konteks.
"""

USER_PROMPT_TEMPLATE = """\
Konteks yang diambil dari database UFC:
---------------------
{context}
---------------------

Pertanyaan user:
{question}

Instruksi pengerjaan:
- Jawab hanya dari konteks di atas.
- Berikan kesimpulan singkat di awal.
- Gunakan angka yang sesuai dengan konteks, jangan melebih-lebihkan.
- Jika ada bukti yang relevan, sertakan referensi sumber seperti [1], [2].
- Jika tidak ada bukti yang cukup, tulis: "Tidak ada informasi yang cukup di konteks untuk menjawab secara pasti."
"""


class GraphRAGPipeline:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        config.require_api_key()

        caching.init_llm_cache(config.LLM_CACHE_PATH)

        self._log("Memuat embeddings (dengan cache)...")
        self.embeddings = build_embeddings()

        self._log("Memuat knowledge graph...")
        self.graph = graph_builder.load_graph(config.GRAPH_PATH)

        self._log("Memuat FAISS index...")
        self.vector_store = vector_store.load_faiss_index(config.FAISS_INDEX_DIR, self.embeddings)
        self.vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": config.VECTOR_TOP_K}
        )

        self._log("Memuat BM25 retriever...")
        self.bm25_retriever = load_bm25_retriever(config.BM25_INDEX_PATH)
        self.bm25_retriever.k = config.BM25_TOP_K

        self.hybrid_retriever = GraphRAGHybridRetriever(
            vector_retriever=self.vector_retriever,
            bm25_retriever=self.bm25_retriever,
            graph=self.graph,
            rrf_k=config.RRF_K,
            top_n=config.HYBRID_TOP_N,
            graph_expand_per_fighter=config.GRAPH_EXPAND_PER_FIGHTER,
        )

        self.semantic_cache = caching.SemanticCache(
            config.SEMANTIC_CACHE_PATH,
            threshold=config.SEMANTIC_CACHE_THRESHOLD,
            max_items=config.SEMANTIC_CACHE_MAX_ITEMS,
        )

        self._log("Menyiapkan model chat Gemini...")
        self.llm = self._build_llm()
        self.chain = self._build_chain()

    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[GraphRAGPipeline] {msg}")

    def _build_llm(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=config.GEMINI_CHAT_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.LLM_TEMPERATURE,
        )

    def _build_chain(self):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("human", USER_PROMPT_TEMPLATE)]
        )
        return prompt | self.llm | StrOutputParser()

    # ------------------------------------------------------------------
    def answer(self, question: str) -> Dict[str, Any]:
        t0 = time.time()
        query_vector = self.embeddings.embed_query(question)

        cached = self.semantic_cache.lookup_by_vector(query_vector)
        if cached:
            return {
                "answer": cached["answer"],
                "sources": cached["sources"],
                "cache_hit": "semantic",
                "similarity": cached["similarity"],
                "elapsed_seconds": round(time.time() - t0, 3),
            }

        docs = self.hybrid_retriever.invoke(question)
        context = format_docs_for_prompt(docs)

        structured = graph_qa.try_head_to_head(question, self.graph)
        if structured:
            context = graph_qa.head_to_head_to_text(structured) + "\n\n" + context

        answer_text = self.chain.invoke({"context": context, "question": question})

        sources = [
            {
                "doc_type": d.metadata.get("doc_type"),
                "ref": d.metadata.get("fight_id") or d.metadata.get("fighter_id"),
                "preview": d.page_content[:160],
            }
            for d in docs
        ]

        self.semantic_cache.store(question, query_vector, answer_text, sources)

        return {
            "answer": answer_text,
            "sources": sources,
            "cache_hit": False,
            "structured_facts_used": bool(structured),
            "elapsed_seconds": round(time.time() - t0, 3),
        }
