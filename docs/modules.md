# Module Guide

## Root files

- `config.py`: konfigurasi path, model, backend embedding, cache, dan parameter retrieval.
- `main.py`: entrypoint CLI untuk perintah build dan ask.
- `api_server.py`: aplikasi FastAPI dan endpoint frontend.
- `requirements.txt`: dependency Python backend.

## `src/`

- `data_loader.py`: membaca CSV dan membentuk dokumen natural-language.
- `embeddings.py`: memilih dan membangun embedding backend dengan cache.
- `graph_builder.py`: membangun, menyimpan, memuat, dan query knowledge graph.
- `graph_qa.py`: mendeteksi pertanyaan head-to-head dan mengambil fakta graph eksak.
- `hybrid_retriever.py`: menggabungkan FAISS, BM25, RRF, dan graph expansion.
- `lexical_retriever.py`: membangun dan memuat retriever BM25.
- `vector_store.py`: membangun dan memuat FAISS vector store.
- `caching.py`: implementasi semantic cache dan embedding/LLM cache.
- `rag_pipeline.py`: orkestrator utama retrieval, grounding, generation, dan cache.

## `scripts/`

- `build_index.py`: command untuk membangun seluruh artifact index.
- `ask.py`: command untuk bertanya melalui CLI.

## `frontend/`

- `src/main.tsx`: komponen chat utama, state pertanyaan, loading, error, dan hasil.
- `src/api.ts`: client HTTP untuk endpoint `/api/ask`.
- `src/styles.css`: layout full-window, chat bubble, responsive styling, dan animasi.
- `package.json`: script Vite dan dependency React/TypeScript.

## `tests/`

- `test_prompt_engineering.py`: regression test kontrak prompt.
- `test_embedding_backend.py`: test normalisasi dan pemilihan backend embedding.
