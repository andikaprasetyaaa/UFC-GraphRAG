# Architecture

## Diagram sistem

```mermaid
flowchart LR
    CSV[data/master.csv] --> LOAD[data_loader.py]
    CSV --> GRAPH[graph_builder.py]
    LOAD --> DOCS[Documents]
    DOCS --> FAISS[FAISS vector index]
    DOCS --> BM25[BM25 index]
    GRAPH --> KG[NetworkX knowledge graph]

    USER[User] --> UI[React/Vite chat]
    UI --> API[FastAPI /api/ask]
    API --> PIPE[GraphRAGPipeline]
    PIPE --> EMB[Sentence Transformer]
    EMB --> CACHE3[L3 embedding cache]
    PIPE --> CACHE1[L1 semantic cache]
    PIPE --> RET[Hybrid retriever]
    RET --> FAISS
    RET --> BM25
    RET --> KG
    RET --> CONTEXT[Grounded context]
    KG --> H2H[Exact head-to-head facts]
    H2H --> CONTEXT
    CONTEXT --> LLM[Gemini chat model]
    LLM --> CACHE2[L2 exact prompt cache]
    LLM --> API
    API --> UI
```

## Lapisan sistem

### 1. Data layer

`data_loader.py` mengubah baris CSV menjadi dokumen fight dan profil fighter.
`graph_builder.py` membangun node serta edge fighter, fight, dan event.

### 2. Retrieval layer

FAISS mencari kemiripan semantik, sedangkan BM25 mencari kecocokan istilah
secara leksikal. `hybrid_retriever.py` menggabungkan peringkat keduanya dengan
RRF lalu memperluas konteks memakai knowledge graph.

### 3. Reasoning layer

`graph_qa.py` menangani pertanyaan head-to-head secara terstruktur. Fakta graph
dimasukkan ke prompt sebagai fakta eksak sebelum Gemini menyusun jawaban.

### 4. Generation layer

Gemini hanya berperan sebagai chat generator berdasarkan konteks yang sudah
dikumpulkan. Embedding tidak menggunakan Gemini secara default.

### 5. Delivery layer

FastAPI menyediakan endpoint untuk frontend. React menampilkan percakapan,
loading state, error state, markdown bold sederhana, dan background motion.

## Cache

- **L1 semantic cache**: melewati retrieval dan generation untuk pertanyaan yang
  sangat mirip.
- **L2 LLM cache**: menyimpan hasil untuk prompt akhir yang sama.
- **L3 embedding cache**: menyimpan embedding teks berdasarkan hash.
