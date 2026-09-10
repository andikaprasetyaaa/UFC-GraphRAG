#!/usr/bin/env python3
"""
build_index.py
================
Jalankan skrip ini SEKALI (atau setiap kali `data/master.csv` berubah)
untuk membangun semua artifact yang dipakai saat tanya-jawab:

  1. Dokumen teks natural-language (per pertandingan & per fighter)
  2. Knowledge graph (NetworkX, fighter <-> fight <-> event)
    3. FAISS vector index (embedding sentence-transformers lokal, dengan cache)
  4. BM25 lexical index

Pemakaian:
    python scripts/build_index.py
    python scripts/build_index.py --csv data/master.csv
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

# Supaya `import config` & `from src import ...` selalu berhasil
# terlepas dari direktori mana skrip ini dijalankan.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src import data_loader, graph_builder, lexical_retriever, vector_store  # noqa: E402
from src.embeddings import build_embeddings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GraphRAG index dari master.csv")
    parser.add_argument("--csv", default=str(config.CSV_PATH), help="Path ke file CSV data UFC")
    args = parser.parse_args()

    config.require_api_key()

    print(f"== 1/6 Memuat & membersihkan CSV: {args.csv} ==")
    t0 = time.time()
    df = data_loader.load_dataframe(args.csv)
    print(f"   {len(df):,} baris pertandingan dimuat ({time.time() - t0:.1f}s)")

    print("== 2/6 Membangun dokumen teks (fight + fighter profile) ==")
    t0 = time.time()
    documents = data_loader.build_all_documents(df)
    n_fight = sum(1 for d in documents if d.metadata.get("doc_type") == "fight")
    n_fighter = sum(1 for d in documents if d.metadata.get("doc_type") == "fighter_profile")
    print(f"   {len(documents):,} dokumen ({n_fight:,} fight + {n_fighter:,} fighter profile) "
          f"({time.time() - t0:.1f}s)")
    with open(config.DOCS_PATH, "wb") as f:
        pickle.dump(documents, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("== 3/6 Membangun knowledge graph ==")
    t0 = time.time()
    graph = graph_builder.build_knowledge_graph(df)
    stats = graph_builder.graph_stats(graph)
    print(f"   {stats} ({time.time() - t0:.1f}s)")
    graph_builder.save_graph(graph, config.GRAPH_PATH)

    print("== 4/6 Menyiapkan embeddings lokal (dengan cache) ==")
    embeddings = build_embeddings()

    print(f"== 5/6 Membangun FAISS index dari {len(documents):,} dokumen "
          f"(membuat embedding lokal, mungkin makan waktu) ==")
    t0 = time.time()
    vector_store.build_faiss_index(documents, embeddings, config.FAISS_INDEX_DIR)
    print(f"   FAISS index tersimpan di {config.FAISS_INDEX_DIR} ({time.time() - t0:.1f}s)")

    print("== 6/6 Membangun BM25 lexical index ==")
    t0 = time.time()
    lexical_retriever.build_bm25_retriever(documents, config.BM25_TOP_K, config.BM25_INDEX_PATH)
    print(f"   BM25 index tersimpan di {config.BM25_INDEX_PATH} ({time.time() - t0:.1f}s)")

    print("\n✅ Index selesai dibangun. Sekarang jalankan: python scripts/ask.py")


if __name__ == "__main__":
    main()
