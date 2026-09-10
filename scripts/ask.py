#!/usr/bin/env python3
"""
ask.py
=======
CLI untuk tanya-jawab memakai GraphRAG Hybrid Retrieval Pipeline.

Mode interaktif:
    python scripts/ask.py

Mode sekali tanya:
    python scripts/ask.py "Bagaimana rekor head to head Jon Jones vs Daniel Cormier?"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.rag_pipeline import GraphRAGPipeline  # noqa: E402


BANNER = """\
============================================================
 GraphRAG Hybrid Retrieval - UFC Fight Database (Gemini)
 Ketik pertanyaan, atau 'exit' / 'keluar' untuk berhenti.
============================================================\
"""


def print_result(result: dict) -> None:
    print("\n" + result["answer"].strip() + "\n")

    cache_hit = result.get("cache_hit")
    if cache_hit == "semantic":
        print(f"⚡ Cache hit (semantic, similarity={result.get('similarity', 0):.3f}) "
              f"-- {result['elapsed_seconds']}s, tidak memanggil Gemini.")
    else:
        struct = " | pakai fakta graph eksak" if result.get("structured_facts_used") else ""
        print(f"🔎 {len(result.get('sources', []))} sumber dipakai{struct} "
              f"-- {result['elapsed_seconds']}s")
        for i, src in enumerate(result.get("sources", [])[:5], start=1):
            print(f"   [{i}] {src['doc_type']} (ref:{src['ref']}) - {src['preview'][:90]}...")
    print("-" * 60)


def check_index_exists() -> bool:
    missing = [
        p for p in (config.FAISS_INDEX_DIR, config.BM25_INDEX_PATH, config.GRAPH_PATH)
        if not Path(p).exists()
    ]
    if missing:
        print("❌ Index belum lengkap, file/folder berikut belum ada:")
        for m in missing:
            print(f"   - {m}")
        print("Jalankan dulu: python scripts/build_index.py")
        return False
    return True


def main() -> None:
    if not check_index_exists():
        sys.exit(1)

    config.require_api_key()
    pipeline = GraphRAGPipeline(verbose=True)

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = pipeline.answer(question)
        print_result(result)
        return

    print(BANNER)
    while True:
        try:
            question = input("\nTanya> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSampai jumpa!")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", "keluar"}:
            print("Sampai jumpa!")
            break
        result = pipeline.answer(question)
        print_result(result)


if __name__ == "__main__":
    main()
