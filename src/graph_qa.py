"""
graph_qa.py
============
Fast-path "structured QA" di atas knowledge graph. Untuk pertanyaan yang
menyebut DUA nama fighter sekaligus (mis. "head to head Jon Jones vs
Daniel Cormier", "rekor pertemuan Nick Diaz lawan KJ Noons"), kita bisa
menghitung jawaban EKSAK langsung dari graph -- tanpa bergantung pada
apakah LLM "mengingat" fakta itu dengan benar. Hasil hitungan ini
disisipkan sebagai konteks tambahan (grounding) ke prompt, sehingga LLM
tinggal merangkai kalimat, bukan menghitung/mengarang statistik.

Deteksi nama fighter di sini sengaja sederhana (substring match atas nama
node graph) -- cukup cepat untuk beberapa ribu fighter, dan mudah dipahami/
dimodifikasi. Untuk dataset jauh lebih besar, pertimbangkan structure
seperti Aho-Corasick automaton untuk multi-pattern matching yang lebih cepat.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import networkx as nx

from . import graph_builder as gb


def _candidate_fighter_names(graph: nx.MultiDiGraph) -> List[str]:
    return [
        attrs.get("name", "")
        for _, attrs in graph.nodes(data=True)
        if attrs.get("type") == "fighter" and attrs.get("name")
    ]


def find_mentioned_fighters(query: str, graph: nx.MultiDiGraph, max_matches: int = 2) -> List[str]:
    """Cari fighter_id yang namanya disebut secara eksplisit di dalam query."""
    q_lower = query.lower()
    matches: List[tuple] = []  # (fighter_id, name, len(name)) -- prioritaskan nama lebih panjang/spesifik
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") != "fighter":
            continue
        name_lower = attrs.get("name_lower", "")
        if len(name_lower) < 3:
            continue
        if name_lower in q_lower:
            matches.append((attrs["fighter_id"], attrs["name"], len(name_lower)))

    # urutkan dari nama terpanjang (paling spesifik) dan buang duplikat fighter_id
    matches.sort(key=lambda m: m[2], reverse=True)
    seen_ids, seen_spans = set(), []
    result = []
    for fid, name, _ in matches:
        if fid in seen_ids:
            continue
        # hindari nama yang merupakan substring dari nama lain yang sudah dipilih
        if any(name.lower() in s for s in seen_spans):
            continue
        seen_ids.add(fid)
        seen_spans.append(name.lower())
        result.append(fid)
        if len(result) >= max_matches:
            break
    return result


def try_head_to_head(query: str, graph: nx.MultiDiGraph) -> Optional[Dict[str, Any]]:
    """Jika query menyebut (kurang lebih) dua nama fighter, hitung rekor
    head-to-head EKSAK dari graph. Return None jika tidak relevan."""
    fighter_ids = find_mentioned_fighters(query, graph, max_matches=2)
    if len(fighter_ids) != 2:
        return None
    h2h = gb.get_head_to_head(graph, fighter_ids[0], fighter_ids[1])
    if h2h["total_meetings"] == 0:
        return None
    return h2h


def head_to_head_to_text(h2h: Dict[str, Any]) -> str:
    nc_txt = f", No Contest {h2h['no_contests']}x" if h2h.get("no_contests") else ""
    lines = [
        f"[Fakta graph EKSAK - Head-to-Head] {h2h['fighter_a_name']} vs {h2h['fighter_b_name']}: "
        f"total {h2h['total_meetings']}x bertemu. "
        f"{h2h['fighter_a_name']} menang {h2h['a_wins']}x, "
        f"{h2h['fighter_b_name']} menang {h2h['b_wins']}x, seri {h2h['draws']}x{nc_txt}."
    ]
    for m in h2h["meetings"]:
        lines.append(
            f"  - {m['date']}: {m['event_name']} ({m['weight_class']}), "
            f"pemenang: {m['winner']}, metode: {m['method']}"
        )
    return "\n".join(lines)
