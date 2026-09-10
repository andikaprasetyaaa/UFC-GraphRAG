"""
graph_builder.py
=================
Membangun *knowledge graph* (NetworkX MultiDiGraph) dari data UFC. Inilah
komponen "Graph" pada GraphRAG: struktur relasi eksplisit antara Fighter,
Fight, dan Event yang bisa ditelusuri secara EKSAK (tanpa perlu embedding)
untuk pertanyaan seperti "rekor head-to-head A vs B" atau "5 laga terakhir
si A" -- lalu hasil traversal ini dipakai untuk MEMPERKAYA konteks yang
diberikan ke LLM (lihat `hybrid_retriever.py` & `graph_qa.py`).

Skema graph
-----------
Node:
  ("fighter", fighter_id)  attrs: name, nickname, height, weight_lbs,
                                  reach_inches, stance, dob, slpm, str_acc,
                                  sapm, str_def, td_avg, td_acc, td_def, sub_avg
  ("event", event_id)      attrs: name, date, location

Edge (MultiDiGraph, key = fight_id):
  fighter_id -> fighter_id   type="FOUGHT"    (dua arah, A->B dan B->A)
                              attrs: fight_id, event_id, event_name, date,
                                     weight_class, method, finish_round,
                                     finish_time, winner_id, result_status,
                                     title_fight, self_result ("menang"/"kalah"/"seri")
  fighter_id -> event_id     type="COMPETED_AT"
                              attrs: fight_id, date
"""
from __future__ import annotations

import difflib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from .data_loader import _is_na, date_str, iso_date, s


def build_knowledge_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()

    for _, row in df.iterrows():
        event_id = row.get("event_id")
        if not _is_na(event_id) and not g.has_node(("event", event_id)):
            g.add_node(
                ("event", event_id),
                type="event",
                event_id=event_id,
                name=s(row.get("event_name"), ""),
                date=iso_date(row.get("event_date")),
                location=s(row.get("event_location"), ""),
            )

        fighter_ids = {}
        for side in ("r", "b"):
            fid = row.get(f"{side}_fighter_id")
            if _is_na(fid):
                continue
            fighter_ids[side] = fid
            node_key = ("fighter", fid)
            if not g.has_node(node_key):
                g.add_node(
                    node_key,
                    type="fighter",
                    fighter_id=fid,
                    name=s(row.get(f"{side}_fighter_name"), ""),
                    name_lower=s(row.get(f"{side}_fighter_name"), "").lower(),
                    nickname=s(row.get(f"{side}_fighter_nick_name"), ""),
                    height=s(row.get(f"{side}_height"), ""),
                    weight_lbs=row.get(f"{side}_weight_lbs"),
                    reach_inches=row.get(f"{side}_reach_inches"),
                    stance=s(row.get(f"{side}_stance"), ""),
                    dob=iso_date(row.get(f"{side}_dob")),
                    slpm=row.get(f"{side}_slpm"),
                    str_acc=row.get(f"{side}_str_acc"),
                    sapm=row.get(f"{side}_sapm"),
                    str_def=row.get(f"{side}_str_def"),
                    td_avg=row.get(f"{side}_td_avg"),
                    td_acc=row.get(f"{side}_td_acc"),
                    td_def=row.get(f"{side}_td_def"),
                    sub_avg=row.get(f"{side}_sub_avg"),
                )

        if not _is_na(event_id):
            for fid in fighter_ids.values():
                g.add_edge(
                    ("fighter", fid),
                    ("event", event_id),
                    key=row["fight_id"],
                    type="COMPETED_AT",
                    fight_id=row["fight_id"],
                    date=iso_date(row.get("event_date")),
                )

        if "r" in fighter_ids and "b" in fighter_ids:
            r_id, b_id = fighter_ids["r"], fighter_ids["b"]
            winner_id = row.get("winner_id")
            result_status = s(row.get("result_status"))

            def self_result(fid: str) -> str:
                if result_status == "draw":
                    return "seri"
                if result_status == "no_contest":
                    return "no contest"
                if _is_na(winner_id):
                    return "tidak diketahui"
                return "menang" if winner_id == fid else "kalah"

            base_attrs = dict(
                fight_id=row["fight_id"],
                event_id=event_id,
                event_name=s(row.get("event_name"), ""),
                date=iso_date(row.get("event_date")),
                weight_class=s(row.get("weight_class"), ""),
                method=s(row.get("method"), ""),
                finish_round=row.get("finish_round"),
                finish_time=s(row.get("finish_time"), ""),
                winner_id=winner_id if not _is_na(winner_id) else None,
                result_status=result_status,
                title_fight=bool(row.get("title_fight") == 1),
            )

            g.add_edge(
                ("fighter", r_id), ("fighter", b_id), key=row["fight_id"],
                type="FOUGHT",
                self_id=r_id, opponent_id=b_id,
                self_name=s(row.get("r_fighter_name"), ""),
                opponent_name=s(row.get("b_fighter_name"), ""),
                self_result=self_result(r_id),
                **base_attrs,
            )
            g.add_edge(
                ("fighter", b_id), ("fighter", r_id), key=row["fight_id"],
                type="FOUGHT",
                self_id=b_id, opponent_id=r_id,
                self_name=s(row.get("b_fighter_name"), ""),
                opponent_name=s(row.get("r_fighter_name"), ""),
                self_result=self_result(b_id),
                **base_attrs,
            )

    return g


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path: Path) -> nx.MultiDiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Query helper -- dipakai oleh hybrid_retriever.py & graph_qa.py
# ---------------------------------------------------------------------------
def _fighter_nodes(graph: nx.MultiDiGraph):
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") == "fighter":
            yield node, attrs


def find_fighter_id(graph: nx.MultiDiGraph, name_query: str) -> Optional[Tuple[str, str, float]]:
    """Cari fighter_id berdasarkan nama (case-insensitive, fuzzy fallback).
    Return (fighter_id, matched_name, confidence 0..1) atau None."""
    if not name_query or not name_query.strip():
        return None
    q = name_query.strip().lower()

    # 1) exact match
    for (_, fid), attrs in _fighter_nodes(graph):
        if attrs.get("name_lower") == q:
            return fid, attrs.get("name", ""), 1.0

    # 2) substring match (nama query ada di dalam nama fighter, atau sebaliknya)
    candidates = []
    for (_, fid), attrs in _fighter_nodes(graph):
        nm = attrs.get("name_lower", "")
        if q in nm or nm in q:
            candidates.append((fid, attrs.get("name", ""), len(nm)))
    if candidates:
        # pilih match dengan nama terpendek (paling spesifik/pas)
        candidates.sort(key=lambda c: c[2])
        return candidates[0][0], candidates[0][1], 0.9

    # 3) fuzzy match pakai difflib
    names = [attrs.get("name", "") for _, attrs in _fighter_nodes(graph)]
    close = difflib.get_close_matches(name_query, names, n=1, cutoff=0.75)
    if close:
        matched = close[0]
        for (_, fid), attrs in _fighter_nodes(graph):
            if attrs.get("name") == matched:
                return fid, matched, 0.75
    return None


def get_fighter_node(graph: nx.MultiDiGraph, fighter_id: str) -> Optional[Dict[str, Any]]:
    return graph.nodes.get(("fighter", fighter_id))


def get_recent_fights(
    graph: nx.MultiDiGraph, fighter_id: str, n: int = 5
) -> List[Dict[str, Any]]:
    """Ambil n laga terbaru seorang fighter (dari edge FOUGHT keluar node-nya)."""
    node_key = ("fighter", fighter_id)
    if not graph.has_node(node_key):
        return []
    fights = []
    for _, _, attrs in graph.out_edges(node_key, data=True):
        if attrs.get("type") == "FOUGHT":
            fights.append(attrs)
    fights.sort(key=lambda a: a.get("date") or "", reverse=True)
    return fights[:n]


def get_head_to_head(
    graph: nx.MultiDiGraph, fighter_a_id: str, fighter_b_id: str
) -> Dict[str, Any]:
    """Rekor pertemuan langsung dua fighter, dihitung EKSAK dari graph
    (tidak melalui LLM/embedding sama sekali)."""
    a_key, b_key = ("fighter", fighter_a_id), ("fighter", fighter_b_id)
    meetings = []
    if graph.has_node(a_key) and graph.has_node(b_key):
        for _, _, attrs in graph.out_edges(a_key, data=True):
            if attrs.get("type") == "FOUGHT" and attrs.get("opponent_id") == fighter_b_id:
                meetings.append(attrs)
    meetings.sort(key=lambda a: a.get("date") or "")

    a_wins = sum(1 for m in meetings if m.get("self_result") == "menang")
    b_wins = sum(1 for m in meetings if m.get("self_result") == "kalah")
    draws = sum(1 for m in meetings if m.get("result_status") == "draw")
    no_contests = sum(1 for m in meetings if m.get("result_status") == "no_contest")

    a_name = graph.nodes.get(a_key, {}).get("name", fighter_a_id)
    b_name = graph.nodes.get(b_key, {}).get("name", fighter_b_id)

    def _winner_label(m: Dict[str, Any]) -> str:
        if m.get("result_status") == "draw":
            return "seri"
        if m.get("result_status") == "no_contest":
            return "No Contest"
        if m.get("self_result") == "menang":
            return a_name
        if m.get("self_result") == "kalah":
            return b_name
        return "tidak diketahui"

    return {
        "fighter_a_id": fighter_a_id,
        "fighter_a_name": a_name,
        "fighter_b_id": fighter_b_id,
        "fighter_b_name": b_name,
        "total_meetings": len(meetings),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "no_contests": no_contests,
        "meetings": [
            {
                "date": m.get("date"),
                "event_name": m.get("event_name"),
                "method": m.get("method"),
                "weight_class": m.get("weight_class"),
                "winner": _winner_label(m),
            }
            for m in meetings
        ],
    }


def graph_stats(graph: nx.MultiDiGraph) -> Dict[str, int]:
    fighters = sum(1 for _, a in graph.nodes(data=True) if a.get("type") == "fighter")
    events = sum(1 for _, a in graph.nodes(data=True) if a.get("type") == "event")
    fought_edges = sum(1 for _, _, a in graph.edges(data=True) if a.get("type") == "FOUGHT")
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "fighters": fighters,
        "events": events,
        "fought_edges": fought_edges,
    }
