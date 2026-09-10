"""
data_loader.py
==============
Membaca `master.csv` (data pertandingan UFC per baris = 1 pertandingan,
dengan kolom statistik fighter merah/biru) dan mengubahnya menjadi:

1. DataFrame yang sudah dibersihkan (`load_dataframe`)
2. Dokumen teks natural-language per PERTANDINGAN (`build_fight_documents`)
   -> dipakai untuk retrieval detail satu laga spesifik.
3. Dokumen teks natural-language per FIGHTER, hasil agregasi seluruh
   laganya (`build_fighter_documents`) -> dipakai untuk pertanyaan level
   karier/profil petarung ("siapa itu Khabib Nurmagomedov?").

Kedua jenis dokumen ini yang akan di-embed ke FAISS dan diindeks BM25.
Struktur relasional (fighter <-> fight <-> event) yang SAMA dipakai lagi
di `graph_builder.py` untuk membangun knowledge graph.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_core.documents import Document

NA_TEXT = "tidak diketahui"


# ---------------------------------------------------------------------------
# Loading & pembersihan
# ---------------------------------------------------------------------------
def load_dataframe(csv_path) -> pd.DataFrame:
    """Baca CSV mentah dan lakukan pembersihan dasar."""
    df = pd.read_csv(csv_path)

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    if "r_dob" in df.columns:
        df["r_dob"] = pd.to_datetime(df["r_dob"], errors="coerce")
    if "b_dob" in df.columns:
        df["b_dob"] = pd.to_datetime(df["b_dob"], errors="coerce")

    # fight_id harus unik & tidak kosong -> jadi primary key dokumen/graph
    df = df.dropna(subset=["fight_id"]).drop_duplicates(subset=["fight_id"])
    return df


# ---------------------------------------------------------------------------
# Helper format angka/teks yang aman terhadap NaN
# ---------------------------------------------------------------------------
def _is_na(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and math.isnan(v):
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def s(v: Any, default: str = NA_TEXT) -> str:
    """String aman: kembalikan default kalau NaN/None/kosong.
    Merapikan whitespace/newline ganda (data mentah UFC sering punya
    newline di dalam kolom seperti `details`)."""
    if _is_na(v):
        return default
    return " ".join(str(v).split())


def num(v: Any, digits: int = 0, default: str = NA_TEXT) -> str:
    if _is_na(v):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f"{f:.{digits}f}" if digits else f"{int(round(f))}"


def pct(v: Any, default: str = NA_TEXT) -> str:
    if _is_na(v):
        return default
    try:
        return f"{float(v):.0f}%"
    except (TypeError, ValueError):
        return default


def date_str(v: Any, default: str = NA_TEXT) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return default
    try:
        return pd.Timestamp(v).strftime("%d %B %Y")
    except (TypeError, ValueError):
        return default


def iso_date(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    try:
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# 1) Dokumen per-PERTANDINGAN
# ---------------------------------------------------------------------------
def row_to_fight_text(row: pd.Series) -> str:
    r_name, b_name = s(row.get("r_fighter_name")), s(row.get("b_fighter_name"))
    event_name = s(row.get("event_name"))
    location = s(row.get("event_location"))
    date = date_str(row.get("event_date"))
    weight_class = s(row.get("weight_class"))
    is_title = bool(row.get("title_fight") == 1)

    winner_id = row.get("winner_id")
    result_status = s(row.get("result_status"))
    if result_status == "no_contest":
        outcome = "Pertandingan berakhir No Contest (tidak ada pemenang)."
    elif result_status == "draw":
        outcome = "Pertandingan berakhir SERI (draw)."
    elif not _is_na(winner_id):
        winner_name = r_name if winner_id == row.get("r_fighter_id") else b_name
        outcome = f"{winner_name} MENANG."
    else:
        outcome = "Hasil pertandingan tidak diketahui."

    method = s(row.get("method"))
    finish_round = row.get("finish_round")
    finish_time = s(row.get("finish_time"))
    time_format = s(row.get("time_format"))
    referee = s(row.get("referee"))
    details = s(row.get("details"), default="")
    bonuses = row.get("bonuses")

    lines: List[str] = []
    title_tag = " (LAGA GELAR JUARA / TITLE FIGHT)" if is_title else ""
    lines.append(
        f"Pertandingan {weight_class}{title_tag} antara {r_name} melawan {b_name} "
        f"pada {date} di acara \"{event_name}\" ({location})."
    )
    lines.append(
        f"{outcome} Metode: {method}"
        + (f", detail: {details}." if details else ".")
    )
    if not _is_na(finish_round):
        lines.append(
            f"Berakhir di ronde {num(finish_round)} pada menit {finish_time} "
            f"(format {time_format}). Wasit: {referee}."
        )
    if not _is_na(bonuses):
        lines.append(f"Bonus malam itu: {bonuses}.")

    # Statistik ringkas kedua fighter (kalau tersedia)
    stat_bits = []
    if not _is_na(row.get("r_total_sig_landed")):
        stat_bits.append(
            f"{r_name} melandaskan {num(row.get('r_total_sig_landed'))}/"
            f"{num(row.get('r_total_sig_atmp'))} significant strikes, "
            f"{num(row.get('r_total_td_success'))} takedown sukses, "
            f"{num(row.get('r_total_kd'))} knockdown."
        )
    if not _is_na(row.get("b_total_sig_landed")):
        stat_bits.append(
            f"{b_name} melandaskan {num(row.get('b_total_sig_landed'))}/"
            f"{num(row.get('b_total_sig_atmp'))} significant strikes, "
            f"{num(row.get('b_total_td_success'))} takedown sukses, "
            f"{num(row.get('b_total_kd'))} knockdown."
        )
    if stat_bits:
        lines.append("Statistik laga: " + " ".join(stat_bits))

    # Profil fisik singkat, membantu pertanyaan pembanding tinggi/reach
    lines.append(
        f"{r_name} ({s(row.get('r_stance'))}, tinggi {s(row.get('r_height'))}, "
        f"reach {num(row.get('r_reach_inches'))} inci) vs {b_name} "
        f"({s(row.get('b_stance'))}, tinggi {s(row.get('b_height'))}, "
        f"reach {num(row.get('b_reach_inches'))} inci)."
    )

    return " ".join(lines)


def build_fight_documents(df: pd.DataFrame) -> List[Document]:
    docs: List[Document] = []
    for _, row in df.iterrows():
        text = row_to_fight_text(row)
        metadata = {
            "doc_id": f"fight:{row['fight_id']}",
            "doc_type": "fight",
            "fight_id": row["fight_id"],
            "event_id": row.get("event_id"),
            "event_name": s(row.get("event_name"), ""),
            "date": iso_date(row.get("event_date")),
            "weight_class": s(row.get("weight_class"), ""),
            "title_fight": bool(row.get("title_fight") == 1),
            "r_fighter_id": row.get("r_fighter_id"),
            "r_fighter_name": s(row.get("r_fighter_name"), ""),
            "b_fighter_id": row.get("b_fighter_id"),
            "b_fighter_name": s(row.get("b_fighter_name"), ""),
            "winner_id": row.get("winner_id") if not _is_na(row.get("winner_id")) else None,
            "method": s(row.get("method"), ""),
            "result_status": s(row.get("result_status"), ""),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# 2) Dokumen per-FIGHTER (agregasi seluruh karier)
# ---------------------------------------------------------------------------
def _build_fighter_records(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """Kumpulkan satu record agregat per fighter_id dari sisi merah & biru."""
    records: Dict[str, Dict[str, Any]] = {}

    def ensure(fid: str) -> Dict[str, Any]:
        if fid not in records:
            records[fid] = {
                "fighter_id": fid,
                "name": None,
                "nickname": None,
                "height": None,
                "weight_lbs": None,
                "reach_inches": None,
                "stance": None,
                "dob": None,
                "slpm": None,
                "str_acc": None,
                "sapm": None,
                "str_def": None,
                "td_avg": None,
                "td_acc": None,
                "td_def": None,
                "sub_avg": None,
                "fights": [],  # list of dict: opponent, date, result, method, event
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "no_contests": 0,
            }
        return records[fid]

    for _, row in df.iterrows():
        for side, opp_side in (("r", "b"), ("b", "r")):
            fid = row.get(f"{side}_fighter_id")
            if _is_na(fid):
                continue
            rec = ensure(fid)
            if rec["name"] is None:
                rec["name"] = s(row.get(f"{side}_fighter_name"), "")
                rec["nickname"] = s(row.get(f"{side}_fighter_nick_name"), "")
                rec["height"] = s(row.get(f"{side}_height"), "")
                rec["weight_lbs"] = row.get(f"{side}_weight_lbs")
                rec["reach_inches"] = row.get(f"{side}_reach_inches")
                rec["stance"] = s(row.get(f"{side}_stance"), "")
                rec["dob"] = row.get(f"{side}_dob")
                rec["slpm"] = row.get(f"{side}_slpm")
                rec["str_acc"] = row.get(f"{side}_str_acc")
                rec["sapm"] = row.get(f"{side}_sapm")
                rec["str_def"] = row.get(f"{side}_str_def")
                rec["td_avg"] = row.get(f"{side}_td_avg")
                rec["td_acc"] = row.get(f"{side}_td_acc")
                rec["td_def"] = row.get(f"{side}_td_def")
                rec["sub_avg"] = row.get(f"{side}_sub_avg")

            opp_id = row.get(f"{opp_side}_fighter_id")
            opp_name = s(row.get(f"{opp_side}_fighter_name"), "")
            winner_id = row.get("winner_id")
            result_status = s(row.get("result_status"))
            if result_status == "draw":
                result, rec["draws"] = "seri", rec["draws"] + 1
            elif result_status == "no_contest":
                result, rec["no_contests"] = "no contest", rec["no_contests"] + 1
            elif _is_na(winner_id):
                result = "tidak diketahui"
            elif winner_id == fid:
                result, rec["wins"] = "menang", rec["wins"] + 1
            else:
                result, rec["losses"] = "kalah", rec["losses"] + 1

            rec["fights"].append(
                {
                    "fight_id": row["fight_id"],
                    "opponent_id": opp_id,
                    "opponent_name": opp_name,
                    "date": row.get("event_date"),
                    "event_name": s(row.get("event_name"), ""),
                    "weight_class": s(row.get("weight_class"), ""),
                    "result": result,
                    "method": s(row.get("method"), ""),
                }
            )
    return records


def fighter_record_to_text(rec: Dict[str, Any]) -> str:
    name = rec["name"] or "(tanpa nama)"
    nickname = f' "{rec["nickname"]}"' if rec.get("nickname") else ""
    total = len(rec["fights"])
    record_str = f"{rec['wins']}-{rec['losses']}-{rec['draws']}"
    if rec["no_contests"]:
        record_str += f" ({rec['no_contests']} NC)"

    lines = [
        f"{name}{nickname} adalah petarung UFC dengan rekor {record_str} "
        f"dari total {total} pertandingan tercatat."
    ]
    lines.append(
        f"Data fisik: tinggi {s(rec.get('height'), '')}, berat "
        f"{num(rec.get('weight_lbs'))} lbs, reach {num(rec.get('reach_inches'))} inci, "
        f"stance {s(rec.get('stance'), '')}, lahir {date_str(rec.get('dob'))}."
    )
    if not _is_na(rec.get("slpm")):
        lines.append(
            f"Statistik karier: {num(rec.get('slpm'), 2)} significant strikes landed/menit, "
            f"akurasi strike {pct(rec.get('str_acc'))}, {num(rec.get('sapm'), 2)} strikes "
            f"diterima/menit, defense strike {pct(rec.get('str_def'))}, rata-rata "
            f"{num(rec.get('td_avg'), 2)} takedown/15 menit (akurasi {pct(rec.get('td_acc'))}, "
            f"defense {pct(rec.get('td_def'))}), {num(rec.get('sub_avg'), 2)} submission "
            f"attempt/15 menit."
        )

    fights_sorted = sorted(
        rec["fights"], key=lambda f: (f["date"] is not None, f["date"]), reverse=True
    )
    recent = fights_sorted[:8]
    fight_lines = []
    for f in recent:
        fight_lines.append(
            f"- {date_str(f['date'])}: {f['result']} vs {f['opponent_name']} "
            f"via {f['method']} di {f['event_name']} ({f['weight_class']})"
        )
    if fight_lines:
        lines.append("Riwayat pertandingan terbaru:\n" + "\n".join(fight_lines))

    return "\n".join(lines)


def build_fighter_documents(df: pd.DataFrame) -> List[Document]:
    records = _build_fighter_records(df)
    docs: List[Document] = []
    for fid, rec in records.items():
        text = fighter_record_to_text(rec)
        metadata = {
            "doc_id": f"fighter:{fid}",
            "doc_type": "fighter_profile",
            "fighter_id": fid,
            "fighter_name": rec["name"] or "",
            "total_fights": len(rec["fights"]),
            "wins": rec["wins"],
            "losses": rec["losses"],
            "draws": rec["draws"],
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ---------------------------------------------------------------------------
# Entry point gabungan
# ---------------------------------------------------------------------------
def build_all_documents(df: pd.DataFrame) -> List[Document]:
    """Gabungkan dokumen per-pertandingan + per-fighter jadi satu corpus."""
    fight_docs = build_fight_documents(df)
    fighter_docs = build_fighter_documents(df)
    return fight_docs + fighter_docs
