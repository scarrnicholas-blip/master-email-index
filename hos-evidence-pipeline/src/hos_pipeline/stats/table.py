"""Join every stage's output into one analysis row per coded decision.

Encoding
--------
H*/D* columns: 1 = "yes", 0 = "no", NaN = "unclear" or not coded.
H*_A columns: 1 when the strict second reader verified the claim (grade A),
0 when the hypothesis was coded but not verified; NaN everywhere when
verified.jsonl does not exist (so an unrun stage never reads as 0%).

duration_days runs from the earliest possible date the landlord was aware to
the latest possible date works were completed (or, when no completion is
recorded, the determination date; see ``duration_end``). With month- or
year-precision dates this is the longest duration consistent with the text;
``duration_days_min`` is the shortest.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from hos_pipeline.config import Settings
from hos_pipeline.io_utils import read_jsonl
from hos_pipeline.models import FINDING_ORDER

log = logging.getLogger(__name__)

N_HYPOTHESES = 12
N_DIMENSIONS = 13
AWARE_EVENT = "landlord_aware"
WORKS_EVENT = "works_completed"
_PRECISION_RANK = {"day": 0, "month": 1, "year": 2, "unknown": 3}
_TRI = {"yes": 1.0, "no": 0.0}

_COUNCIL_RE = re.compile(
    r"\b(council|borough|city of|district|county|corporation of|royal borough|metropolitan borough)\b", re.I
)
_HA_RE = re.compile(r"\b(housing|homes|association|trust|group|living|society|partnership|properties)\b", re.I)


def _fallback_landlord_type(name: str | None) -> str:
    if not name:
        return "other"
    if _COUNCIL_RE.search(name):
        return "local_authority"
    if _HA_RE.search(name):
        return "housing_association"
    return "other"


def _landlord_type_fn() -> Callable[[str], str]:
    """``hos_pipeline.screen.sample.landlord_type`` when available, else a minimal heuristic."""
    try:
        from hos_pipeline.screen.sample import landlord_type  # noqa: PLC0415 (written by another stage)
    except Exception:  # noqa: BLE001 - any import failure means "use the fallback"
        return _fallback_landlord_type

    def safe(name: str) -> str:
        try:
            value = landlord_type(name)
            return str(value) if value else _fallback_landlord_type(name)
        except Exception:  # noqa: BLE001
            return _fallback_landlord_type(name)

    return safe


def natural_ids(ids: Iterable[str], prefix: str, n: int) -> list[str]:
    """``prefix1..prefixN`` plus any extra ids found, in natural order."""
    found = {i for i in ids if isinstance(i, str) and re.fullmatch(rf"{prefix}\d+", i)}
    found |= {f"{prefix}{i}" for i in range(1, n + 1)}
    return sorted(found, key=lambda s: int(s[len(prefix):]))


def _read(path: Path) -> list[dict]:
    return list(read_jsonl(path)) if path.exists() else []


def _to_date(value) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _decision_index(path: Path) -> dict[str, dict]:
    """case_id -> {landlord, year, decision_date}, reading decisions.jsonl one line at a time."""
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for rec in read_jsonl(path):
        out[rec["case_id"]] = {
            "landlord": rec.get("landlord"),
            "year": rec.get("year"),
            "decision_date": _to_date(rec.get("decision_date")),
        }
    return out


def _most_serious(findings: Iterable[str]) -> str | None:
    best, best_rank = None, -2
    for f in findings:
        rank = FINDING_ORDER.index(f) if f in FINDING_ORDER else -1
        if rank > best_rank:
            best, best_rank = f, rank
    return best


def _duration(events: list[dict], determination: date | None) -> dict:
    out = {"duration_days": np.nan, "duration_days_min": np.nan, "duration_precision": None, "duration_end": None}
    aware = [e for e in events if e.get("type") == AWARE_EVENT and _to_date(e.get("earliest"))]
    if not aware:
        return out
    start = min(aware, key=lambda e: _to_date(e["earliest"]))
    start_early, start_late = _to_date(start["earliest"]), _to_date(start.get("latest")) or _to_date(start["earliest"])
    works = [e for e in events if e.get("type") == WORKS_EVENT and (_to_date(e.get("latest")) or _to_date(e.get("earliest")))]
    if works:
        end = max(works, key=lambda e: _to_date(e.get("latest")) or _to_date(e.get("earliest")))
        end_late = _to_date(end.get("latest")) or _to_date(end["earliest"])
        end_early = _to_date(end.get("earliest")) or end_late
        end_prec, end_kind = end.get("precision", "unknown"), "works_completed"
    elif determination:
        end_early = end_late = determination
        end_prec, end_kind = "day", "determination"
    else:
        return out
    longest = (end_late - start_early).days
    if longest < 0:
        return out
    shortest = max((end_early - start_late).days, 0)
    prec = max([start.get("precision", "unknown"), end_prec], key=lambda p: _PRECISION_RANK.get(p, 3))
    out.update(duration_days=float(longest), duration_days_min=float(shortest), duration_precision=prec,
               duration_end=end_kind)
    return out


def build_analysis_table(settings: Settings) -> pd.DataFrame:
    """One row per coded decision; also written to ``work/analysis_table.csv``."""
    work = settings.paths.work_dir
    coded = _read(work / "coded.jsonl")
    decisions = _decision_index(work / "decisions.jsonl")
    screen = {r["case_id"]: r for r in _read(work / "screen.jsonl")}
    sample = {r["case_id"]: r for r in _read(work / "sample.jsonl")}
    outcomes = {r["case_id"]: r for r in _read(work / "outcomes.jsonl")}
    verified_path = work / "verified.jsonl"
    has_verified = verified_path.exists()
    grade_a = {(r["case_id"], r["claim_id"]) for r in _read(verified_path) if r.get("grade") == "A"}
    awaab_path = work / "awaab_checks.jsonl"
    awaab: dict[str, dict[str, int]] = defaultdict(lambda: {"checks": 0, "breached": 0, "possibly": 0})
    for r in _read(awaab_path):
        a = awaab[r["case_id"]]
        a["checks"] += 1
        a["breached"] += r.get("status") == "breached"
        a["possibly"] += r.get("status") == "possibly_breached"

    h_ids = natural_ids((h["id"] for c in coded for h in c.get("hypotheses", [])), "H", N_HYPOTHESES)
    d_ids = natural_ids((d["id"] for c in coded for d in c.get("remediation", [])), "D", N_DIMENSIONS)
    ltype = _landlord_type_fn()

    rows = []
    for c in coded:
        cid = c["case_id"]
        dec, scr, smp, out = decisions.get(cid, {}), screen.get(cid, {}), sample.get(cid), outcomes.get(cid)
        landlord = dec.get("landlord") or scr.get("landlord") or ""
        ddate = dec.get("decision_date")
        year = dec.get("year") or scr.get("year") or (ddate.year if ddate else None)
        row: dict = {
            "case_id": cid,
            "landlord": landlord,
            "landlord_type": ltype(landlord),
            "year": year,
            "decision_date": ddate,
            "stratum": smp["stratum"] if smp else None,
            "population_n": smp["population_n"] if smp else np.nan,
            "weight": smp["weight"] if smp else np.nan,
            "codebook_version": c.get("codebook_version"),
            "coder": c.get("coder"),
        }
        hyp = {h["id"]: h.get("present") for h in c.get("hypotheses", [])}
        for h in h_ids:
            row[h] = _TRI.get(hyp.get(h), np.nan)
        for h in h_ids:
            if not has_verified or h not in hyp:
                row[f"{h}_A"] = np.nan
            else:
                row[f"{h}_A"] = 1.0 if (cid, h) in grade_a else 0.0
        dims = {d["id"]: d.get("present") for d in c.get("remediation", [])}
        for d in d_ids:
            row[d] = _TRI.get(dims.get(d), np.nan)

        row["any_severe"] = float(out["any_severe"]) if out else np.nan
        row["severe_damp"] = float(out["severe_damp"]) if out else np.nan
        row["severe_complaint_handling"] = float(out["severe_complaint_handling"]) if out else np.nan
        row["kw_severe_damp"] = float(scr["keyword_severe_damp"]) if "keyword_severe_damp" in scr else np.nan
        damp = [d["finding"] for d in c.get("determinations", []) if d.get("category") == "damp_mould"]
        row["finding_damp"] = _most_serious(damp)
        row["severe_damp_coded"] = float(row["finding_damp"] == "severe_maladministration") if damp else np.nan
        row["total_compensation_ordered"] = float(
            sum(x.get("amount_gbp", 0.0) for x in c.get("compensation", []) if x.get("kind") == "ordered")
        )
        det_events = [_to_date(e.get("latest")) for e in c.get("events", []) if e.get("type") == "determination"]
        row.update(_duration(c.get("events", []), ddate or max((d for d in det_events if d), default=None)))
        if awaab_path.exists():
            a = awaab.get(cid, {"checks": 0, "breached": 0, "possibly": 0})
            row["n_awaab_checks"], row["n_awaab_breached"] = a["checks"], a["breached"]
            row["n_awaab_possibly_breached"] = a["possibly"]
        else:
            row["n_awaab_checks"] = row["n_awaab_breached"] = row["n_awaab_possibly_breached"] = np.nan
        rows.append(row)

    columns = (
        ["case_id", "landlord", "landlord_type", "year", "decision_date", "stratum", "population_n", "weight",
         "codebook_version", "coder"]
        + h_ids + [f"{h}_A" for h in h_ids] + d_ids
        + ["any_severe", "severe_damp", "severe_complaint_handling", "kw_severe_damp", "finding_damp",
           "severe_damp_coded", "total_compensation_ordered", "duration_days", "duration_days_min",
           "duration_precision", "duration_end", "n_awaab_checks", "n_awaab_breached", "n_awaab_possibly_breached"]
    )
    df = pd.DataFrame(rows, columns=columns)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if not coded:
        log.warning("coded.jsonl missing or empty: analysis table has no rows")
    df.to_csv(settings.work("analysis_table.csv"), index=False)
    return df
