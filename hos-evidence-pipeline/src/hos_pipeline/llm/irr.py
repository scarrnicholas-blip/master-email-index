"""Inter-rater reliability between the LLM coder and a human second reader.

``export_irr_sheet`` writes a blank sheet (one row per case x codebook item:
hypotheses, dimensions, admin flags) for the human, and the LLM's coding of
the same cases in a separate long-format file, so the human codes blind and
the comparison is done afterwards.

Sheet columns the human fills in:
    value         yes / no / unclear (y, n, u, 1, 0, ? also accepted); blank = not coded
    citations     paragraph positions as in [pN], e.g. "p12; p15"
    item_value    the free value where the codebook asks for one (amount, deadline)
    coder         initials
    notes         anything else

``compare`` reports, per item, agreement and Cohen's kappa twice: with
"unclear" as its own category (yes / no / unclear), and with "unclear"
collapsed into "no" (present vs not shown present), which is how the
prevalence figures are usually read. Cases are counted for an item only when
both coders gave a value.
"""

from __future__ import annotations

import csv
import math
import re
import warnings
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..io_utils import read_jsonl
from ..models import Citation, CodedDecision, DimensionCode, HypothesisCode
from .codebook import Codebook, load_codebook
from .coder import iter_decisions

SHEET_COLUMNS = [
    "case_id", "landlord", "url", "item_type", "item_id", "item_label",
    "value", "citations", "item_value", "coder", "notes",
]
LLM_COLUMNS = [
    "case_id", "item_type", "item_id", "value", "citations", "item_value", "coder", "codebook_version",
]
_VALUES = {
    "yes": "yes", "y": "yes", "1": "yes", "true": "yes",
    "no": "no", "n": "no", "0": "no", "false": "no",
    "unclear": "unclear", "u": "unclear", "?": "unclear",
}


def _items(codebook: Codebook) -> list[tuple[str, str, str]]:
    return (
        [("hypothesis", x.id, x.label) for x in codebook.hypotheses]
        + [("dimension", x.id, x.label) for x in codebook.dimensions]
        + [("admin_flag", x.id, x.label) for x in codebook.admin_flags]
    )


def item_type(item_id: str) -> str:
    if re.fullmatch(r"H\d+", item_id):
        return "hypothesis"
    if re.fullmatch(r"D\d+", item_id):
        return "dimension"
    return "admin_flag"


def _item_order(item_id: str) -> tuple[int, int, str]:
    m = re.fullmatch(r"([HD])(\d+)", item_id)
    if m:
        return (0 if m[1] == "H" else 1, int(m[2]), "")
    return (2, 0, item_id)


def _cites(citations: Sequence[Citation]) -> str:
    return "; ".join(f"p{c.pos}" for c in citations)


def export_irr_sheet(settings: Settings, case_ids: list[str], out_csv: Path) -> tuple[Path, Path | None]:
    """Blank human coding sheet, plus the LLM coding of the same cases in
    ``<out>_llm_coding.csv`` (None when coded.jsonl does not exist)."""
    codebook = load_codebook(settings.paths.codebook)
    ids = list(dict.fromkeys(case_ids))
    meta = {d.case_id: d for d in iter_decisions(settings.work("decisions.jsonl"), set(ids))}
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        for cid in ids:
            d = meta.get(cid)
            for kind, iid, label in _items(codebook):
                w.writerow(
                    {"case_id": cid, "landlord": d.landlord if d else "", "url": (d.url or "") if d else "",
                     "item_type": kind, "item_id": iid, "item_label": label}
                )

    coded_path = settings.work("coded.jsonl")
    if not coded_path.exists():
        return out_csv, None
    wanted = set(ids)
    coded = {c.case_id: c for c in read_jsonl(coded_path, CodedDecision) if c.case_id in wanted}
    llm_path = out_csv.with_name(out_csv.stem + "_llm_coding.csv")
    with llm_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LLM_COLUMNS)
        w.writeheader()
        for cid in ids:
            c = coded.get(cid)
            if c is None:
                continue
            base = {"case_id": cid, "coder": c.coder, "codebook_version": c.codebook_version}
            for h in c.hypotheses:
                w.writerow({**base, "item_type": "hypothesis", "item_id": h.id, "value": h.present,
                            "citations": _cites(h.citations)})
            for dim in c.remediation:
                w.writerow({**base, "item_type": "dimension", "item_id": dim.id, "value": dim.present,
                            "citations": _cites(dim.citations), "item_value": dim.value or ""})
            for fid, value in c.admin_flags.items():
                w.writerow({**base, "item_type": "admin_flag", "item_id": fid, "value": value})
    return out_csv, llm_path


def _value(raw: str | None, where: str) -> str | None:
    key = (raw or "").strip().lower()
    if not key:
        return None
    if key not in _VALUES:
        raise ValueError(f"{where}: value {raw!r} is not yes / no / unclear")
    return _VALUES[key]


def load_human_codes(csv_path: str | Path, codebook_version: str = "human") -> list[CodedDecision]:
    """Read a filled-in IRR sheet. Blank values are treated as not coded."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    out: dict[str, CodedDecision] = {}
    for i, row in df.iterrows():
        value = _value(row.get("value"), f"{csv_path} row {int(i) + 2}")
        if value is None:
            continue
        cid, iid = row["case_id"].strip(), row["item_id"].strip()
        coder = (row.get("coder") or "").strip() or "human"
        rec = out.setdefault(
            cid, CodedDecision(case_id=cid, codebook_version=codebook_version, coder=coder, hypotheses=[], remediation=[])
        )
        cites = [Citation(pos=int(p)) for p in re.findall(r"\d+", row.get("citations") or "")]
        kind = (row.get("item_type") or "").strip() or item_type(iid)
        if kind == "hypothesis":
            rec.hypotheses.append(HypothesisCode(id=iid, present=value, citations=cites))
        elif kind == "dimension":
            rec.remediation.append(
                DimensionCode(id=iid, present=value, value=(row.get("item_value") or "").strip() or None, citations=cites)
            )
        else:
            rec.admin_flags[iid] = value
    return list(out.values())


def _codes(path: Path) -> dict[tuple[str, str], str]:
    """(case_id, item_id) -> value from a CodedDecision JSONL or a filled-in sheet."""
    path = Path(path)
    coded = load_human_codes(path) if path.suffix.lower() == ".csv" else list(read_jsonl(path, CodedDecision))
    out: dict[tuple[str, str], str] = {}
    for c in coded:  # later records for the same case replace earlier ones
        for h in c.hypotheses:
            out[(c.case_id, h.id)] = h.present
        for d in c.remediation:
            out[(c.case_id, d.id)] = d.present
        for k, v in c.admin_flags.items():
            out[(c.case_id, k)] = v
    return out


def _kappa(a: list[str], b: list[str]) -> float:
    if not a:
        return math.nan
    try:
        from ..stats.core import cohens_kappa
    except ImportError:
        warnings.warn("hos_pipeline.stats.core.cohens_kappa is not available; kappa left blank", stacklevel=2)
        return math.nan
    return float(cohens_kappa(a, b))


def _share(values: list[str], target: str) -> float:
    return sum(v == target for v in values) / len(values) if values else math.nan


def compare(coded_a: Path, coded_b: Path, items: list[str] | None = None) -> pd.DataFrame:
    """Per-item agreement and Cohen's kappa between two coders.

    Each input is a CodedDecision JSONL (e.g. coded.jsonl) or a filled-in IRR
    sheet (.csv). ``agreement``/``kappa`` treat yes/no/unclear as three
    categories; ``*_collapsed`` count unclear as no. By default the items are
    those both coders coded for at least one common case.
    """
    a, b = _codes(Path(coded_a)), _codes(Path(coded_b))
    chosen = items or sorted({i for k, i in a if (k, i) in b}, key=_item_order)
    rows = []
    for item in chosen:
        cases = sorted({c for c, i in a if i == item} & {c for c, i in b if i == item})
        va = [a[(c, item)] for c in cases]
        vb = [b[(c, item)] for c in cases]
        ca = ["no" if v == "unclear" else v for v in va]
        cb = ["no" if v == "unclear" else v for v in vb]
        n = len(cases)
        rows.append(
            {
                "item_id": item,
                "item_type": item_type(item),
                "n": n,
                "agreement": sum(x == y for x, y in zip(va, vb, strict=True)) / n if n else math.nan,
                "kappa": _kappa(va, vb),
                "agreement_collapsed": sum(x == y for x, y in zip(ca, cb, strict=True)) / n if n else math.nan,
                "kappa_collapsed": _kappa(ca, cb),
                "prevalence_yes_a": _share(va, "yes"),
                "prevalence_yes_b": _share(vb, "yes"),
                "prevalence_unclear_a": _share(va, "unclear"),
                "prevalence_unclear_b": _share(vb, "unclear"),
            }
        )
    return pd.DataFrame(rows)
