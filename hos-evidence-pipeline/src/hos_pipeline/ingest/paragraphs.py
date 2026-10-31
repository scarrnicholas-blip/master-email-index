"""Convert extraction positions ("[p109]") to the Ombudsman's paragraph numbers ("paragraph 23").

Coders cite ``[p{pos}]``; readers need the published numbering. Text that the
Ombudsman did not number is labelled by where it sits: after which numbered
paragraph, or under which heading.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..io_utils import read_jsonl, write_jsonl
from ..models import Decision, Paragraph

log = logging.getLogger(__name__)

_CITATION = re.compile(r"\[\s*p\d+(?:\s*(?:[,;]|[-–—]|to\b|and\b)\s*p?\d+)*\s*\]", re.I)
_RANGE = re.compile(r"^p?(\d+)\s*(?:[-–—]|to)\s*p?(\d+)$", re.I)
_CASE_ID = re.compile(r'"case_id"\s*:\s*"([^"]+)"')


def pos_label(decision: Decision, pos: int) -> str:
    """"paragraph 23", "paragraph 29(a)", "unnumbered text after paragraph 15" or
    "unnumbered text under 'Our approach'". Raises KeyError for an unknown position."""
    para = _find(decision, pos)
    if para.number is not None:
        return f"paragraph {_official(para)}"
    prev = _previous_numbered(decision, para)
    if prev is not None:
        return f"unnumbered text after paragraph {_official(prev)}"
    if para.section:
        return f"unnumbered text under '{para.section}'"
    return "unnumbered text in the report header"


def short_label(decision: Decision, pos: int) -> str:
    """The compact form used inside citations: "23", "29(a)", "after 15", "'Our approach' (unnumbered)"."""
    para = _find(decision, pos)
    if para.number is not None:
        return _official(para)
    prev = _previous_numbered(decision, para)
    if prev is not None:
        return f"after {_official(prev)}"
    return f"'{para.section}' (unnumbered)" if para.section else "header (unnumbered)"


def number_to_pos(decision: Decision, number: int, sub: str | None = None) -> int | None:
    """The first position carrying official paragraph ``number`` (and sub-item ``sub``)."""
    want = _sub_key(sub)
    fallback = None
    for p in decision.paragraphs:
        if p.number != number:
            continue
        if _sub_key(p.sub) == want:
            return p.pos
        if want is None and fallback is None:
            fallback = p.pos
    return fallback


def rewrite_citations(text: str, decision: Decision, template: str = "[para {label}]") -> str:
    """Replace "[p127]", "[p127, p128]" and "[p127-p129]" with official numbers.

    ``template`` may use {label} ("23", "23, 24", "23-25"), {long} (the
    ``pos_label`` wording) and {pos} (the original "p127-p129"). A citation with
    any unknown position is left unchanged so nothing is silently mislabelled.
    """
    if not isinstance(text, str):
        return text

    def repl(m: re.Match) -> str:
        inner = m.group(0).strip("[] \t")
        try:
            short, long = _labels(inner, decision)
        except (KeyError, ValueError):
            return m.group(0)
        return template.format(label=", ".join(short), long="; ".join(long), pos=inner)

    return _CITATION.sub(repl, text)


def load_decisions(decisions_path: str | Path, case_ids: Iterable[str]) -> dict[str, Decision]:
    """Stream decisions.jsonl and parse only the wanted decisions."""
    wanted = {str(c) for c in case_ids}
    found: dict[str, Decision] = {}
    with Path(decisions_path).open(encoding="utf-8") as fh:
        for line in fh:
            m = _CASE_ID.search(line[:300])
            if not m or m[1] not in wanted or m[1] in found:
                continue
            found[m[1]] = Decision.model_validate_json(line)
            if len(found) == len(wanted):
                break
    return found


def convert_file(
    in_path: str | Path,
    decisions_path: str | Path,
    out_path: str | Path,
    columns: list[str],
    case_id_column: str = "case_id",
) -> int:
    """Add ``<col>_official`` beside each named column, with citations converted.

    Reads and writes CSV, TSV or JSONL (chosen by file extension). The original
    columns are kept, so the ``[pN]`` audit trail survives. Returns the row count.
    """
    in_path, out_path = Path(in_path), Path(out_path)
    rows, fields = _read_rows(in_path)
    missing = [c for c in [case_id_column, *columns] if c not in fields]
    if missing:
        raise ValueError(f"{in_path} has no column(s) {missing}; found {fields}")

    ids = {str(r[case_id_column]) for r in rows if r.get(case_id_column) not in (None, "")}
    decisions = load_decisions(decisions_path, ids)
    unknown = 0
    for r in rows:
        d = decisions.get(str(r.get(case_id_column)))
        unknown += d is None
        for col in columns:
            r[f"{col}_official"] = _rewrite_value(r.get(col), d) if d else r.get(col)
    if unknown:
        log.warning("%d of %d rows have a case id not found in %s; left unchanged", unknown, len(rows), decisions_path)

    out_fields: list[str] = []
    for f in fields:
        out_fields.append(f)
        if f in columns:
            out_fields.append(f"{f}_official")
    _write_rows(out_path, rows, out_fields)
    return len(rows)


# --------------------------------------------------------------------------


def _find(decision: Decision, pos: int) -> Paragraph:
    paras = decision.paragraphs
    if 1 <= pos <= len(paras) and paras[pos - 1].pos == pos:
        return paras[pos - 1]
    for p in paras:
        if p.pos == pos:
            return p
    raise KeyError(f"[p{pos}] is not in decision {decision.case_id} ({len(paras)} paragraphs)")


def _official(p: Paragraph) -> str:
    subs = "".join(f"({s})" for s in p.sub.split(".")) if p.sub else ""
    return f"{p.number}{subs}"


def _previous_numbered(decision: Decision, para: Paragraph) -> Paragraph | None:
    """The nearest numbered paragraph before ``para`` within the same section."""
    idx = next(i for i, p in enumerate(decision.paragraphs) if p is para)
    for p in reversed(decision.paragraphs[:idx]):
        if p.section != para.section:
            return None
        if p.number is not None:
            return p
    return None


def _sub_key(sub: str | None) -> str | None:
    if sub is None:
        return None
    return ".".join(re.findall(r"[a-z0-9]+", sub.lower())) or None


def _labels(inner: str, decision: Decision) -> tuple[list[str], list[str]]:
    short: list[str] = []
    long: list[str] = []
    for part in re.split(r"\s*(?:[,;]|\band\b)\s*", inner):
        part = part.strip()
        if not part:
            continue
        if m := _RANGE.match(part):
            a, b = int(m[1]), int(m[2])
            sa, sb = short_label(decision, a), short_label(decision, b)
            la, lb = pos_label(decision, a), pos_label(decision, b)
            s, lng = (sa, la) if sa == sb else (f"{sa}-{sb}", f"{la} to {lb}")
        else:
            pos = int(part.lstrip("pP"))
            s, lng = short_label(decision, pos), pos_label(decision, pos)
        if s not in short:
            short.append(s)
            long.append(lng)
    if not short:
        raise ValueError(inner)
    return short, long


def _rewrite_value(value: Any, decision: Decision) -> Any:
    if isinstance(value, str):
        return rewrite_citations(value, decision)
    if isinstance(value, list):
        return [_rewrite_value(v, decision) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_value(v, decision) for k, v in value.items()}
    return value


def _format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix in (".csv", ".tsv"):
        return suffix[1:]
    raise ValueError(f"{path}: expected .csv, .tsv or .jsonl")


def _read_rows(path: Path) -> tuple[list[dict], list[str]]:
    fmt = _format(path)
    if fmt == "jsonl":
        rows = list(read_jsonl(path))
        fields: list[str] = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        return rows, fields
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t" if fmt == "tsv" else ",")
        return list(reader), list(reader.fieldnames or [])


def _write_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    fmt = _format(path)
    if fmt == "jsonl":
        write_jsonl(path, rows)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t" if fmt == "tsv" else ",", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
