"""Stage 2: keyword screening for damp and mould (evidence grade D).

Deliberately simple regex rules. They decide which decisions enter the sample;
the LLM outcome classifier replaces ``keyword_findings`` for reporting.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..config import Screening, Settings
from ..ingest.parse import section_kind
from ..io_utils import read_jsonl, write_jsonl
from ..models import FINDING_ORDER, Decision, Paragraph, ScreenResult

_DAMP_GROUPS = ("damp", "mould")
_STOP_COMPLAINT = {"background", "assessment", "determination", "orders"}
_IN_ACCORDANCE = re.compile(r"\bin accordance with paragraph\s+\d+", re.I)

# Alternatives are tried left to right at each position, and a match consumes
# its text, so "severe maladministration" and "no maladministration" are never
# also counted as plain "maladministration". Negated severity is ignored.
_TO = r"(?:(?:amount(?:ing|ed)?|amounts)\s+to\s+)?"
_FINDINGS = re.compile(
    rf"(?P<ignore>\b(?:no|not)\s+{_TO}(?:severe\s+maladministration|service\s+failure)\b)"
    rf"|(?P<no_maladministration>\b(?:no|not)\s+{_TO}maladministration\b)"
    r"|(?P<severe_maladministration>\bsevere\s+maladministration\b)"
    r"|(?P<reasonable_redress>\breasonable\s+redress\b)"
    r"|(?P<service_failure>\bservice\s+failures?\b)"
    r"|(?P<outside_jurisdiction>\boutside\s+(?:of\s+)?(?:the\s+)?"
    r"(?:(?:housing\s+)?ombudsman['’]?s\s+)?jurisdiction\b)"
    r"|(?P<maladministration>\bmaladministration\b)",
    re.I,
)

_compiled: dict[str, tuple[dict[str, re.Pattern], re.Pattern, tuple[str, ...]]] = {}


def keyword_findings(text: str) -> list[str]:
    """Findings named in ``text``, least to most serious."""
    found = {m.lastgroup for m in _FINDINGS.finditer(text)}
    return [f for f in FINDING_ORDER if f in found]


def complaint_definition_paragraphs(decision: Decision) -> list[Paragraph]:
    """Paragraphs under "The complaint" / "Complaint definition", before Background or Assessment."""
    out = []
    for p in decision.paragraphs:
        kind = section_kind(p.section)
        if kind in _STOP_COMPLAINT:
            break
        if kind == "complaint":
            out.append(p)
    return out


def determination_paragraphs(decision: Decision) -> list[Paragraph]:
    """Paragraphs under a Determination heading (plus "In accordance with paragraph N"
    rulings under a Jurisdiction heading). With neither, any paragraph that makes
    such a ruling."""
    out = [
        p for p in decision.paragraphs
        if (kind := section_kind(p.section)) == "determination"
        or (kind == "jurisdiction" and _IN_ACCORDANCE.search(p.text))
    ]
    return out or [p for p in decision.paragraphs if _IN_ACCORDANCE.search(p.text)]


def screen_decision(decision: Decision, cfg: Screening) -> ScreenResult:
    patterns, damp_rx, damp_groups = _patterns(cfg)
    text = decision.text()
    counts = {group: sum(1 for _ in rx.finditer(text)) for group, rx in patterns.items()}
    mentions = sum(counts[g] for g in damp_groups)

    complaint_hit = any(damp_rx.search(p.text) for p in complaint_definition_paragraphs(decision))
    det = determination_paragraphs(decision)
    det_hit = any(damp_rx.search(p.text) for p in det)

    findings: set[str] = set()
    severe_damp = False
    for p in det:
        found = keyword_findings(p.text)
        findings.update(found)
        severe_damp |= "severe_maladministration" in found and bool(damp_rx.search(p.text))

    fired = [
        name for name, hit in (
            ("complaint_definition", complaint_hit),
            ("determination", det_hit),
            (f"mentions>={cfg.min_mentions}", mentions >= cfg.min_mentions),
        ) if hit
    ]
    return ScreenResult(
        case_id=decision.case_id,
        landlord=decision.landlord,
        year=decision.year,
        mention_counts=counts,
        complaint_definition_hit=complaint_hit,
        determination_hit=det_hit,
        damp_flag=bool(fired),
        rule="+".join(fired) or "none",
        keyword_findings=[f for f in FINDING_ORDER if f in findings],
        keyword_severe_damp=severe_damp,
    )


def run_screen(settings: Settings) -> Path:
    """decisions.jsonl -> screen.jsonl and screen_summary.csv (counts by year)."""
    out = settings.work("screen.jsonl")
    by_year: dict[int | None, Counter] = defaultdict(Counter)

    def results():
        for decision in read_jsonl(settings.work("decisions.jsonl"), Decision):
            r = screen_decision(decision, settings.screening)
            c = by_year[r.year]
            c["decisions"] += 1
            c["damp_flagged"] += r.damp_flag
            c["keyword_severe_damp"] += r.keyword_severe_damp
            c["keyword_severe_any"] += "severe_maladministration" in r.keyword_findings
            yield r

    write_jsonl(out, results())
    _write_summary(settings.work("screen_summary.csv"), by_year)
    return out


def _write_summary(path: Path, by_year: dict[int | None, Counter]) -> None:
    fields = ["decisions", "damp_flagged", "keyword_severe_damp", "keyword_severe_any"]
    total: Counter = Counter()
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", *fields])
        for year in sorted(by_year, key=lambda y: (y is None, y or 0)):
            c = by_year[year]
            total.update(c)
            writer.writerow([year if year is not None else "unknown", *(c[f] for f in fields)])
        writer.writerow(["total", *(total[f] for f in fields)])


def _patterns(cfg: Screening) -> tuple[dict[str, re.Pattern], re.Pattern, tuple[str, ...]]:
    """Compiled term groups, the combined damp/mould pattern and the groups it uses."""
    key = json.dumps(cfg.terms, sort_keys=True)
    if key not in _compiled:
        groups = {g: "|".join(f"(?:{p})" for p in pats) for g, pats in cfg.terms.items() if pats}
        damp_groups = tuple(g for g in _DAMP_GROUPS if g in groups) or tuple(groups)
        _compiled[key] = (
            {g: re.compile(rx, re.I) for g, rx in groups.items()},
            re.compile("|".join(groups[g] for g in damp_groups), re.I),
            damp_groups,
        )
    return _compiled[key]
