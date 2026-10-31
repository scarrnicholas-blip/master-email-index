"""Right-of-reply dossier for one landlord.

Gathers, for every coded decision about the landlord: the determinations, each
failure claim with its evidence grade, the official paragraph numbers and
quotes it rests on, compensation ordered and any Awaab's Law duty checks. The
markdown is for sending to the landlord before publication; the CSV is a
tracker with one row per claim and blank columns for the landlord's response.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

from ..config import Settings
from ..io_utils import read_jsonl
from ..models import ClaimVerification, CodedDecision, Decision, DutyCheck


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _labels(codebook_path: Path) -> dict[str, str]:
    data = yaml.safe_load(codebook_path.read_text()) if codebook_path.exists() else {}
    items = (data.get("hypotheses") or []) + (data.get("dimensions") or [])
    return {item["id"]: item.get("label", item["id"]) for item in items}


def _load(path: Path, model):
    return list(read_jsonl(path, model)) if path.exists() else []


def landlord_dossier(settings: Settings, landlord: str, post_commencement_only: bool = False) -> Path:
    from ..awaab.duties import case_dossier
    from ..ingest.paragraphs import pos_label

    pattern = landlord.lower()
    coded = {c.case_id: c for c in _load(settings.work("coded.jsonl"), CodedDecision)}
    decisions: dict[str, Decision] = {}
    if settings.work("decisions.jsonl").exists():
        for d in read_jsonl(settings.work("decisions.jsonl"), Decision):
            if d.case_id in coded and pattern in d.landlord.lower():
                decisions[d.case_id] = d
    verified: dict[tuple[str, str], ClaimVerification] = {
        (v.case_id, v.claim_id): v for v in _load(settings.work("verified.jsonl"), ClaimVerification)
    }
    checks: dict[str, list[DutyCheck]] = {}
    for c in _load(settings.work("awaab_checks.jsonl"), DutyCheck):
        checks.setdefault(c.case_id, []).append(c)
    if post_commencement_only:
        decisions = {k: d for k, d in decisions.items() if any(c.in_force for c in checks.get(k, []))}

    labels = _labels(settings.paths.codebook)
    md_path = settings.work("outputs", "dossiers", f"{_slug(landlord)}.md")
    csv_path = md_path.with_name(f"{_slug(landlord)}_claims.csv")

    lines = [
        f"# Right-of-reply dossier: {landlord}",
        "",
        f"{len(decisions)} coded decisions. Paragraph numbers are the Ombudsman's published numbering.",
        "Grade A = confirmed by the strict second reader against the cited paragraphs; "
        "grade B = coded from the full text, not confirmed.",
        "",
    ]
    rows = []
    for cid, d in sorted(decisions.items(), key=lambda kv: (kv[1].decision_date or pd.Timestamp.min.date(), kv[0])):
        cd = coded[cid]
        lines += [f"## {d.landlord} ({cid})", ""]
        lines.append(f"- Decision date: {d.decision_date or 'unknown'}")
        if d.url:
            lines.append(f"- Published at: {d.url}")
        for det in cd.determinations:
            lines.append(f"- Determination ({det.category}): {det.finding.replace('_', ' ')} — {det.element}")
        ordered = sum(c.amount_gbp for c in cd.compensation if c.kind == "ordered")
        if ordered:
            lines.append(f"- Compensation ordered: £{ordered:,.2f}")
        lines.append("")

        claims = [h for h in cd.hypotheses if h.present == "yes"]
        if claims:
            lines += ["| Claim | Grade | Paragraphs | Quote |", "|---|---|---|---|"]
        for h in claims:
            v = verified.get((cid, h.id))
            grade = v.grade if v else "B"
            paras = []
            for cit in h.citations:
                try:
                    paras.append(pos_label(d, cit.pos))
                except Exception:
                    paras.append(f"[p{cit.pos}]")
            quote = next((c.quote for c in h.citations if c.quote), "")
            label = labels.get(h.id, h.id)
            lines.append(f"| {h.id} {label} | {grade} | {'; '.join(paras)} | {quote.replace('|', '/')} |")
            rows.append(
                {
                    "case_id": cid,
                    "landlord": d.landlord,
                    "decision_date": d.decision_date,
                    "url": d.url,
                    "claim_id": h.id,
                    "claim": label,
                    "grade": grade,
                    "verifier_verdict": v.verdict if v else "",
                    "paragraphs": "; ".join(paras),
                    "quote": quote,
                    "landlord_response": "",
                    "response_date": "",
                    "action_taken": "",
                }
            )
        if checks.get(cid):
            lines += ["", "### Awaab's Law duty checks", "", case_dossier(cid, checks[cid])]
        lines.append("")

    md_path.write_text("\n".join(lines))
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return md_path
