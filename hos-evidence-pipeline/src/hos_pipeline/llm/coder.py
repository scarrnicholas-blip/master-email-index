"""Stages 3-4: structured full-text coding of sampled decisions (grade B).

Each sampled decision is sent once, rendered as ``[p{pos}] (para N) text``
lines under its section headings, with the codebook in the cached system
prompt. The model's JSON (``CodingOutput``) is then checked deterministically
before it becomes a ``models.CodedDecision``:

- every hypothesis, dimension and admin-flag id exactly once (missing ->
  "unclear", unknown ids dropped), each fix recorded in ``notes``;
- citations must point at an existing paragraph; quotes must appear in the
  cited paragraph (whitespace, quote marks and case normalised, "..." allowed
  between fragments) or the quote is dropped and the citation kept;
- a "yes" with no valid citation becomes "unclear"; events, compensation and
  determinations with no valid citation are dropped;
- event precision and bounds are recomputed from ``date_text`` by
  ``dates.parse_date_bounds``; the model's own bounds are kept only when the
  text cannot be parsed and the bounds are self-consistent.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..io_utils import append_jsonl, done_ids, read_jsonl
from ..models import (
    Citation,
    CodedDecision,
    Compensation,
    DatePrecision,
    Decision,
    Determination,
    DimensionCode,
    ElementCategory,
    Event,
    Finding,
    HypothesisCode,
    Paragraph,
    SampleRecord,
    Tristate,
)
from .client import (
    LLMRequest,
    LLMResult,
    LLMRunner,
    build_params,
    cache_ttl_for,
    estimate_cost,
    log,
    make_custom_ids,
    write_preview,
)
from .codebook import EVENT_TYPES, Codebook, load_codebook
from .dates import bounds_consistent, parse_date_bounds

STAGE = "coding"
OUTPUT_TOKENS_ASSUMED = 8000  # per decision, thinking included; for the cost estimate only
PROMPTS_DIR = Path(__file__).parent / "prompts"
MAX_QUOTE_WORDS = 40

EventType = Literal[EVENT_TYPES]  # type: ignore[valid-type]


# --------------------------------------------------------------------------
# Output schema sent as output_config.format (no defaults: every field required)
# --------------------------------------------------------------------------


class CitationOut(BaseModel):
    pos: int = Field(description="Paragraph position N from the [pN] tag")
    quote: str = Field(description="Exact words from that paragraph, at most 40 words, or empty")


class HypothesisOut(BaseModel):
    id: str = Field(description="Hypothesis id, H1 to H12")
    present: Tristate
    citations: list[CitationOut]
    rationale: str = Field(description="One or two sentences")


class DimensionOut(BaseModel):
    id: str = Field(description="Dimension id, D1 to D13")
    present: Tristate
    value: str | None = Field(description="Value where the codebook asks for one, else null")
    citations: list[CitationOut]


class FlagOut(BaseModel):
    id: str = Field(description="Administration flag id")
    value: Tristate


class EventOut(BaseModel):
    type: EventType  # type: ignore[valid-type]
    date_text: str = Field(description="The date exactly as written in the decision")
    year_if_omitted: int | None = Field(description="Year from context when date_text has none, else null")
    precision: DatePrecision
    earliest: date | None
    latest: date | None
    citations: list[CitationOut]


class CompensationOut(BaseModel):
    amount_gbp: float
    kind: Literal["ordered", "recommended", "landlord_offered", "other"]
    purpose: str
    citations: list[CitationOut]


class DeterminationOut(BaseModel):
    element: str = Field(description="The complaint element as worded in the decision")
    category: ElementCategory
    finding: Finding
    citations: list[CitationOut]


class CodingOutput(BaseModel):
    hypotheses: list[HypothesisOut]
    remediation: list[DimensionOut]
    admin_flags: list[FlagOut]
    events: list[EventOut]
    compensation: list[CompensationOut]
    determinations: list[DeterminationOut]
    notes: str


# --------------------------------------------------------------------------
# Shared helpers (also used by the verifier and the outcome classifier)
# --------------------------------------------------------------------------


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8").rstrip() + "\n"


def coder_system_prompt(codebook: Codebook) -> str:
    return load_prompt("coder_system.md") + "\n" + codebook.render_markdown()


def render_header(decision: Decision) -> str:
    lines = [f"Case ID: {decision.case_id}", f"Landlord: {decision.landlord}"]
    if decision.title:
        lines.append(f"Title: {decision.title}")
    lines.append(f"Decision date: {decision.decision_date.isoformat() if decision.decision_date else 'not stated'}")
    return "\n".join(lines)


def render_paragraphs(paragraphs: Iterable[Paragraph]) -> str:
    """``[p{pos}] (para {number}{sub}) text`` lines under section headings."""
    lines: list[str] = []
    current: str | None = None
    for p in paragraphs:
        if p.section != current:
            current = p.section
            if lines:
                lines.append("")
            lines.append(f"## {current or '(no heading)'}")
        tag = f"[p{p.pos}]"
        if p.number is not None:
            tag += f" (para {p.number}{p.sub or ''})"
        lines.append(f"{tag} {' '.join(p.text.split())}")
    return "\n".join(lines)


def iter_decisions(path: Path, ids: set[str]) -> Iterator[Decision]:
    """Stream decisions.jsonl, validating only the wanted records."""
    if not ids or not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("case_id") in ids:
                yield Decision.model_validate(rec)


def load_decisions(path: Path, ids: set[str]) -> dict[str, Decision]:
    return {d.case_id: d for d in iter_decisions(path, ids)}


_TRANSLATE = str.maketrans(
    {"‘": "'", "’": "'", "‛": "'", "′": "'", "`": "'", "“": '"', "”": '"',
     "„": '"', "″": '"', "–": "-", "—": "-", "−": "-", " ": " "}
)


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    return re.sub(r"\s+", " ", text).strip().casefold()


def quote_in_text(quote: str, text: str) -> bool:
    """Whether every fragment of ``quote`` (split at ellipses) occurs in ``text``."""
    haystack = normalise_text(text)
    fragments = re.split(r"\[\s*\.\.\.\s*\]|\.\.\.|…", normalise_text(quote))
    fragments = [f.strip(" \"'.,;:") for f in fragments]
    fragments = [f for f in fragments if f]
    return bool(fragments) and all(f in haystack for f in fragments)


# --------------------------------------------------------------------------
# Deterministic post-validation
# --------------------------------------------------------------------------


class _Checker:
    def __init__(self, decision: Decision) -> None:
        self.paras = {p.pos: p for p in decision.paragraphs}
        self.notes: list[str] = []

    def citations(self, raw: Iterable[CitationOut], where: str) -> list[Citation]:
        out: list[Citation] = []
        seen: set[tuple[int, str]] = set()
        for c in raw:
            para = self.paras.get(c.pos)
            if para is None:
                self.notes.append(f"{where}: dropped citation to non-existent [p{c.pos}]")
                continue
            quote = " ".join(c.quote.split())
            if quote and not quote_in_text(quote, para.text):
                self.notes.append(f"{where}: quote not found in [p{c.pos}]; quote dropped, citation kept")
                quote = ""
            elif len(quote.split()) > MAX_QUOTE_WORDS:
                self.notes.append(f"{where}: quote in [p{c.pos}] is over {MAX_QUOTE_WORDS} words")
            if (c.pos, quote) not in seen:
                seen.add((c.pos, quote))
                out.append(Citation(pos=c.pos, quote=quote))
        return out

    def by_id(self, items: Iterable[Any], expected: list[str], kind: str) -> dict[str, Any]:
        lookup = {e.upper(): e for e in expected}
        found: dict[str, Any] = {}
        for item in items:
            key = lookup.get(item.id.strip().upper())
            if key is None:
                self.notes.append(f"{kind}: dropped unknown id {item.id!r}")
            elif key in found:
                self.notes.append(f"{kind}: duplicate {key}; kept the first")
            else:
                found[key] = item
        for key in expected:
            if key not in found:
                self.notes.append(f"{key}: missing from model output; set to unclear")
        return found

    def event(self, ev: EventOut) -> Event | None:
        where = f"event {ev.type} ({ev.date_text!r})"
        if ev.type not in EVENT_TYPES:
            self.notes.append(f"{where}: unknown event type; dropped")
            return None
        cites = self.citations(ev.citations, where)
        if not cites:
            self.notes.append(f"{where}: no valid citation; dropped")
            return None
        precision, lo, hi = parse_date_bounds(ev.date_text)
        if precision == "unknown" and ev.year_if_omitted:
            precision, lo, hi = parse_date_bounds(ev.date_text, ev.year_if_omitted)
            if precision != "unknown":
                self.notes.append(f"{where}: year {ev.year_if_omitted} taken from context")
        if precision == "unknown":
            if ev.precision != "unknown" and bounds_consistent(ev.precision, ev.earliest, ev.latest):
                self.notes.append(f"{where}: date_text not parsed; kept the model's {ev.precision} bounds")
                precision, lo, hi = ev.precision, ev.earliest, ev.latest
            elif ev.earliest or ev.latest:
                self.notes.append(f"{where}: date_text not parsed and model bounds inconsistent; set unknown")
        elif (precision, lo, hi) != (ev.precision, ev.earliest, ev.latest):
            self.notes.append(f"{where}: bounds recomputed from date_text as {precision}")
        return Event(type=ev.type, date_text=ev.date_text, precision=precision, earliest=lo, latest=hi, citations=cites)


def postprocess(output: CodingOutput, decision: Decision, codebook: Codebook, coder: str) -> CodedDecision:
    """Turn model output into a checked CodedDecision; every fix is noted."""
    chk = _Checker(decision)

    hyps = chk.by_id(output.hypotheses, codebook.hypothesis_ids, "hypotheses")
    hypotheses: list[HypothesisCode] = []
    for hid in codebook.hypothesis_ids:
        h = hyps.get(hid)
        if h is None:
            hypotheses.append(HypothesisCode(id=hid, present="unclear"))
            continue
        cites = chk.citations(h.citations, hid)
        present = h.present
        if present == "yes" and not cites:
            present = "unclear"
            chk.notes.append(f"{hid}: 'yes' without a valid citation; downgraded to unclear")
        hypotheses.append(HypothesisCode(id=hid, present=present, citations=cites, rationale=h.rationale.strip()))

    dims = chk.by_id(output.remediation, codebook.dimension_ids, "remediation")
    remediation: list[DimensionCode] = []
    for did in codebook.dimension_ids:
        d = dims.get(did)
        if d is None:
            remediation.append(DimensionCode(id=did, present="unclear"))
            continue
        cites = chk.citations(d.citations, did)
        present = d.present
        if present == "yes" and not cites:
            present = "unclear"
            chk.notes.append(f"{did}: 'yes' without a valid citation; downgraded to unclear")
        value = d.value.strip() if d.value and d.value.strip() else None
        remediation.append(DimensionCode(id=did, present=present, value=value, citations=cites))

    flags = chk.by_id(output.admin_flags, codebook.admin_flag_ids, "admin_flags")
    admin_flags = {fid: (flags[fid].value if fid in flags else "unclear") for fid in codebook.admin_flag_ids}

    events = [e for e in (chk.event(ev) for ev in output.events) if e is not None]

    compensation: list[Compensation] = []
    for c in output.compensation:
        where = f"compensation £{c.amount_gbp:g} ({c.kind})"
        cites = chk.citations(c.citations, where)
        if c.amount_gbp <= 0:
            chk.notes.append(f"{where}: non-positive amount; dropped")
        elif not cites:
            chk.notes.append(f"{where}: no valid citation; dropped")
        else:
            compensation.append(
                Compensation(amount_gbp=c.amount_gbp, kind=c.kind, purpose=c.purpose.strip(), citations=cites)
            )

    determinations: list[Determination] = []
    for d in output.determinations:
        where = f"determination {d.finding} ({d.category})"
        cites = chk.citations(d.citations, where)
        if not cites:
            chk.notes.append(f"{where}: no valid citation; dropped")
            continue
        determinations.append(
            Determination(element=d.element.strip(), category=d.category, finding=d.finding, citations=cites)
        )

    notes = [output.notes.strip()] if output.notes.strip() else []
    notes += [f"[check] {n}" for n in chk.notes]
    return CodedDecision(
        case_id=decision.case_id,
        codebook_version=codebook.version,
        coder=coder,
        hypotheses=hypotheses,
        remediation=remediation,
        admin_flags=admin_flags,
        events=events,
        compensation=compensation,
        determinations=determinations,
        notes="\n".join(notes),
    )


# --------------------------------------------------------------------------
# Stage runner
# --------------------------------------------------------------------------


def coding_user_message(decision: Decision) -> str:
    return (
        f"{render_header(decision)}\n\n<decision>\n{render_paragraphs(decision.paragraphs)}\n</decision>\n\n"
        "Code this decision against the codebook. Return every hypothesis (H1-H12), remediation "
        "dimension (D1-D13) and administration flag exactly once, citing [pN] positions."
    )


def build_coding_requests(
    settings: Settings, codebook: Codebook, decisions: list[Decision], mode: str
) -> list[LLMRequest]:
    system = coder_system_prompt(codebook)
    custom = make_custom_ids(d.case_id for d in decisions)
    return [
        LLMRequest(
            custom_id=custom[d.case_id],
            case_id=d.case_id,
            params=build_params(
                model=settings.llm.model,
                system=system,
                user=coding_user_message(d),
                max_tokens=settings.llm.max_tokens,
                effort=settings.llm.effort_coding,
                output_model=CodingOutput,
                cache_ttl=cache_ttl_for(mode),
            ),
        )
        for d in decisions
    ]


def _sample_ids(settings: Settings, case_ids: list[str] | None) -> list[str]:
    path = settings.work("sample.jsonl")
    sample = list(dict.fromkeys(r.case_id for r in read_jsonl(path, SampleRecord))) if path.exists() else []
    if not case_ids:
        if not sample:
            raise FileNotFoundError(f"{path} is missing or empty; run the sample stage first")
        return sample
    in_sample = set(sample)
    extra = [c for c in dict.fromkeys(case_ids) if c not in in_sample]
    if extra:
        log(f"[{STAGE}] {len(extra)} requested ids are not in sample.jsonl; coding them anyway")
    wanted = set(case_ids)
    return [c for c in sample if c in wanted] + extra


def run_coding(
    settings: Settings,
    case_ids: list[str] | None = None,
    limit: int | None = None,
    dry_run: bool = False,
    mode: str = "auto",
    *,
    client: Any = None,
    async_client: Any = None,
    poll_interval: float = 60.0,
    max_wait: float | None = None,
) -> Path:
    """Code sampled decisions into coded.jsonl (resumable). Returns the output
    path, or the request preview path on a dry run (which makes no API call)."""
    codebook = load_codebook(settings.paths.codebook)
    out_path = settings.work("coded.jsonl")
    runner = LLMRunner(
        settings, STAGE, client=client, async_client=async_client, poll_interval=poll_interval, max_wait=max_wait
    )
    mode = runner.resolve_mode(mode)

    done = done_ids(out_path)
    todo = [c for c in _sample_ids(settings, case_ids) if c not in done]
    if limit is not None:
        todo = todo[:limit]
    pending = runner.pending_case_ids() if mode == "batch" and not dry_run else set()
    decisions = load_decisions(settings.work("decisions.jsonl"), set(todo) | pending)
    for cid in todo:
        if cid not in decisions:
            if dry_run:
                log(f"[{STAGE}] {cid}: not in decisions.jsonl")
            else:
                runner.log_error(cid, "decision_not_found", "not in decisions.jsonl")
    requests = build_coding_requests(settings, codebook, [decisions[c] for c in todo if c in decisions], mode)

    if dry_run:
        preview = write_preview(settings.work("coding_requests_preview.jsonl"), requests)
        est = estimate_cost(requests, output_tokens_per_request=OUTPUT_TOKENS_ASSUMED, batch=mode == "batch")
        log(f"[{STAGE}] dry run, no API call. {len(done)} already coded; {len(requests)} to code.")
        log(f"[{STAGE}] {est.summary()}")
        return preview

    def handle(res: LLMResult) -> str | None:
        if res.case_id in done:
            return None  # e.g. a batch collected twice after an interruption
        decision = decisions.get(res.case_id)
        if decision is None:
            return "decision not loaded for this result"
        try:
            output = CodingOutput.model_validate_json(res.text)
        except ValidationError as exc:
            return f"output does not match the schema: {exc}"
        append_jsonl(out_path, postprocess(output, decision, codebook, coder=res.model or settings.llm.model))
        done.add(res.case_id)
        return None

    log(f"[{STAGE}] {len(done)} already coded; {len(requests)} to code ({mode})")
    runner.run(requests, handle, mode)
    return out_path
