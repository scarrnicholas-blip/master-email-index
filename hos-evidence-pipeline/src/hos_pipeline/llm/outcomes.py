"""Outcome classifier over the Determination section (replaces grade D).

Keyword screening found only about half of the decisions with severe
maladministration on damp and mould, so every damp-flagged decision's
Determination (plus Orders/Recommendations) paragraphs are classified finding
by finding, element by element. ``severe_damp`` and the other flags are then
derived deterministically from the list of determinations.

``regex_determinations`` is the deterministic first pass over the same
paragraphs: it splits each paragraph at finding phrases and categorises the
element text by keywords. It is what the keyword screen can use, and the
LLM output is the reference the recall report compares the keyword flag to.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from ..config import Settings
from ..io_utils import append_jsonl, done_ids, read_jsonl
from ..models import (
    Citation,
    Decision,
    Determination,
    ElementCategory,
    Finding,
    OutcomeClassification,
    Paragraph,
    ScreenResult,
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
from .codebook import Codebook, load_codebook
from .coder import iter_decisions, load_prompt, render_header, render_paragraphs

STAGE = "outcomes"
OUTPUT_TOKENS_ASSUMED = 1500  # per decision, thinking included; for the cost estimate only

_APOS = r"(?:'|’)"
# Alternation order matters: at any position the longer, more specific phrase
# wins, so "severe maladministration" and "no maladministration" are never
# also counted as plain "maladministration".
FINDING_RE = re.compile(
    r"\b(?:"
    r"(?P<severe>severe\s+maladministration)"
    r"|(?P<no_mal>no\s+(?:maladministration|service\s+failure))"
    r"|(?P<mal>maladministration)"
    r"|(?P<sf>service\s+failure)"
    r"|(?P<redress>reasonable\s+redress|resolve[sd]?\s+[^.;]{0,160}?satisfactorily)"
    r"|(?P<juris>(?:outside|not\s+within)\s+(?:of\s+)?(?:the\s+)?(?:ombudsman" + _APOS + r"s\s+|our\s+)?jurisdiction)"
    r")\b",
    re.IGNORECASE,
)
_GROUP_FINDING: dict[str, Finding] = {
    "severe": "severe_maladministration",
    "no_mal": "no_maladministration",
    "mal": "maladministration",
    "sf": "service_failure",
    "redress": "reasonable_redress",
    "juris": "outside_jurisdiction",
}

# Element categories, tested in order (first match wins).
_CATEGORY_RULES: list[tuple[ElementCategory, re.Pattern[str]]] = [
    (
        "complaint_handling",
        re.compile(
            r"complaints?[\s-]+handling|handling\s+of\s+(?:the\s+|its\s+|their\s+|this\s+)?"
            r"(?:associated\s+|formal\s+|related\s+|subsequent\s+|resident" + _APOS + r"s\s+)?complaints?\b"
            r"(?!\s+(?:about|regarding|concerning|relating|in\s+relation|of))",
            re.I,
        ),
    ),
    ("record_keeping", re.compile(r"record[\s-]*keeping|records?\s+management|knowledge\s+and\s+information", re.I)),
    ("damp_mould", re.compile(r"\bdamp(?:ness)?\b|\bmou?ld(?:y|s)?\b|\bcondensation\b", re.I)),
    ("leak_water", re.compile(r"\bleak(?:s|ing|age)?\b|water\s+ingress|\broof\b|\bflood(?:ing)?\b|escape\s+of\s+water", re.I)),
    ("anti_social_behaviour", re.compile(r"anti[\s-]?social|\bASB\b|noise\s+nuisance|neighbour\s+nuisance", re.I)),
    # A bare "complaint" names the complaint process, unless it introduces a subject.
    (
        "complaint_handling",
        re.compile(r"\bcomplaints?\b(?!\s+(?:about|regarding|concerning|relating|in\s+relation|that|of|is|was|are|were)\b)", re.I),
    ),
    ("repairs_other", re.compile(r"\brepairs?\b|\bdisrepair\b", re.I)),
]

_BOILERPLATE = re.compile(
    r"^(?:in\s+accordance\s+with\s+paragraph\s+[\w.()]+\s+of\s+the\s+(?:housing\s+ombudsman\s+)?scheme,?\s*)?"
    r"(?:there\s+(?:was|were|has\s+been|is)\s*:?\s*)?",
    re.I,
)
_LEADING = re.compile(
    r"^(?:in\s+respect\s+of|with\s+(?:regard|respect)\s+to|in\s+relation\s+to|regarding|concerning|in|by|for|on|with)\s+",
    re.I,
)
_RESOLVES = re.compile(r"^resolve[sd]?\s+(?P<inner>.+?)\s+satisfactorily$", re.I)


def categorise_element(text: str) -> ElementCategory:
    for category, pattern in _CATEGORY_RULES:
        if pattern.search(text):
            return category
    return "other"


def _clean_element(text: str) -> str:
    text = " ".join(text.split()).strip(" .,;:-")
    text = _BOILERPLATE.sub("", text)
    text = _LEADING.sub("", text)
    text = re.sub(r"^(?:and|or|but)\s+|\s+(?:and|or|but|is|are|was|were)$", "", text, flags=re.I)
    return text.strip(" .,;:-")[:200]


def regex_determinations(decision: Decision, sections: tuple[str, ...] = ("determination",)) -> list[Determination]:
    """Deterministic findings from the Determination-section paragraphs (grade D)."""
    out: list[Determination] = []
    seen: set[tuple[int, str, str, str]] = set()
    paragraphs = [p for p in decision.paragraphs if any(s in p.section.lower() for s in sections)]
    for p in paragraphs:
        text = " ".join(p.text.split())
        matches = list(FINDING_RE.finditer(text))
        prev_end = 0
        for i, m in enumerate(matches):
            next_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            after = text[m.end() : next_start]
            before = text[prev_end : m.start()]
            prev_end = m.end()
            group = next(k for k, v in m.groupdict().items() if v)
            finding = _GROUP_FINDING[group]
            # The element usually follows the finding ("maladministration in the
            # landlord's handling of ..."); if not, it precedes it.
            element_text = after if len(re.findall(r"\w+", after)) >= 3 else before
            inner = _RESOLVES.match(" ".join(m.group(0).split()))
            if inner and len(inner["inner"].split()) >= 3:  # "resolves the complaint about X satisfactorily"
                element_text = inner["inner"]
            category = categorise_element(m.group(0) + " " + element_text)
            element = _clean_element(element_text) or _clean_element(m.group(0))
            key = (p.pos, category, finding, element)
            if key in seen:
                continue
            seen.add(key)
            quote = " ".join(text[m.start() : next_start].split()[:40])
            out.append(
                Determination(element=element, category=category, finding=finding, citations=[Citation(pos=p.pos, quote=quote)])
            )
    return out


def classification_from(case_id: str, determinations: list[Determination], classifier: str) -> OutcomeClassification:
    def severe(category: str) -> bool:
        return any(d.category == category and d.finding == "severe_maladministration" for d in determinations)

    return OutcomeClassification(
        case_id=case_id,
        determinations=determinations,
        severe_damp=severe("damp_mould"),
        severe_complaint_handling=severe("complaint_handling"),
        any_severe=any(d.finding == "severe_maladministration" for d in determinations),
        classifier=classifier,
    )


def regex_outcome(decision: Decision) -> OutcomeClassification:
    return classification_from(decision.case_id, regex_determinations(decision), classifier="regex")


# --------------------------------------------------------------------------
# LLM classifier
# --------------------------------------------------------------------------


class OutcomeOut(BaseModel):
    element: str = Field(description="The complaint element as worded in the decision")
    category: ElementCategory
    finding: Finding
    pos: list[int] = Field(description="Positions [pN] where the finding is stated")


class OutcomesOutput(BaseModel):
    determinations: list[OutcomeOut]
    notes: str


def outcomes_system_prompt(codebook: Codebook) -> str:
    return load_prompt("outcomes_system.md") + "\n" + codebook.render_markdown(["element_categories", "findings"])


def outcome_paragraphs(decision: Decision) -> tuple[list[Paragraph], str]:
    """Determination + Orders/Recommendations paragraphs, and where they came from.

    Without a Determination heading, fall back to paragraphs that contain a
    finding phrase anywhere in the decision.
    """
    det = [p for p in decision.paragraphs if "determination" in p.section.lower()]
    orders = [p for p in decision.paragraphs if re.search(r"\border|recommendation", p.section, re.I)]
    source = "determination"
    if not det:
        det = [p for p in decision.paragraphs if FINDING_RE.search(p.text)]
        source = "finding_phrases" if det else "none"
    chosen = {p.pos: p for p in det + orders} if det else {}
    return [chosen[k] for k in sorted(chosen)], source


def outcomes_user_message(decision: Decision, paragraphs: list[Paragraph], source: str) -> str:
    caveat = (
        ""
        if source == "determination"
        else "\nThis decision has no Determination heading; these are the paragraphs that contain finding phrases.\n"
    )
    return (
        f"{render_header(decision)}\n{caveat}\n<determination>\n{render_paragraphs(paragraphs)}\n</determination>\n\n"
        "List every finding the Ombudsman made, one entry per element."
    )


def _damp_flagged(settings: Settings) -> list[str]:
    path = settings.work("screen.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run the screen stage first")
    return list(dict.fromkeys(r.case_id for r in read_jsonl(path, ScreenResult) if r.damp_flag))


def run_outcomes(
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
    """Classify the Determination section of every damp-flagged decision into outcomes.jsonl."""
    codebook = load_codebook(settings.paths.codebook)
    out_path = settings.work("outcomes.jsonl")
    runner = LLMRunner(
        settings, STAGE, client=client, async_client=async_client, poll_interval=poll_interval, max_wait=max_wait
    )
    mode = runner.resolve_mode(mode)
    done = done_ids(out_path)
    ids = _damp_flagged(settings)
    if case_ids:
        wanted = set(case_ids)
        ids = [c for c in ids if c in wanted]
    todo = [c for c in ids if c not in done]
    if limit is not None:
        todo = todo[:limit]
    pending = runner.pending_case_ids() if mode == "batch" and not dry_run else set()

    # Keep only the paragraphs sent, not whole decisions: there are thousands.
    system = outcomes_system_prompt(codebook)
    custom = make_custom_ids(todo)
    todo_set = set(todo)
    sent_pos: dict[str, set[int]] = {}
    requests: list[LLMRequest] = []
    sources: dict[str, int] = {}
    for decision in iter_decisions(settings.work("decisions.jsonl"), todo_set | pending):
        paragraphs, source = outcome_paragraphs(decision)
        sent_pos[decision.case_id] = {p.pos for p in paragraphs}
        if decision.case_id not in todo_set:
            continue
        sources[source] = sources.get(source, 0) + 1
        if not paragraphs:
            if not dry_run:
                runner.log_error(decision.case_id, "no_determination_section", "no Determination or finding paragraphs")
            continue
        requests.append(
            LLMRequest(
                custom_id=custom[decision.case_id],
                case_id=decision.case_id,
                params=build_params(
                    model=settings.llm.classifier_model,
                    system=system,
                    user=outcomes_user_message(decision, paragraphs, source),
                    max_tokens=settings.llm.max_tokens,
                    effort=settings.llm.effort_classify,
                    output_model=OutcomesOutput,
                    cache_ttl=cache_ttl_for(mode),
                ),
            )
        )
    order = {c: i for i, c in enumerate(todo)}
    requests.sort(key=lambda r: order[r.case_id])
    missing = todo_set - set(sent_pos)
    for cid in sorted(missing):
        if not dry_run:
            runner.log_error(cid, "decision_not_found", "not in decisions.jsonl")

    if dry_run:
        preview = write_preview(settings.work("outcomes_requests_preview.jsonl"), requests)
        est = estimate_cost(requests, output_tokens_per_request=OUTPUT_TOKENS_ASSUMED, batch=mode == "batch")
        log(f"[{STAGE}] dry run, no API call. {len(done)} done; {len(requests)} to classify; paragraph sources {sources}.")
        log(f"[{STAGE}] {est.summary()}")
        return preview

    def handle(res: LLMResult) -> str | None:
        if res.case_id in done:
            return None
        if res.case_id not in sent_pos:
            return "decision not loaded for this result"
        try:
            output = OutcomesOutput.model_validate_json(res.text)
        except ValidationError as exc:
            return f"output does not match the schema: {exc}"
        allowed = sent_pos[res.case_id]
        dets: list[Determination] = []
        for d in output.determinations:
            pos = sorted({p for p in d.pos if p in allowed})
            if not pos:
                runner.log_error(
                    res.case_id, "dropped_uncited_determination", f"{d.finding} / {d.category}: {d.element}"
                )
                continue
            dets.append(
                Determination(
                    element=d.element.strip(), category=d.category, finding=d.finding, citations=[Citation(pos=p) for p in pos]
                )
            )
        append_jsonl(out_path, classification_from(res.case_id, dets, classifier=res.model or settings.llm.classifier_model))
        done.add(res.case_id)
        return None

    log(f"[{STAGE}] {len(done)} done; {len(requests)} to classify ({mode}); paragraph sources {sources}")
    runner.run(requests, handle, mode)
    return out_path


# --------------------------------------------------------------------------
# Keyword recall against the classifier
# --------------------------------------------------------------------------


def recall_report(settings: Settings) -> pd.DataFrame:
    """Keyword severe-damp flag vs the classifier, by decision year.

    The classifier's ``severe_damp`` is the reference. keyword_recall =
    both / llm_severe; keyword_precision = both / keyword_severe; agreement =
    (both + neither) / n. Only damp-flagged decisions classified by both.
    """
    screen = {r.case_id: r for r in read_jsonl(settings.work("screen.jsonl"), ScreenResult) if r.damp_flag}
    outcomes = {r.case_id: r for r in read_jsonl(settings.work("outcomes.jsonl"), OutcomeClassification)}
    rows = [
        {
            "year": screen[cid].year if screen[cid].year is not None else "unknown",
            "kw": bool(screen[cid].keyword_severe_damp),
            "llm": bool(outcomes[cid].severe_damp),
        }
        for cid in screen
        if cid in outcomes
    ]
    df = pd.DataFrame(rows, columns=["year", "kw", "llm"])

    def summarise(g: pd.DataFrame, year: Any) -> dict[str, Any]:
        both = int((g.kw & g.llm).sum())
        llm_n, kw_n, n = int(g.llm.sum()), int(g.kw.sum()), len(g)
        neither = int((~g.kw & ~g.llm).sum())
        return {
            "year": year,
            "n": n,
            "llm_severe": llm_n,
            "keyword_severe": kw_n,
            "both": both,
            "keyword_only": kw_n - both,
            "llm_only": llm_n - both,
            "neither": neither,
            "keyword_recall": both / llm_n if llm_n else float("nan"),
            "keyword_precision": both / kw_n if kw_n else float("nan"),
            "agreement": (both + neither) / n if n else float("nan"),
        }

    years = sorted({y for y in df.year if y != "unknown"}) + (["unknown"] if (df.year == "unknown").any() else [])
    table = [summarise(df[df.year == y], y) for y in years]
    table.append(summarise(df, "all"))
    report = pd.DataFrame(table)
    report.to_csv(settings.work("outputs", "severe_recall_by_year.csv"), index=False)
    return report
