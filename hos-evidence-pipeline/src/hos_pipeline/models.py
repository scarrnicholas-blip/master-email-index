"""Shared data contracts for every pipeline stage.

Each stage reads and writes these models as JSON Lines, so stages can be run,
re-run and audited independently. Field names here are the contract between
modules: change them only together with every reader and writer.

Paragraph references
--------------------
``Paragraph.pos`` is the 1-based position of a paragraph in the extracted text
and is what the coders cite, rendered ``[p{pos}]``. ``Paragraph.number`` is the
Ombudsman's own published paragraph number (``None`` for unnumbered text such
as the "Our approach" boilerplate). ``ingest.paragraphs`` converts between the
two so quotes can cite "paragraph 23" rather than "[p109]".

Evidence grades
---------------
A  failure claim confirmed by the strict second reader against the cited text
B  full-text coded against the codebook (LLM or human), not yet second-read
C  structured field read from the decision (determinations, orders, amounts)
D  keyword / regex screening only
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

EvidenceGrade = Literal["A", "B", "C", "D"]
DatePrecision = Literal["day", "month", "year", "unknown"]
Tristate = Literal["yes", "no", "unclear"]

# Ombudsman findings, ordered from least to most serious.
Finding = Literal[
    "outside_jurisdiction",
    "no_maladministration",
    "reasonable_redress",
    "service_failure",
    "maladministration",
    "severe_maladministration",
    "other",
]
FINDING_ORDER: tuple[str, ...] = (
    "outside_jurisdiction",
    "no_maladministration",
    "reasonable_redress",
    "service_failure",
    "maladministration",
    "severe_maladministration",
)

# What a determination is about. "damp_mould" is the element the pack compares
# across landlords; "complaint_handling" is reported separately.
ElementCategory = Literal[
    "damp_mould",
    "complaint_handling",
    "repairs_other",
    "leak_water",
    "record_keeping",
    "anti_social_behaviour",
    "other",
]


# --------------------------------------------------------------------------
# Stage 1: ingestion
# --------------------------------------------------------------------------


class Paragraph(BaseModel):
    pos: int = Field(description="1-based position in the extracted text; cited as [p{pos}]")
    number: int | None = Field(default=None, description="Official published paragraph number")
    sub: str | None = Field(default=None, description="Sub-item label within a paragraph, e.g. 'a'")
    section: str = Field(default="", description="Nearest preceding heading")
    text: str


class Decision(BaseModel):
    case_id: str
    landlord: str
    title: str = ""
    url: str | None = None
    decision_date: date | None = None
    year: int | None = Field(default=None, description="Decision year; falls back to case-id prefix")
    paragraphs: list[Paragraph]
    source_path: str | None = None
    source_sha256: str | None = None
    sha256_ok: bool | None = Field(default=None, description="Matches manifest.tsv; None if not checked")
    parse_warnings: list[str] = Field(default_factory=list)

    def text(self) -> str:
        return "\n".join(p.text for p in self.paragraphs)

    def section(self, name: str) -> list[Paragraph]:
        key = name.lower()
        return [p for p in self.paragraphs if key in p.section.lower()]


# --------------------------------------------------------------------------
# Stage 2: screening and sampling
# --------------------------------------------------------------------------


class ScreenResult(BaseModel):
    case_id: str
    landlord: str
    year: int | None
    mention_counts: dict[str, int] = Field(description="Term group -> count in the full text")
    complaint_definition_hit: bool = Field(description="Damp/mould named in the complaint definition")
    determination_hit: bool = Field(description="Damp/mould named in a determination paragraph")
    damp_flag: bool = Field(description="Passes the screening rule (grade D)")
    rule: str = Field(description="Which rule fired, for audit")
    keyword_findings: list[Finding] = Field(
        default_factory=list, description="Findings matched by regex in the Determination section (grade D)"
    )
    keyword_severe_damp: bool = False


class SampleRecord(BaseModel):
    case_id: str
    stratum: str
    population_n: int = Field(description="Damp-flagged decisions in this stratum")
    sample_n: int = Field(description="Decisions sampled from this stratum")
    weight: float = Field(description="population_n / sample_n")
    draw_order: int = Field(description="Position in the seeded random draw, for top-ups")


# --------------------------------------------------------------------------
# Stages 3-4: structured full-text coding (grade B)
# --------------------------------------------------------------------------


class Citation(BaseModel):
    pos: int = Field(description="Paragraph position [pN] in the extracted text")
    quote: str = Field(default="", description="Short verbatim quote from that paragraph")


class HypothesisCode(BaseModel):
    id: str = Field(description="Codebook id, e.g. 'H3'")
    present: Tristate
    citations: list[Citation] = Field(default_factory=list)
    rationale: str = ""


class DimensionCode(BaseModel):
    id: str = Field(description="Codebook id, e.g. 'D7'")
    present: Tristate
    value: str | None = Field(default=None, description="Free value where the codebook asks for one")
    citations: list[Citation] = Field(default_factory=list)


class Event(BaseModel):
    type: str = Field(description="Codebook event type, e.g. 'landlord_aware'")
    date_text: str = Field(description="Date exactly as written in the decision")
    precision: DatePrecision
    earliest: date | None = Field(default=None, description="First possible calendar date")
    latest: date | None = Field(default=None, description="Last possible calendar date")
    citations: list[Citation] = Field(default_factory=list)


class Compensation(BaseModel):
    amount_gbp: float
    kind: Literal["ordered", "recommended", "landlord_offered", "other"]
    purpose: str = ""
    citations: list[Citation] = Field(default_factory=list)


class Determination(BaseModel):
    element: str = Field(description="The complaint element as worded in the decision")
    category: ElementCategory
    finding: Finding
    citations: list[Citation] = Field(default_factory=list)


class CodedDecision(BaseModel):
    case_id: str
    codebook_version: str
    coder: str = Field(description="Model id or human coder initials")
    hypotheses: list[HypothesisCode]
    remediation: list[DimensionCode]
    admin_flags: dict[str, Tristate] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    compensation: list[Compensation] = Field(default_factory=list)
    determinations: list[Determination] = Field(default_factory=list)
    notes: str = ""


# --------------------------------------------------------------------------
# Stage 5: strict second reader (grade A)
# --------------------------------------------------------------------------


class ClaimVerification(BaseModel):
    case_id: str
    claim_id: str = Field(description="Hypothesis id being checked, e.g. 'H3'")
    verdict: Literal["verified", "not_verified", "insufficient_citation"]
    grade: EvidenceGrade = Field(description="'A' when verified, otherwise 'B'")
    supporting_pos: list[int] = Field(default_factory=list)
    reason: str = ""
    verifier: str = ""


# --------------------------------------------------------------------------
# Grade D replacement: outcome classifier over Determination sections
# --------------------------------------------------------------------------


class OutcomeClassification(BaseModel):
    case_id: str
    determinations: list[Determination]
    severe_damp: bool = Field(description="Severe maladministration on a damp/mould element")
    severe_complaint_handling: bool
    any_severe: bool
    classifier: str = ""


# --------------------------------------------------------------------------
# Stage 6: Awaab's Law duty checks
# --------------------------------------------------------------------------


class DutyCheck(BaseModel):
    case_id: str
    duty: str = Field(description="Duty key from the duty table, e.g. 'investigate_significant'")
    trigger_event: str
    trigger_earliest: date | None
    trigger_latest: date | None
    action_event: str
    action_earliest: date | None
    action_latest: date | None
    limit: str = Field(description="Human-readable limit, e.g. '10 working days'")
    elapsed_min: int | None = Field(default=None, description="Fewest elapsed units given date bounds")
    elapsed_max: int | None = Field(default=None, description="Most elapsed units given date bounds")
    unit: Literal["working_days", "hours", "calendar_days", "weeks"]
    status: Literal["met", "breached", "possibly_breached", "not_yet_due", "no_action_recorded", "not_applicable"]
    in_force: bool = Field(description="Trigger on or after commencement")
    note: str = ""
