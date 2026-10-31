"""Load and render the coding codebook (codebook/codebook.yaml).

The rendered markdown is the cached part of the coder and verifier system
prompts, so ``render_markdown`` must be byte-for-byte deterministic: it follows
the YAML order and never adds timestamps or other run-specific text.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import get_args

import yaml
from pydantic import BaseModel, Field, model_validator

from ..models import ElementCategory, Finding

HYPOTHESIS_IDS: tuple[str, ...] = tuple(f"H{i}" for i in range(1, 13))
DIMENSION_IDS: tuple[str, ...] = tuple(f"D{i}" for i in range(1, 14))

# Fixed vocabulary consumed by the Awaab's Law engine from CodedDecision.events.
EVENT_TYPES: tuple[str, ...] = (
    "first_report",
    "landlord_aware",
    "emergency_identified",
    "emergency_action",
    "investigation_started",
    "investigation_concluded",
    "written_summary_sent",
    "safety_work_started",
    "safety_work_completed",
    "preventive_work_started",
    "decant_offered",
    "decant_started",
    "works_completed",
    "complaint_made",
    "stage1_response",
    "stage2_request",
    "stage2_response",
    "ombudsman_referral",
    "determination",
)

SECTIONS: tuple[str, ...] = (
    "rules",
    "hypotheses",
    "dimensions",
    "admin_flags",
    "event_types",
    "element_categories",
    "findings",
)

_HEADINGS = {
    "rules": "General coding rules",
    "hypotheses": "Failure hypotheses (code each: yes / no / unclear)",
    "dimensions": "Remediation dimensions (code each: yes / no / unclear)",
    "admin_flags": "Administration flags (code each: yes / no / unclear)",
    "event_types": "Event types (dated events; use these keys only)",
    "element_categories": "Element categories (what a determination is about)",
    "findings": "Findings (the Ombudsman's outcome for an element)",
}


class CodebookRule(BaseModel):
    id: str
    text: str


class CodebookItem(BaseModel):
    id: str
    label: str
    definition: str
    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    value_type: str | None = None


class Codebook(BaseModel):
    version: str
    rules: list[CodebookRule] = Field(default_factory=list)
    hypotheses: list[CodebookItem]
    dimensions: list[CodebookItem]
    admin_flags: list[CodebookItem] = Field(default_factory=list)
    event_types: list[CodebookItem] = Field(default_factory=list)
    element_categories: list[CodebookItem] = Field(default_factory=list)
    findings: list[CodebookItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "Codebook":
        problems: list[str] = []
        for name in SECTIONS:
            ids = [x.id for x in getattr(self, name)]
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            if dupes:
                problems.append(f"{name}: duplicate ids {dupes}")
        coded = [x.id for x in self.hypotheses + self.dimensions + self.admin_flags]
        clash = sorted({i for i in coded if coded.count(i) > 1})
        if clash:
            problems.append(f"ids used in more than one coded section: {clash}")

        def exact(name: str, expected: Sequence[str]) -> None:
            got = [x.id for x in getattr(self, name)]
            missing = [i for i in expected if i not in got]
            extra = [i for i in got if i not in expected]
            if missing:
                problems.append(f"{name}: missing {missing}")
            if extra:
                problems.append(f"{name}: unexpected {extra}")

        exact("hypotheses", HYPOTHESIS_IDS)
        exact("dimensions", DIMENSION_IDS)
        exact("event_types", EVENT_TYPES)
        exact("element_categories", get_args(ElementCategory))
        exact("findings", get_args(Finding))
        if problems:
            raise ValueError("invalid codebook: " + "; ".join(problems))
        return self

    # ------------------------------------------------------------------ lookups

    @property
    def hypothesis_ids(self) -> list[str]:
        return [x.id for x in self.hypotheses]

    @property
    def dimension_ids(self) -> list[str]:
        return [x.id for x in self.dimensions]

    @property
    def admin_flag_ids(self) -> list[str]:
        return [x.id for x in self.admin_flags]

    @property
    def event_type_ids(self) -> list[str]:
        return [x.id for x in self.event_types]

    def item(self, item_id: str) -> CodebookItem:
        for name in SECTIONS[1:]:
            for x in getattr(self, name):
                if x.id == item_id:
                    return x
        raise KeyError(item_id)

    # ---------------------------------------------------------------- rendering

    def render_markdown(self, sections: Sequence[str] | None = None) -> str:
        """Render the chosen sections (default: all) as stable markdown."""
        chosen = list(SECTIONS if sections is None else sections)
        unknown = [s for s in chosen if s not in SECTIONS]
        if unknown:
            raise ValueError(f"unknown codebook sections: {unknown}")
        out = [f"# Codebook (version {self.version})"]
        for name in SECTIONS:  # fixed order regardless of how `sections` is ordered
            if name not in chosen:
                continue
            out.append(f"\n## {_HEADINGS[name]}\n")
            if name == "rules":
                out.extend(f"- **{r.id}.** {_clean(r.text)}" for r in self.rules)
            else:
                out.extend(render_item(x) for x in getattr(self, name))
        return "\n".join(out).rstrip() + "\n"


def render_item(item: CodebookItem) -> str:
    """One codebook item as a markdown block."""
    lines = [f"### {item.id}: {item.label}", "", _clean(item.definition)]
    if item.value_type:
        lines += ["", f"Value: {item.value_type}."]
    if item.include:
        lines += ["", "Include:"] + [f"- {_clean(x)}" for x in item.include]
    if item.exclude:
        lines += ["", "Exclude:"] + [f"- {_clean(x)}" for x in item.exclude]
    return "\n".join(lines) + "\n"


def _clean(text: str) -> str:
    return " ".join(str(text).split())


def load_codebook(path: str | Path) -> Codebook:
    """Read and validate the codebook YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Codebook.model_validate(data)
