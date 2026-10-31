"""Stage 6: test dated events in each coded decision against Awaab's Law duties.

Event selection
---------------
Trigger: the earliest dated event of the duty's trigger type, else of each
fallback trigger type in order. Action: the first dated event of the action type
whose date range ends on or after the trigger's earliest date, else of each
fallback action type in order. Undated events are never measured but are named
in the note.

Imprecise dates
---------------
Each event is a date interval [earliest, latest]. ``elapsed_min`` runs from
trigger.latest to action.earliest and ``elapsed_max`` from trigger.earliest to
action.latest. met: elapsed_max <= limit; breached: elapsed_min > limit;
otherwise possibly_breached. A fallback action with bound ``lower`` leaves
elapsed_max open (None); bound ``upper`` sets elapsed_min to 0.

Units
-----
working_days   WorkingCalendar convention (trigger day is day 0). An action on a
               non-working day counts as the next working day, so elapsed <= limit
               exactly when the action date <= the deadline date.
calendar_days  plain day difference.
weeks          started weeks, ceil(days / 7): elapsed <= limit iff days <= 7 * limit.
hours          calendar days x 24. Dates carry no time, so for a 24-hour limit the
               same or next calendar day is met and two or more days is breached.

With no action, the deadline (from trigger.latest) is compared with the decision
date: passed -> no_action_recorded, otherwise not_yet_due. A decision that does
not mention an action is not proof that none was taken.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml

from ..config import Settings
from ..io_utils import read_jsonl, write_jsonl
from ..models import CodedDecision, DutyCheck, Event
from .calendar import WorkingCalendar

log = logging.getLogger(__name__)

Unit = Literal["working_days", "hours", "calendar_days", "weeks"]
Bound = Literal["equivalent", "lower", "upper"]
AppliesTo = Literal["significant", "emergency"]

UNITS: tuple[str, ...] = ("working_days", "hours", "calendar_days", "weeks")
BOUNDS: tuple[str, ...] = ("equivalent", "lower", "upper")
APPLIES_TO: tuple[str, ...] = ("significant", "emergency")
STATUSES: tuple[str, ...] = (
    "met",
    "possibly_breached",
    "breached",
    "no_action_recorded",
    "not_yet_due",
    "not_applicable",
)
PERIODS: tuple[str, ...] = ("in_force", "straddles_commencement", "pre_commencement", "no_trigger")

DEFAULT_DUTIES = Path(__file__).with_name("duties.yaml")
EMERGENCY_EVENT = "emergency_identified"
AWARE_EVENT = "landlord_aware"
DETERMINATION_EVENT = "determination"

_UNIT_WORD = {"working_days": "working day", "hours": "hour", "calendar_days": "calendar day", "weeks": "week"}
_HOURS_NOTE = "hours not measurable from dates: elapsed = calendar days x 24 (same or next day met, 2+ days breached)"
_NO_ACTION_CAVEAT = "no dated action in the decision text, which is not proof that none was taken"


@dataclass(frozen=True)
class FallbackAction:
    event: str
    bound: Bound = "equivalent"


@dataclass(frozen=True)
class Duty:
    key: str
    label: str
    applies_to: AppliesTo
    trigger: str
    action: str
    limit_value: int
    limit_unit: Unit
    fallback_triggers: tuple[str, ...] = ()
    fallback_actions: tuple[FallbackAction, ...] = ()
    source: str = ""
    notes: str = ""

    @property
    def trigger_events(self) -> tuple[str, ...]:
        return (self.trigger, *self.fallback_triggers)

    @property
    def action_events(self) -> tuple[str, ...]:
        return (self.action, *(f.event for f in self.fallback_actions))

    @property
    def limit_text(self) -> str:
        word = _UNIT_WORD[self.limit_unit]
        return f"{self.limit_value} {word}{'' if self.limit_value == 1 else 's'}"

    @classmethod
    def from_dict(cls, row: dict) -> Duty:
        key = row.get("key", "?")
        limit = row.get("limit") or {}
        unit = limit.get("unit")
        if unit not in UNITS:
            raise ValueError(f"duty {key}: limit.unit must be one of {UNITS}, got {unit!r}")
        value = limit.get("value")
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"duty {key}: limit.value must be a non-negative integer, got {value!r}")
        if row.get("applies_to") not in APPLIES_TO:
            raise ValueError(f"duty {key}: applies_to must be one of {APPLIES_TO}")
        fallbacks = []
        for fb in row.get("fallback_actions") or []:
            fb = {"event": fb} if isinstance(fb, str) else fb
            bound = fb.get("bound", "equivalent")
            if bound not in BOUNDS:
                raise ValueError(f"duty {key}: fallback bound must be one of {BOUNDS}, got {bound!r}")
            fallbacks.append(FallbackAction(event=fb["event"], bound=bound))
        return cls(
            key=row["key"],
            label=row.get("label", row["key"]),
            applies_to=row["applies_to"],
            trigger=row["trigger"],
            action=row["action"],
            limit_value=value,
            limit_unit=unit,
            fallback_triggers=tuple(row.get("fallback_triggers") or ()),
            fallback_actions=tuple(fallbacks),
            source=str(row.get("source", "")),
            notes=" ".join(str(row.get("notes", "")).split()),
        )


def load_duties(path: Path | None = None) -> list[Duty]:
    """Read a duty table (default: the packaged duties.yaml)."""
    path = Path(path) if path is not None else DEFAULT_DUTIES
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get("duties", []) if isinstance(data, dict) else data
    duties = [Duty.from_dict(row) for row in rows]
    keys = [d.key for d in duties]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate duty keys in {path}")
    return duties


# --------------------------------------------------------------------------
# Date arithmetic
# --------------------------------------------------------------------------


def duty_deadline(duty: Duty, start: date, calendar: WorkingCalendar) -> date:
    """Last date on which an action meets ``duty`` when triggered on ``start``."""
    n = duty.limit_value
    if duty.limit_unit == "working_days":
        return calendar.add_working_days(start, n)
    if duty.limit_unit == "weeks":
        return start + timedelta(days=7 * n)
    if duty.limit_unit == "hours":
        return start + timedelta(days=-(-n // 24))
    return start + timedelta(days=n)


def _elapsed(unit: str, start: date, end: date, calendar: WorkingCalendar) -> int:
    if unit == "working_days":
        return calendar.working_days_between(start, calendar.roll_forward(end)) if end > start else 0
    days = max(0, (end - start).days)
    if unit == "weeks":
        return -(-days // 7)
    if unit == "hours":
        return days * 24
    return days


# --------------------------------------------------------------------------
# Event selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Span:
    type: str
    earliest: date
    latest: date
    precision: str


def _spans(events: list[Event]) -> tuple[dict[str, list[_Span]], set[str]]:
    """Dated events by type, plus the types seen only without a usable date."""
    dated: dict[str, list[_Span]] = {}
    undated: set[str] = set()
    for ev in events:
        lo, hi = ev.earliest or ev.latest, ev.latest or ev.earliest
        if lo is None or hi is None:
            undated.add(ev.type)
            continue
        lo, hi = min(lo, hi), max(lo, hi)
        dated.setdefault(ev.type, []).append(_Span(ev.type, lo, hi, ev.precision))
    return dated, undated - set(dated)


def _first(spans: list[_Span]) -> _Span:
    return min(spans, key=lambda s: (s.earliest, s.latest))


def _pick_trigger(duty: Duty, spans: dict[str, list[_Span]]) -> _Span | None:
    for etype in duty.trigger_events:
        if spans.get(etype):
            return _first(spans[etype])
    return None


def _pick_action(duty: Duty, spans: dict[str, list[_Span]], trig: _Span) -> tuple[_Span, Bound] | None:
    rules = (FallbackAction(duty.action, "equivalent"), *duty.fallback_actions)
    for rule in rules:
        after = [s for s in spans.get(rule.event, []) if s.latest >= trig.earliest]
        if after:
            return _first(after), rule.bound
    return None


# --------------------------------------------------------------------------
# Duty checks
# --------------------------------------------------------------------------


def _determination_date(spans: dict[str, list[_Span]]) -> date | None:
    dets = spans.get(DETERMINATION_EVENT)
    return _first(dets).earliest if dets else None


def check_decision(
    coded: CodedDecision,
    *,
    decision_date: date | None,
    commencement: date,
    calendar: WorkingCalendar,
    duties: list[Duty],
) -> list[DutyCheck]:
    """One DutyCheck per duty. ``decision_date`` is the as-at date for actions not
    recorded; if None, the earliest ``determination`` event date is used."""
    spans, undated = _spans(coded.events)
    as_at, as_at_note = decision_date, ""
    if as_at is None:
        as_at = _determination_date(spans)
        if as_at is not None:
            as_at_note = "decision date taken from the determination event"
    has_emergency = EMERGENCY_EVENT in spans or EMERGENCY_EVENT in undated
    aware = _first(spans[AWARE_EVENT]) if spans.get(AWARE_EVENT) else None
    return [
        _check_duty(
            coded.case_id,
            duty,
            spans,
            undated,
            has_emergency=has_emergency,
            as_at=as_at,
            as_at_note=as_at_note,
            aware=aware,
            commencement=commencement,
            calendar=calendar,
        )
        for duty in duties
    ]


def _check_duty(
    case_id: str,
    duty: Duty,
    spans: dict[str, list[_Span]],
    undated: set[str],
    *,
    has_emergency: bool,
    as_at: date | None,
    as_at_note: str,
    aware: _Span | None,
    commencement: date,
    calendar: WorkingCalendar,
) -> DutyCheck:
    notes: list[str] = []

    def result(
        status: str,
        trig: _Span | None = None,
        act: _Span | None = None,
        emin: int | None = None,
        emax: int | None = None,
        in_force: bool = False,
    ) -> DutyCheck:
        return DutyCheck(
            case_id=case_id,
            duty=duty.key,
            trigger_event=trig.type if trig else duty.trigger,
            trigger_earliest=trig.earliest if trig else None,
            trigger_latest=trig.latest if trig else None,
            action_event=act.type if act else duty.action,
            action_earliest=act.earliest if act else None,
            action_latest=act.latest if act else None,
            limit=duty.limit_text,
            elapsed_min=emin,
            elapsed_max=emax,
            unit=duty.limit_unit,
            status=status,
            in_force=in_force,
            note="; ".join(notes),
        )

    if duty.applies_to == "emergency" and not has_emergency:
        notes.append(f"no {EMERGENCY_EVENT} event: emergency duty not checked")
        return result("not_applicable")

    trig = _pick_trigger(duty, spans)
    if trig is None:
        notes.append(f"no dated {duty.trigger} event")
        notes += [f"{t} recorded without a usable date" for t in duty.trigger_events if t in undated]
        return result("not_applicable")

    if trig.type != duty.trigger:
        notes.append(f"trigger fallback: {trig.type} (no dated {duty.trigger})")
    if trig.precision != "day":
        notes.append(f"trigger date known to the {trig.precision}")

    in_force = trig.earliest >= commencement
    if not in_force and trig.latest >= commencement:
        notes.append(
            f"trigger range {trig.earliest} to {trig.latest} straddles commencement {commencement}; "
            "treated as not in force"
        )
    elif in_force and duty.trigger != AWARE_EVENT and aware is not None and aware.earliest < commencement:
        notes.append(f"landlord aware from {aware.earliest}, before commencement; check transitional provisions")

    first, last = duty_deadline(duty, trig.earliest, calendar), duty_deadline(duty, trig.latest, calendar)
    notes.append(f"deadline {first}" if first == last else f"deadline between {first} and {last}")
    if duty.limit_unit == "hours":
        notes.append(_HOURS_NOTE)

    picked = _pick_action(duty, spans, trig)
    if picked is None:
        notes += [f"{a} recorded without a usable date" for a in duty.action_events if a in undated]
        if as_at_note:
            notes.append(as_at_note)
        if as_at is not None and last >= as_at:
            notes.append(f"deadline had not passed by the decision date {as_at}")
            return result("not_yet_due", trig, in_force=in_force)
        if as_at is None:
            notes.append("decision date unknown, so the deadline cannot be confirmed as passed")
        else:
            notes.append(f"deadline passed before the decision date {as_at}")
        notes.append(_NO_ACTION_CAVEAT)
        return result("no_action_recorded", trig, in_force=in_force)

    act, bound = picked
    if act.type != duty.action:
        role = f"used as a {bound}-bound date for {duty.action}"
        if bound == "equivalent":
            role = f"counted as {duty.action}"
        notes.append(f"action fallback: {act.type} {role} (none dated on or after the trigger)")
    if act.precision != "day":
        notes.append(f"action date known to the {act.precision}")

    emin = 0 if bound == "upper" else _elapsed(duty.limit_unit, trig.latest, act.earliest, calendar)
    emax = None if bound == "lower" else _elapsed(duty.limit_unit, trig.earliest, act.latest, calendar)
    if emax is not None and emax <= duty.limit_value:
        status = "met"
    elif emin > duty.limit_value:
        status = "breached"
    else:
        status = "possibly_breached"
    return result(status, trig, act, emin, emax, in_force)


# --------------------------------------------------------------------------
# Stage runner
# --------------------------------------------------------------------------

_CASE_ID_RE = re.compile(r'"case_id"\s*:\s*"([^"\\]*)"')


def _decision_dates(path: Path, ids: set[str]) -> dict[str, date | None]:
    """Stream decisions.jsonl, parsing only the records for ``ids``."""
    out: dict[str, date | None] = {}
    if not path.exists() or not ids:
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = _CASE_ID_RE.search(line)
            if not m or m.group(1) not in ids:
                continue
            value = json.loads(line).get("decision_date")
            out[m.group(1)] = date.fromisoformat(value) if value else None
            if len(out) == len(ids):
                break
    return out


def trigger_period(check: DutyCheck, commencement: date) -> str:
    """in_force | straddles_commencement | pre_commencement | no_trigger."""
    if check.trigger_earliest is None:
        return "no_trigger"
    if check.in_force:
        return "in_force"
    if check.trigger_latest is not None and check.trigger_latest >= commencement:
        return "straddles_commencement"
    return "pre_commencement"


def summarise_checks(checks: list[DutyCheck], duties: list[Duty], commencement: date) -> pd.DataFrame:
    """Counts of each status per duty, split by when the trigger fell."""
    labels = {d.key: d.label for d in duties}
    keys = [d.key for d in duties] + sorted({c.duty for c in checks} - set(labels))
    rows = []
    for key in keys:
        for period in PERIODS:
            sel = [c for c in checks if c.duty == key and trigger_period(c, commencement) == period]
            row: dict[str, object] = {"duty": key, "label": labels.get(key, key), "period": period}
            row.update({s: sum(c.status == s for c in sel) for s in STATUSES})
            row["checks"] = len(sel)
            row["cases"] = len({c.case_id for c in sel})
            rows.append(row)
    return pd.DataFrame(rows, columns=["duty", "label", "period", *STATUSES, "checks", "cases"])


def run_awaab(settings: Settings, case_ids: set[str] | None = None, post_commencement_only: bool = False) -> Path:
    """Check every coded decision; write awaab_checks.jsonl and outputs/awaab_summary.csv.

    ``post_commencement_only`` keeps only checks whose trigger is in force.
    Returns the path of awaab_checks.jsonl.
    """
    commencement = date.fromisoformat(settings.awaab.commencement)
    bh_json = settings.paths.bank_holidays_json
    if bh_json is not None and not Path(bh_json).exists():
        log.warning("bank holidays file %s not found; using the built-in calendar", bh_json)
        bh_json = None
    calendar = WorkingCalendar(bh_json)
    duties = load_duties(settings.awaab.duties_file)

    coded: dict[str, CodedDecision] = {}
    for rec in read_jsonl(settings.work("coded.jsonl"), CodedDecision):
        if case_ids is None or rec.case_id in case_ids:
            coded[rec.case_id] = rec  # a later record for the same case supersedes an earlier one
    dates = _decision_dates(settings.work("decisions.jsonl"), set(coded))

    checks: list[DutyCheck] = []
    for case_id, rec in coded.items():
        rows = check_decision(
            rec, decision_date=dates.get(case_id), commencement=commencement, calendar=calendar, duties=duties
        )
        checks.extend(r for r in rows if r.in_force or not post_commencement_only)

    out = settings.work("awaab_checks.jsonl")
    write_jsonl(out, checks)
    summary = summarise_checks(checks, duties, commencement)
    summary.to_csv(settings.work("outputs", "awaab_summary.csv"), index=False)
    log.info("awaab: %d checks for %d decisions (calendar: %s)", len(checks), len(coded), calendar.source)
    return out


# --------------------------------------------------------------------------
# Case dossier
# --------------------------------------------------------------------------


def _md(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _event_cell(etype: str, lo: date | None, hi: date | None) -> str:
    if lo is None:
        return f"{etype}: none dated"
    return f"{etype}: {lo}" if lo == hi or hi is None else f"{etype}: {lo} to {hi}"


def _elapsed_cell(check: DutyCheck) -> str:
    if check.elapsed_min is None and check.elapsed_max is None:
        return "-"
    word = _UNIT_WORD[check.unit] + "s"
    suffix = " (from dates only)" if check.unit == "hours" else ""
    lo, hi = check.elapsed_min, check.elapsed_max
    if hi is None:
        return f"at least {lo} {word}{suffix}"
    return f"{hi} {word}{suffix}" if lo == hi else f"{lo}-{hi} {word}{suffix}"


def case_dossier(case_id: str, checks: list[DutyCheck], duties: list[Duty] | None = None) -> str:
    """Markdown duty-by-duty table for one case (strong-evidence and right-of-reply files)."""
    duties = duties if duties is not None else load_duties()
    labels = {d.key: d.label for d in duties}
    rows = [c for c in checks if c.case_id == case_id]
    lines = [f"### Awaab's Law duty checks: {case_id}", ""]
    if not rows:
        lines.append("No duty checks recorded for this case.")
        return "\n".join(lines) + "\n"
    lines += [
        "| Duty | Trigger date(s) | Action date(s) | Elapsed | Limit | Status | In force | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in rows:
        cells = [
            f"{labels.get(c.duty, c.duty)} (`{c.duty}`)",
            _event_cell(c.trigger_event, c.trigger_earliest, c.trigger_latest),
            _event_cell(c.action_event, c.action_earliest, c.action_latest),
            _elapsed_cell(c),
            c.limit,
            c.status.replace("_", " "),
            "yes" if c.in_force else "no",
            c.note,
        ]
        lines.append("| " + " | ".join(_md(x) for x in cells) + " |")
    lines += [
        "",
        "Dates are as recorded in the published decision; a range means the decision gives only a month "
        "or year. Where In force is \"no\", the trigger fell before the duties commenced and the status "
        "shows only what the timetable would have required. \"No action recorded\" means the decision "
        "records no dated action, not that none was taken.",
    ]
    if any("verify" in d.source.lower() for d in duties if d.key in {c.duty for c in rows}):
        lines.append("")
        lines.append(
            "Duty limits are taken from the evidence pack and have not yet been verified against SI 2025/1042."
        )
    return "\n".join(lines) + "\n"
