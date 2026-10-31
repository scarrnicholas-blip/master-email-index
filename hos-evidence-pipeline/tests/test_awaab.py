"""Stage 6: England and Wales working-day calendar and Awaab's Law duty checks."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from hos_pipeline.awaab.calendar import WorkingCalendar, builtin_bank_holidays, easter_sunday
from hos_pipeline.awaab.duties import Duty, case_dossier, check_decision, load_duties, run_awaab
from hos_pipeline.config import Settings
from hos_pipeline.io_utils import read_jsonl, write_jsonl
from hos_pipeline.models import CodedDecision, Decision, DutyCheck, Event, Paragraph

CAL = WorkingCalendar()
DUTIES = {d.key: d for d in load_duties()}
COMMENCEMENT = date(2025, 10, 27)
D = date.fromisoformat


def ev(etype: str, lo: str | None, hi: str | None = None, precision: str = "day") -> Event:
    earliest = D(lo) if lo else None
    latest = D(hi) if hi else earliest
    return Event(type=etype, date_text=lo or "", precision=precision, earliest=earliest, latest=latest)


def month(etype: str, year: int, m: int) -> Event:
    last = (date(year + m // 12, m % 12 + 1, 1) - date.resolution).isoformat()
    return ev(etype, date(year, m, 1).isoformat(), last, precision="month")


def coded(*events: Event, case_id: str = "C1") -> CodedDecision:
    return CodedDecision(
        case_id=case_id, codebook_version="test", coder="test", hypotheses=[], remediation=[], events=list(events)
    )


def check(*events: Event, duty: str = "investigate_significant", decision_date: date | None = date(2026, 6, 1)):
    return check_decision(
        coded(*events), decision_date=decision_date, commencement=COMMENCEMENT, calendar=CAL, duties=[DUTIES[duty]]
    )[0]


# --------------------------------------------------------------------------
# Calendar
# --------------------------------------------------------------------------

KNOWN_HOLIDAYS = [
    "2019-12-25", "2019-12-26", "2020-05-08", "2021-12-27", "2021-12-28", "2022-06-02", "2022-06-03",
    "2022-09-19", "2023-05-08", "2025-04-18", "2025-04-21", "2025-05-05", "2025-05-26", "2025-08-25",
    "2025-12-25", "2025-12-26", "2026-04-03", "2026-04-06", "2026-05-04", "2026-05-25", "2026-08-31",
    "2026-12-25", "2026-12-28", "2027-12-27", "2027-12-28",
    "2011-04-29", "2012-06-04", "2012-06-05", "2022-01-03", "2022-12-26", "2022-12-27",
]  # fmt: skip


@pytest.mark.parametrize("day", KNOWN_HOLIDAYS)
def test_known_bank_holidays(day):
    d = D(day)
    assert d in CAL.bank_holidays(d.year)
    assert not CAL.is_working_day(d)


@pytest.mark.parametrize("day", ["2022-05-30", "2020-05-04", "2012-05-28", "2026-12-29", "2025-12-29"])
def test_moved_or_ordinary_days_are_working_days(day):
    d = D(day)
    assert d not in CAL.bank_holidays(d.year)
    assert CAL.is_working_day(d)


def test_regular_year_has_eight_holidays():
    assert all(len(builtin_bank_holidays(y)) == 8 for y in (2019, 2021, 2024, 2025, 2026, 2030, 2040))
    assert len(builtin_bank_holidays(2022)) == 10  # Platinum Jubilee pair and State Funeral


def test_easter_computus():
    assert easter_sunday(2011) == date(2011, 4, 24)
    assert easter_sunday(2025) == date(2025, 4, 20)
    assert easter_sunday(2026) == date(2026, 4, 5)
    assert easter_sunday(2027) == date(2027, 3, 28)
    assert easter_sunday(2038) == date(2038, 4, 25)


def test_working_days_across_christmas_2025():
    # 23, 24 Dec; 25-26 bank holidays; 29, 30, 31 Dec; 1 Jan holiday; 2, 5, 6, 7, 8 Jan.
    assert CAL.add_working_days(date(2025, 12, 22), 10) == date(2026, 1, 8)
    assert CAL.working_days_between(date(2025, 12, 22), date(2026, 1, 8)) == 10
    assert CAL.add_working_days(date(2025, 12, 24), 1) == date(2025, 12, 29)
    assert CAL.working_days_between(date(2025, 12, 24), date(2025, 12, 28)) == 0


def test_working_days_across_easter_2026():
    # 2 Apr; 3 Apr Good Friday; 6 Apr Easter Monday; 7-10 Apr; then 13-17 Apr.
    assert CAL.add_working_days(date(2026, 4, 1), 5) == date(2026, 4, 10)
    assert CAL.add_working_days(date(2026, 4, 1), 10) == date(2026, 4, 17)
    assert CAL.working_days_between(date(2026, 4, 2), date(2026, 4, 7)) == 1


def test_working_day_edges():
    sat = date(2025, 11, 8)
    assert CAL.add_working_days(sat, 0) == sat
    assert CAL.add_working_days(sat, 1) == date(2025, 11, 10)
    assert CAL.working_days_between(date(2025, 11, 10), date(2025, 11, 10)) == 0
    assert CAL.working_days_between(date(2025, 11, 10), date(2025, 11, 3)) == 0
    assert CAL.roll_forward(date(2025, 12, 25)) == date(2025, 12, 29)
    with pytest.raises(ValueError):
        CAL.add_working_days(sat, -1)


def test_govuk_json_override(tmp_path):
    # Covers 2025 only: adds an invented 2 June, omits the summer holiday, Christmas and Good Friday.
    events = [{"title": t, "date": d} for t, d in [
        ("New Year", "2025-01-01"), ("Easter Monday", "2025-04-21"), ("Early May", "2025-05-05"),
        ("Spring", "2025-05-26"), ("Test day", "2025-06-02"), ("Boxing Day", "2025-12-26"),
    ]]  # fmt: skip
    path = tmp_path / "bank-holidays.json"
    path.write_text(json.dumps({"england-and-wales": {"division": "england-and-wales", "events": events}}))
    cal = WorkingCalendar(path)
    assert not cal.is_working_day(date(2025, 6, 2))  # from the file
    assert cal.is_working_day(date(2025, 8, 25))  # file takes precedence for a covered year
    assert not cal.is_working_day(date(2025, 12, 25))  # Christmas Day always excluded
    assert not cal.is_working_day(date(2025, 4, 18))  # Good Friday always excluded
    assert not cal.is_working_day(date(2026, 8, 31))  # 2026 not in the file: built-in rules
    assert cal.bank_holidays(2026) == CAL.bank_holidays(2026)


# --------------------------------------------------------------------------
# Duty table
# --------------------------------------------------------------------------


def test_default_duty_table():
    assert set(DUTIES) == {
        "emergency_24h", "investigate_significant", "written_summary", "safety_work_begin", "preventive_work_begin",
    }  # fmt: skip
    assert DUTIES["investigate_significant"].limit_text == "10 working days"
    assert DUTIES["emergency_24h"].limit_unit == "hours"
    assert DUTIES["preventive_work_begin"].limit_text == "12 weeks"
    assert all("SI 2025/1042" in d.source for d in DUTIES.values())


def test_invalid_duty_table(tmp_path):
    path = tmp_path / "duties.yaml"
    path.write_text(
        "duties:\n  - {key: x, applies_to: significant, trigger: a, action: b, limit: {value: 3, unit: days}}\n"
    )
    with pytest.raises(ValueError, match="unit"):
        load_duties(path)


# --------------------------------------------------------------------------
# Status paths
# --------------------------------------------------------------------------


def test_met_on_deadline_day():
    c = check(ev("landlord_aware", "2025-11-03"), ev("investigation_concluded", "2025-11-17"))
    assert (c.status, c.elapsed_min, c.elapsed_max, c.in_force) == ("met", 10, 10, True)
    assert "deadline 2025-11-17" in c.note


def test_breached_day_after_deadline():
    c = check(ev("landlord_aware", "2025-11-03"), ev("investigation_concluded", "2025-11-18"))
    assert (c.status, c.elapsed_min, c.elapsed_max) == ("breached", 11, 11)


def test_action_on_weekend_after_deadline_is_breached():
    # Aware Friday 7 Nov: deadline Friday 21 Nov; Saturday 22 Nov counts as Monday 24 Nov.
    assert check(ev("landlord_aware", "2025-11-07"), ev("investigation_concluded", "2025-11-21")).status == "met"
    c = check(ev("landlord_aware", "2025-11-07"), ev("investigation_concluded", "2025-11-22"))
    assert (c.status, c.elapsed_max) == ("breached", 11)


def test_possibly_breached_with_month_precision():
    c = check(ev("landlord_aware", "2025-11-03"), month("investigation_concluded", 2025, 11))
    assert c.status == "possibly_breached"
    assert c.elapsed_min == 0 and c.elapsed_max == 20  # 30 Nov is a Sunday, counted as Monday 1 Dec
    assert "known to the month" in c.note


def test_breached_even_with_month_precision():
    c = check(month("landlord_aware", 2025, 11), month("investigation_concluded", 2026, 2))
    assert c.status == "breached"
    assert c.elapsed_min > 10
    assert "deadline between" in c.note


def test_no_action_recorded():
    c = check(ev("landlord_aware", "2025-11-03"), decision_date=date(2026, 3, 1))
    assert c.status == "no_action_recorded"
    assert c.elapsed_min is None and c.elapsed_max is None
    assert "not proof" in c.note


def test_not_yet_due_until_deadline_passes():
    aware = ev("landlord_aware", "2025-11-03")
    assert check(aware, decision_date=date(2025, 11, 10)).status == "not_yet_due"
    assert check(aware, decision_date=date(2025, 11, 17)).status == "not_yet_due"
    assert check(aware, decision_date=date(2025, 11, 18)).status == "no_action_recorded"


def test_decision_date_falls_back_to_determination_event():
    c = check(ev("landlord_aware", "2025-11-03"), ev("determination", "2025-11-12"), decision_date=None)
    assert c.status == "not_yet_due" and "determination" in c.note
    c = check(ev("landlord_aware", "2025-11-03"), decision_date=None)
    assert c.status == "no_action_recorded" and "decision date unknown" in c.note


def test_not_applicable_without_trigger():
    c = check(ev("investigation_concluded", "2025-11-10"))
    assert (c.status, c.in_force, c.trigger_earliest) == ("not_applicable", False, None)
    c = check(ev("landlord_aware", None), ev("investigation_concluded", "2025-11-10"))
    assert c.status == "not_applicable" and "without a usable date" in c.note


def test_in_force_before_after_and_straddling_commencement():
    action = ev("investigation_concluded", "2025-11-28")
    assert check(ev("landlord_aware", "2025-10-24"), action).in_force is False
    assert check(ev("landlord_aware", "2025-10-27"), action).in_force is True
    c = check(month("landlord_aware", 2025, 10), action)
    assert c.in_force is False and "straddles commencement" in c.note


def test_transitional_note_when_awareness_precedes_commencement():
    c = check(
        ev("landlord_aware", "2025-10-01"),
        ev("investigation_concluded", "2025-11-03"),
        ev("written_summary_sent", "2025-11-05"),
        duty="written_summary",
    )
    assert c.in_force is True and c.status == "met" and "transitional" in c.note


def test_written_summary_across_christmas():
    concluded = ev("investigation_concluded", "2025-12-19")  # Friday
    assert check(concluded, ev("written_summary_sent", "2025-12-24"), duty="written_summary").status == "met"
    c = check(concluded, ev("written_summary_sent", "2025-12-29"), duty="written_summary")
    assert (c.status, c.elapsed_max) == ("breached", 4)


def test_earliest_trigger_and_first_action_after_it():
    c = check(
        ev("landlord_aware", "2025-12-01"),
        ev("landlord_aware", "2025-11-03"),
        ev("investigation_concluded", "2025-10-01"),  # before the trigger: ignored
        ev("investigation_concluded", "2025-12-15"),
        ev("investigation_concluded", "2025-11-10"),
    )
    assert c.trigger_earliest == date(2025, 11, 3)
    assert c.action_earliest == date(2025, 11, 10)
    assert c.status == "met"


def test_lower_bound_fallback_can_only_show_breach():
    aware = ev("landlord_aware", "2025-11-03")
    c = check(aware, ev("investigation_started", "2025-11-05"))
    assert (c.status, c.action_event, c.elapsed_max) == ("possibly_breached", "investigation_started", None)
    assert "lower-bound date for investigation_concluded" in c.note
    assert check(aware, ev("investigation_started", "2025-12-01")).status == "breached"


def test_upper_bound_fallback_can_only_show_met():
    concluded = ev("investigation_concluded", "2025-11-03")
    c = check(concluded, ev("safety_work_completed", "2025-11-06"), duty="safety_work_begin")
    assert (c.status, c.elapsed_min) == ("met", 0)
    c = check(concluded, ev("safety_work_completed", "2025-12-01"), duty="safety_work_begin")
    assert c.status == "possibly_breached"


def test_fallback_trigger():
    duty = Duty(
        key="t", label="t", applies_to="significant", trigger="a", action="b",
        limit_value=3, limit_unit="calendar_days", fallback_triggers=("a2",),
    )  # fmt: skip
    rec = coded(ev("a2", "2025-11-03"), ev("b", "2025-11-06"))
    c = check_decision(rec, decision_date=None, commencement=COMMENCEMENT, calendar=CAL, duties=[duty])[0]
    assert (c.status, c.trigger_event, c.elapsed_max) == ("met", "a2", 3)
    assert "trigger fallback" in c.note


def test_hours_rule():
    found = ev("emergency_identified", "2025-11-03")
    same = check(found, ev("emergency_action", "2025-11-03"), duty="emergency_24h")
    nxt = check(found, ev("emergency_action", "2025-11-04"), duty="emergency_24h")
    late = check(found, ev("emergency_action", "2025-11-05"), duty="emergency_24h")
    assert (same.status, same.elapsed_max) == ("met", 0)
    assert (nxt.status, nxt.elapsed_max) == ("met", 24)
    assert (late.status, late.elapsed_min) == ("breached", 48)
    assert "hours not measurable" in late.note
    fallback = check(found, ev("decant_started", "2025-11-04"), duty="emergency_24h")
    assert (fallback.status, fallback.action_event) == ("met", "decant_started")


def test_emergency_duty_needs_emergency_event():
    c = check(ev("landlord_aware", "2025-11-03"), ev("emergency_action", "2025-11-03"), duty="emergency_24h")
    assert c.status == "not_applicable" and "emergency_identified" in c.note


def test_weeks_counted_as_started_weeks():
    concluded = ev("investigation_concluded", "2025-11-03")
    c = check(concluded, ev("preventive_work_started", "2026-01-26"), duty="preventive_work_begin")
    assert (c.status, c.elapsed_max) == ("met", 12)  # 84 days
    c = check(concluded, ev("preventive_work_started", "2026-01-27"), duty="preventive_work_begin")
    assert (c.status, c.elapsed_min) == ("breached", 13)  # 85 days


def test_every_duty_reported():
    checks = check_decision(
        coded(ev("landlord_aware", "2025-11-03")),
        decision_date=date(2026, 3, 1),
        commencement=COMMENCEMENT,
        calendar=CAL,
        duties=list(DUTIES.values()),
    )
    assert [c.duty for c in checks] == list(DUTIES)
    assert all(isinstance(c, DutyCheck) for c in checks)


# --------------------------------------------------------------------------
# Dossier and stage runner
# --------------------------------------------------------------------------


def test_case_dossier_renders():
    checks = check_decision(
        coded(
            ev("landlord_aware", "2025-11-03"),
            month("investigation_concluded", 2025, 11),
            ev("written_summary_sent", "2026-01-20"),
        ),
        decision_date=date(2026, 3, 1),
        commencement=COMMENCEMENT,
        calendar=CAL,
        duties=list(DUTIES.values()),
    )
    md = case_dossier("C1", checks)
    lines = md.splitlines()
    assert lines[0] == "### Awaab's Law duty checks: C1"
    assert lines[2].startswith("| Duty | Trigger date(s) | Action date(s) | Elapsed | Limit | Status |")
    assert sum(line.startswith("| ") for line in lines) == 1 + len(DUTIES)
    assert "investigation_concluded: 2025-11-01 to 2025-11-30" in md
    assert "0-20 working days" in md
    assert "possibly breached" in md and "not applicable" in md
    assert "not yet been verified against SI 2025/1042" in md
    assert "No duty checks" in case_dossier("other", checks)


def test_run_awaab(tmp_path):
    settings = Settings()
    settings.paths.work_dir = tmp_path
    stale = coded(ev("landlord_aware", "2025-11-03"), case_id="A")
    write_jsonl(
        settings.work("coded.jsonl"),
        [
            stale,
            coded(ev("landlord_aware", "2025-11-03"), ev("investigation_concluded", "2025-11-10"), case_id="A"),
            coded(ev("landlord_aware", "2024-03-04"), case_id="B"),
            coded(ev("landlord_aware", "2025-11-03"), ev("determination", "2025-11-05"), case_id="C"),
        ],
    )
    para = [Paragraph(pos=1, text='The "case_id": "B" text is quoted here.')]
    write_jsonl(
        settings.work("decisions.jsonl"),
        [
            Decision(case_id="A", landlord="L", decision_date=date(2026, 2, 1), paragraphs=para),
            Decision(case_id="B", landlord="L", decision_date=date(2025, 1, 6), paragraphs=para),
            Decision(case_id="Z", landlord="L", decision_date=date(2025, 1, 6), paragraphs=para),
        ],
    )
    out = run_awaab(settings)
    checks = list(read_jsonl(out, DutyCheck))
    assert len(checks) == 3 * len(DUTIES)
    status = {(c.case_id, c.duty): c.status for c in checks}
    assert status["A", "investigate_significant"] == "met"  # later record supersedes the stale one
    assert status["B", "investigate_significant"] == "no_action_recorded"
    assert status["C", "investigate_significant"] == "not_yet_due"  # as at the determination event

    summary = pd.read_csv(tmp_path / "outputs" / "awaab_summary.csv")
    row = summary[(summary.duty == "investigate_significant") & (summary.period == "in_force")].iloc[0]
    assert (row.met, row.not_yet_due, row.checks) == (1, 1, 2)
    row = summary[(summary.duty == "investigate_significant") & (summary.period == "pre_commencement")].iloc[0]
    assert (row.no_action_recorded, row.cases) == (1, 1)

    run_awaab(settings, case_ids={"A", "B"}, post_commencement_only=True)
    checks = list(read_jsonl(out, DutyCheck))
    assert checks and all(c.in_force and c.case_id == "A" for c in checks)
