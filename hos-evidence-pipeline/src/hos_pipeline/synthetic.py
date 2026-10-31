"""Synthetic work directory for tests and the end-to-end demo.

Every record validates against ``hos_pipeline.models``. Landlords, case ids
and paragraph text are invented; no real decision text is used. Two effects
are built in so the statistics have something to find:

* H3 clusters by landlord: each landlord's H3 rate is either low (~10%) or
  high (~75%).
* severe_damp depends on H1 and H2: logit = -2.0 + 2.0*H1 + 1.3*H2 + 0.1*(year-2019).
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from hos_pipeline.io_utils import write_jsonl
from hos_pipeline.models import (
    Citation,
    ClaimVerification,
    CodedDecision,
    Compensation,
    Decision,
    Determination,
    DimensionCode,
    DutyCheck,
    Event,
    HypothesisCode,
    OutcomeClassification,
    Paragraph,
    SampleRecord,
    ScreenResult,
)

COUNCILS = [
    "Exampleshire Borough Council", "Northfield City Council", "Riverton District Council",
    "Lowmoor Metropolitan Borough Council", "Eastbridge Council", "Westcombe County Council",
    "Hollowmere Borough Council", "Saltmarsh City Council", "Kingsford District Council",
    "Ashdown Borough Council",
]
ASSOCIATIONS = [
    "Northfield Housing Association", "Greenvale Homes", "Brookside Housing Trust",
    "Meadowbank Housing Association", "Oakridge Homes Group", "Harbourside Housing",
    "Fernhill Housing Association", "Copperbeech Homes", "Stonegate Housing Group",
    "Willowbrook Housing Association", "Larkspur Living", "Elmstead Housing Trust",
    "Thornbury Homes", "Silverdale Housing Association", "Pinecrest Housing Group",
]
YEARS = list(range(2019, 2027))
YEAR_SHARE = np.array([0.06, 0.08, 0.11, 0.14, 0.17, 0.19, 0.18, 0.07])
AWAAB_COMMENCEMENT = date(2025, 10, 27)
OTHER_FINDINGS = ["maladministration", "service_failure", "no_maladministration", "reasonable_redress"]


def _logit(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _fuzz_date(rng: np.random.Generator, d: date) -> tuple[str, str, date, date]:
    """Return (date_text, precision, earliest, latest) with day/month/year precision."""
    r = rng.random()
    if r < 0.7:
        return d.strftime("%d %B %Y").lstrip("0"), "day", d, d
    if r < 0.95:
        last = calendar.monthrange(d.year, d.month)[1]
        return d.strftime("%B %Y"), "month", d.replace(day=1), d.replace(day=last)
    return str(d.year), "year", date(d.year, 1, 1), date(d.year, 12, 31)


def make_synthetic_work_dir(work_dir: Path, n_decisions: int = 400, seed: int = 1) -> None:
    """Write decisions, screen, sample, coded, verified, outcomes and awaab_checks JSONL files."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    landlords = [(n, "local_authority") for n in COUNCILS] + [(n, "housing_association") for n in ASSOCIATIONS]
    size = rng.gamma(2.0, 1.0, len(landlords))
    size /= size.sum()
    h3_rate = np.where(rng.random(len(landlords)) < 0.5, 0.10, 0.75)
    h_rates = np.linspace(0.15, 0.6, 12)

    decisions, screens, latent = [], [], {}
    for i in range(n_decisions):
        year = int(rng.choice(YEARS, p=YEAR_SHARE))
        li = int(rng.choice(len(landlords), p=size))
        landlord, ltype = landlords[li]
        case_id = f"{year}{i:05d}"
        start = date(year, 1, 1)
        ddate = start + timedelta(days=int(rng.integers(0, 365 if year < 2026 else 300)))
        damp = rng.random() < 0.75
        issue = "damp and mould in the bedroom" if damp else "a delayed repair to a garden fence"
        paras = [
            Paragraph(pos=1, section="The complaint", text=f"[SYNTHETIC] The complaint is about the landlord's handling of {issue}."),
            Paragraph(pos=2, number=1, section="Background", text="[SYNTHETIC] The resident is an assured tenant of the landlord."),
            Paragraph(pos=3, number=2, section="Assessment and findings", text=f"[SYNTHETIC] The resident reported {issue} on several occasions."),
        ]
        decisions.append(Decision(case_id=case_id, landlord=landlord, title=f"Synthetic decision {case_id}",
                                  decision_date=ddate, year=year, paragraphs=paras))
        if not damp:
            screens.append(ScreenResult(case_id=case_id, landlord=landlord, year=year,
                                        mention_counts={"damp": 0, "mould": 0, "condensation": 0},
                                        complaint_definition_hit=False, determination_hit=False, damp_flag=False,
                                        rule="none", keyword_findings=["service_failure"]))
            continue
        h = rng.random(12) < h_rates
        h[2] = rng.random() < h3_rate[li]
        p_severe = _logit(-2.0 + 2.0 * h[0] + 1.3 * h[1] + 0.1 * (year - 2019))
        severe = bool(rng.random() < p_severe)
        severe_ch = bool(rng.random() < 0.08)
        kw_severe = bool(rng.random() < (0.85 if severe else 0.03))
        latent[case_id] = dict(h=h, severe=severe, severe_ch=severe_ch, landlord=landlord, ltype=ltype,
                               year=year, ddate=ddate)
        screens.append(ScreenResult(
            case_id=case_id, landlord=landlord, year=year,
            mention_counts={"damp": int(rng.integers(1, 12)), "mould": int(rng.integers(1, 10)),
                            "condensation": int(rng.integers(0, 3))},
            complaint_definition_hit=True, determination_hit=bool(rng.random() < 0.8), damp_flag=True,
            rule="complaint_definition",
            keyword_findings=["severe_maladministration"] if kw_severe else ["maladministration"],
            keyword_severe_damp=kw_severe,
        ))

    # Stratified sample of damp-flagged decisions, unequal fractions by landlord type.
    strata: dict[str, list[str]] = {}
    for cid, lat in latent.items():
        strata.setdefault(f"{lat['year']}|{lat['ltype']}", []).append(cid)
    samples = []
    for stratum, ids in sorted(strata.items()):
        frac = 0.85 if stratum.endswith("local_authority") else 0.65
        n = min(len(ids), max(5, round(frac * len(ids))))
        drawn = [ids[j] for j in rng.permutation(len(ids))[:n]]
        for order, cid in enumerate(drawn):
            samples.append(SampleRecord(case_id=cid, stratum=stratum, population_n=len(ids), sample_n=n,
                                        weight=len(ids) / n, draw_order=order))

    coded, verified, awaab = [], [], []
    for s in samples:
        lat = latent[s.case_id]
        hyps = []
        for k in range(12):
            r = rng.random()
            present = "unclear" if r < 0.05 else ("yes" if lat["h"][k] else "no")
            cites = [Citation(pos=3, quote="[SYNTHETIC] reported")] if present == "yes" else []
            hyps.append(HypothesisCode(id=f"H{k + 1}", present=present, citations=cites))
            if present == "yes":
                ok = rng.random() < 0.85
                verified.append(ClaimVerification(
                    case_id=s.case_id, claim_id=f"H{k + 1}", verdict="verified" if ok else "not_verified",
                    grade="A" if ok else "B", supporting_pos=[3] if ok else [], verifier="synthetic"))
        dims = [DimensionCode(id=f"D{k + 1}", present=("unclear" if rng.random() < 0.05 else
                                                       ("yes" if rng.random() < 0.2 + 0.04 * k else "no")))
                for k in range(13)]
        ddate = lat["ddate"]
        gap = int(rng.lognormal(6.4 if lat["severe"] else 6.0, 0.4))
        aware = ddate - timedelta(days=gap)
        a_text, a_prec, a_lo, a_hi = _fuzz_date(rng, aware)
        events = [Event(type="landlord_aware", date_text=a_text, precision=a_prec, earliest=a_lo, latest=a_hi)]
        if rng.random() < 0.6:
            done = aware + timedelta(days=int(gap * rng.uniform(0.3, 0.9)))
            w_text, w_prec, w_lo, w_hi = _fuzz_date(rng, done)
            events.append(Event(type="works_completed", date_text=w_text, precision=w_prec, earliest=w_lo, latest=w_hi))
        damp_finding = "severe_maladministration" if lat["severe"] else str(rng.choice(OTHER_FINDINGS))
        dets = [
            Determination(element="handling of damp and mould", category="damp_mould", finding=damp_finding),
            Determination(element="complaint handling", category="complaint_handling",
                          finding="severe_maladministration" if lat["severe_ch"] else "service_failure"),
        ]
        comp = []
        if lat["severe"] or rng.random() < 0.7:
            amount = float(rng.integers(1000, 5000) if lat["severe"] else rng.integers(100, 1000))
            comp.append(Compensation(amount_gbp=amount, kind="ordered", purpose="distress and inconvenience"))
        coded.append(CodedDecision(case_id=s.case_id, codebook_version="synthetic-1", coder="synthetic",
                                   hypotheses=hyps, remediation=dims, events=events, compensation=comp,
                                   determinations=dets))
        for duty, limit, unit in [("investigate_significant", "10 working days", "working_days"),
                                  ("written_summary", "3 working days", "working_days")]:
            elapsed = int(rng.integers(2, 60))
            limit_n = int(limit.split()[0])
            status = "breached" if elapsed > limit_n else "met"
            if a_prec != "day" and status == "breached" and rng.random() < 0.3:
                status = "possibly_breached"
            awaab.append(DutyCheck(
                case_id=s.case_id, duty=duty, trigger_event="landlord_aware", trigger_earliest=a_lo,
                trigger_latest=a_hi, action_event="investigation_started", action_earliest=a_lo + timedelta(days=elapsed),
                action_latest=a_hi + timedelta(days=elapsed), limit=limit, elapsed_min=max(elapsed - 2, 0),
                elapsed_max=elapsed, unit=unit, status=status, in_force=a_lo >= AWAAB_COMMENCEMENT))

    outcomes = []
    for cid, lat in latent.items():
        finding = "severe_maladministration" if lat["severe"] else "maladministration"
        outcomes.append(OutcomeClassification(
            case_id=cid,
            determinations=[Determination(element="handling of damp and mould", category="damp_mould", finding=finding)],
            severe_damp=lat["severe"], severe_complaint_handling=lat["severe_ch"],
            any_severe=lat["severe"] or lat["severe_ch"], classifier="synthetic"))

    write_jsonl(work_dir / "decisions.jsonl", decisions)
    write_jsonl(work_dir / "screen.jsonl", screens)
    write_jsonl(work_dir / "sample.jsonl", samples)
    write_jsonl(work_dir / "coded.jsonl", coded)
    write_jsonl(work_dir / "verified.jsonl", verified)
    write_jsonl(work_dir / "outcomes.jsonl", outcomes)
    write_jsonl(work_dir / "awaab_checks.jsonl", awaab)
