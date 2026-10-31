"""Stage 2 tests on invented decisions: keyword screening and stratified sampling."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from hos_pipeline.config import Screening, Settings
from hos_pipeline.ingest.parse import parse_decision, parse_text_decision
from hos_pipeline.io_utils import read_jsonl, write_jsonl
from hos_pipeline.models import Decision, Paragraph, SampleRecord, ScreenResult
from hos_pipeline.screen.sample import (
    allocate,
    irr_subsample,
    landlord_type,
    run_sample,
    stratified_sample,
)
from hos_pipeline.screen.screen import keyword_findings, run_screen, screen_decision

FIXTURES = Path(__file__).parent / "fixtures" / "decisions"
CFG = Screening()


def fixture(name: str, case_id: str) -> Decision:
    return parse_decision((FIXTURES / name).read_text(encoding="utf-8"), case_id)


def decision(*paras: tuple[str, str], case_id: str = "202400001", landlord: str = "Testford Council") -> Decision:
    return Decision(
        case_id=case_id, landlord=landlord, year=2024,
        paragraphs=[Paragraph(pos=i, number=i, section=s, text=t) for i, (s, t) in enumerate(paras, 1)],
    )


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("there was maladministration in its handling", ["maladministration"]),
        ("there was no maladministration in its handling", ["no_maladministration"]),
        ("there was severe maladministration in its handling", ["severe_maladministration"]),
        ("severe maladministration on repairs and maladministration on complaint handling",
         ["maladministration", "severe_maladministration"]),
        ("the failings did not amount to maladministration", ["no_maladministration"]),
        ("there was no severe maladministration", []),
        ("there was service failure in its complaint handling", ["service_failure"]),
        ("the landlord offered reasonable redress", ["reasonable_redress"]),
        ("the complaint is outside the Ombudsman’s jurisdiction", ["outside_jurisdiction"]),
        ("the complaint is outside of the Ombudsman's jurisdiction", ["outside_jurisdiction"]),
        ("this element is outside jurisdiction", ["outside_jurisdiction"]),
        ("SEVERE MALADMINISTRATION", ["severe_maladministration"]),
    ],
)
def test_keyword_findings_overlaps(text: str, expected: list[str]) -> None:
    assert keyword_findings(text) == expected


def test_screen_v1_maladministration_and_service_failure() -> None:
    r = screen_decision(fixture("v1_literal_entry_content.html", "202312345"), CFG)
    assert r.complaint_definition_hit and r.determination_hit and r.damp_flag
    assert r.rule.startswith("complaint_definition+determination")
    assert r.keyword_findings == ["service_failure", "maladministration"]
    assert not r.keyword_severe_damp
    assert r.mention_counts["mould"] >= 3 and r.mention_counts["condensation"] == 0


def test_screen_v3_no_maladministration_is_not_maladministration() -> None:
    r = screen_decision(fixture("v3_ol_continuing.html", "202154321"), CFG)
    assert r.complaint_definition_hit and r.determination_hit
    assert r.keyword_findings == ["no_maladministration", "service_failure"]


def test_screen_v4_and_text_severe_damp() -> None:
    r = screen_decision(fixture("v4_older_findings.html", "201901234"), CFG)
    assert r.complaint_definition_hit and r.determination_hit
    assert r.keyword_findings == ["no_maladministration", "severe_maladministration"]
    assert r.keyword_severe_damp
    t = parse_text_decision((FIXTURES / "exampleshire_202420001.txt").read_text(encoding="utf-8"))
    rt = screen_decision(t, CFG)
    # Severe on damp in one paragraph, plain maladministration on complaint handling in another.
    assert rt.keyword_findings == ["maladministration", "severe_maladministration"] and rt.keyword_severe_damp


def test_severe_damp_needs_both_in_one_paragraph() -> None:
    d = decision(
        ("The complaint", "The complaint is about damp in the kitchen."),
        ("Determination", "There was severe maladministration in the landlord's complaint handling."),
        ("Determination", "There was maladministration in its response to damp and mould."),
    )
    r = screen_decision(d, CFG)
    assert r.keyword_findings == ["maladministration", "severe_maladministration"]
    assert not r.keyword_severe_damp


def test_screen_rules_and_mention_threshold() -> None:
    # Damp only in Background: no complaint-definition hit; two mentions is below the threshold.
    d = decision(
        ("The complaint", "The complaint is about a broken boiler."),
        ("Background", "The resident also mentioned damp and mould once."),
        ("Determination", "There was maladministration in its handling of the boiler."),
    )
    r = screen_decision(d, CFG)
    assert (r.complaint_definition_hit, r.determination_hit, r.damp_flag, r.rule) == (False, False, False, "none")
    d.paragraphs[1].text += " Later the damp spread."
    r = screen_decision(d, CFG)
    assert r.damp_flag and r.rule == "mentions>=3"


def test_complaint_definition_stops_at_assessment() -> None:
    d = decision(
        ("The complaint", "The complaint is about repairs."),
        ("Assessment and findings", "Mould was mentioned."),
        ("The complaint", "A later section that happens to share the name mentions damp."),
    )
    assert not screen_decision(d, CFG).complaint_definition_hit


def test_determination_fallback_without_heading() -> None:
    d = decision(
        ("Summary", "The complaint is about the stairs."),
        ("Summary", "In accordance with paragraph 52 of the Scheme, there was maladministration "
                    "in the response to damp."),
    )
    r = screen_decision(d, CFG)
    assert r.determination_hit and r.keyword_findings == ["maladministration"] and r.rule == "determination"


def test_run_screen_writes_results_and_summary(tmp_path: Path) -> None:
    settings = Settings()
    settings.paths.work_dir = tmp_path
    decisions = [
        fixture("v1_literal_entry_content.html", "202312345"),
        fixture("v2_ol_start_nested.html", "202298765"),
        fixture("v4_older_findings.html", "201901234"),
        decision(("The complaint", "The complaint is about a fence."), case_id="202400777"),
    ]
    write_jsonl(settings.work("decisions.jsonl"), decisions)
    out = run_screen(settings)
    results = {r.case_id: r for r in read_jsonl(out, ScreenResult)}
    assert len(results) == 4 and not results["202400777"].damp_flag
    with settings.work("screen_summary.csv").open() as fh:
        rows = {r["year"]: r for r in csv.DictReader(fh)}
    assert rows["2020"]["keyword_severe_damp"] == "1"
    assert rows["2024"] == {"year": "2024", "decisions": "2", "damp_flagged": "1",
                            "keyword_severe_damp": "0", "keyword_severe_any": "0"}
    assert rows["total"]["decisions"] == "4" and rows["total"]["damp_flagged"] == "3"


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Exampleshire Borough Council", "local_authority"),
        ("London Borough of Testham", "local_authority"),
        ("Royal Borough of Kingsmere", "local_authority"),
        ("City of Westmere Council", "local_authority"),
        ("Northfield Housing Association", "housing_association"),
        ("Ashdown Vale Housing Trust", "housing_association"),
        ("Riverbank Homes Limited", "housing_association"),
        ("Example Living Group", "housing_association"),
        ("Homes for Haringey", "almo_tmo"),
        ("Lambeth Living", "almo_tmo"),
        ("Nottingham City Homes", "almo_tmo"),
        ("Kingsmere Tenant Management Organisation", "almo_tmo"),
        ("Example TMO", "almo_tmo"),
        ("Mr A Smith", "other"),
        ("", "other"),
    ],
)
def test_landlord_type(name: str, expected: str) -> None:
    assert landlord_type(name) == expected


LANDLORDS = {"local_authority": "Testford Council", "housing_association": "Testford Housing Association",
             "almo_tmo": "Homes for Haringey", "other": "Mr A Smith"}


def population(sizes: dict[tuple[int, str], int], unflagged: int = 0) -> list[ScreenResult]:
    out, n = [], 0
    for (year, ltype), size in sizes.items():
        for flagged in [True] * size + [False] * unflagged:
            n += 1
            out.append(ScreenResult(
                case_id=f"{year}{n:05d}", landlord=LANDLORDS[ltype], year=year, mention_counts={},
                complaint_definition_hit=flagged, determination_hit=False, damp_flag=flagged, rule="test",
            ))
    return out


SIZES = {(2021, "local_authority"): 120, (2021, "housing_association"): 300, (2022, "local_authority"): 40,
         (2022, "housing_association"): 500, (2022, "almo_tmo"): 3, (2023, "other"): 7, (2023, "almo_tmo"): 30}
STRATA = ["year", "landlord_type"]


def test_allocate_exact_with_floors_and_caps() -> None:
    sizes = {"a": 1000, "b": 10, "c": 2, "d": 333}
    alloc = allocate(sizes, 100, floor=5)
    assert sum(alloc.values()) == 100
    assert alloc["c"] == 2 and alloc["b"] >= 5
    assert allocate(sizes, 10_000, floor=5) == sizes  # capped at the population
    assert sum(allocate({"x": 3, "y": 4, "z": 5}, 7, floor=5).values()) == 7  # floors exceed target


def test_stratified_sample_exact_floors_weights() -> None:
    results = population(SIZES, unflagged=2)
    sample = stratified_sample(results, STRATA, target_n=150, seed=7, min_per_stratum=5)
    assert len(sample) == 150 and len({r.case_id for r in sample}) == 150
    flagged = {r.case_id for r in results if r.damp_flag}
    assert {r.case_id for r in sample} <= flagged
    counts = Counter(r.stratum for r in sample)
    assert counts["2022|almo_tmo"] == 3  # whole stratum: floor is min(5, size)
    assert counts["2023|other"] == 5
    assert all(n >= 5 for s, n in counts.items() if s != "2022|almo_tmo")
    for r in sample:
        assert r.sample_n == counts[r.stratum]
        assert r.weight == pytest.approx(r.population_n / r.sample_n)
    by_stratum: dict[str, list[SampleRecord]] = {}
    for r in sample:
        by_stratum.setdefault(r.stratum, []).append(r)
    assert all(sorted(r.draw_order for r in rs) == list(range(len(rs))) for rs in by_stratum.values())
    # Weighted total recovers the flagged population.
    assert sum(r.weight for r in sample) == pytest.approx(len(flagged))


def test_stratified_sample_deterministic_and_seeded() -> None:
    results = population(SIZES)
    a = stratified_sample(results, STRATA, 150, seed=7, min_per_stratum=5)
    b = stratified_sample(list(reversed(results)), STRATA, 150, seed=7, min_per_stratum=5)
    c = stratified_sample(results, STRATA, 150, seed=8, min_per_stratum=5)
    assert [r.model_dump() for r in a] == [r.model_dump() for r in b]
    assert {r.case_id for r in a} != {r.case_id for r in c}


def test_top_up_takes_the_next_draws() -> None:
    results = population({(2021, "local_authority"): 100})
    small = stratified_sample(results, STRATA, 10, seed=1, min_per_stratum=5)
    large = stratified_sample(results, STRATA, 20, seed=1, min_per_stratum=5)
    assert [r.case_id for r in sorted(large, key=lambda r: r.draw_order)][:10] == \
        [r.case_id for r in sorted(small, key=lambda r: r.draw_order)]


def test_sample_capped_at_population() -> None:
    results = population({(2021, "local_authority"): 4, (2022, "other"): 6})
    sample = stratified_sample(results, STRATA, 1667, seed=1, min_per_stratum=5)
    assert len(sample) == 10 and all(r.weight == 1.0 for r in sample)


def test_irr_subsample_size_and_spread() -> None:
    sample = stratified_sample(population(SIZES), STRATA, 150, seed=7, min_per_stratum=5)
    irr = irr_subsample(sample, 0.10, seed=3)
    assert len(irr) == 15 and len(set(irr)) == 15
    assert set(irr) <= {r.case_id for r in sample}
    assert irr == irr_subsample(sample, 0.10, seed=3)
    assert irr != irr_subsample(sample, 0.10, seed=4)
    strata = {r.case_id: r.stratum for r in sample}
    share = Counter(r.stratum for r in sample)
    got = Counter(strata[c] for c in irr)
    for s, n in share.items():
        assert abs(got[s] - 0.10 * n) < 1
    assert len(irr_subsample(sample[:25], 0.10)) == 3  # 2.5 rounds half up


def test_run_sample_files(tmp_path: Path) -> None:
    settings = Settings()
    settings.paths.work_dir = tmp_path
    settings.sampling.target_n = 60
    results = population(SIZES) + population({(2017, "local_authority"): 50})  # 2017 is out of scope
    write_jsonl(settings.work("screen.jsonl"), results)
    out = run_sample(settings)
    sample = list(read_jsonl(out, SampleRecord))
    assert len(sample) == 60 and not any(r.stratum.startswith("2017") for r in sample)
    with settings.work("sample_strata.csv").open() as fh:
        strata = list(csv.DictReader(fh))
    assert {r["stratum"] for r in strata} == {f"{y}|{t}" for y, t in SIZES}
    assert sum(int(r["sample_n"]) for r in strata) == 60
    assert strata[0].keys() >= {"year", "landlord_type", "population_n", "weight"}
    irr = settings.work("irr_sample.txt").read_text().split()
    assert len(irr) == 6
