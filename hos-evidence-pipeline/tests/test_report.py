from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hos_pipeline.config import Paths, Settings, Stats
from hos_pipeline.denominators.rsh import run_denominators
from hos_pipeline.report.build import build_report
from hos_pipeline.synthetic import make_synthetic_work_dir

ALIASES = Path(__file__).resolve().parents[1] / "codebook" / "entity_aliases.yaml"
TABLES = ["screening_by_year", "sample_strata", "hypothesis_prevalence", "remediation_prevalence", "severe_by_year",
          "duration_comparisons", "landlord_clustering", "logistic_severe_damp", "logistic_cv_auc", "awaab_summary"]
FIGURES = ["prevalence.png", "severe_by_year.png", "duration_box.png"]


def _settings(work: Path, **paths) -> Settings:
    return Settings(paths=Paths(work_dir=work, codebook=work / "no_codebook.yaml", entity_aliases=ALIASES, **paths),
                    stats=Stats(permutations=400))


@pytest.fixture(scope="module")
def full_report(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    make_synthetic_work_dir(work, n_decisions=400, seed=1)
    summary = build_report(_settings(work))
    return work, summary


def test_build_report_writes_expected_files(full_report):
    work, summary = full_report
    out = work / "outputs"
    assert summary == out / "summary.md"
    for name in TABLES:
        assert (out / "tables" / f"{name}.csv").stat().st_size > 0, name
    for name in FIGURES:
        assert (out / "figures" / name).stat().st_size > 5000, name
    text = summary.read_text()
    assert "nan%" not in text.lower() and "| nan" not in text.lower()
    assert "failed (" not in text  # every section computed
    for heading in ["## Methods in plain English", "## Limitations", "## Reproducibility", "SYNTHETIC DATA"]:
        assert heading in text
    assert "seed: 20261030" in text and "permutations: 400" in text and "sha256=" in text


def test_report_detects_built_in_effects(full_report):
    work, _ = full_report
    tables = work / "outputs" / "tables"
    clus = pd.read_csv(tables / "landlord_clustering.csv").set_index("id")
    assert clus.loc["H3", "p_holm"] < 0.05
    assert (clus.drop(index="H3")["p_holm"] > 0.05).mean() > 0.8
    model = pd.read_csv(tables / "logistic_severe_damp.csv").set_index("term")
    assert model.loc["H1", "odds_ratio"] > 1 and model.loc["H1", "p"] < 0.05
    auc = pd.read_csv(tables / "logistic_cv_auc.csv").iloc[0]
    assert auc["auc_mean"] > 0.6
    prev = pd.read_csv(tables / "hypothesis_prevalence.csv")
    b, a = prev[prev.grade == "B"].set_index("id"), prev[prev.grade == "A"].set_index("id")
    assert (a["k"] <= b["k"]).all()  # grade A is a subset of grade B
    assert prev["w_lo"].le(prev["w_p"]).all() and prev["w_p"].le(prev["w_hi"]).all()


def test_summary_numbers_come_from_tables(full_report):
    work, summary = full_report
    text = summary.read_text()
    screening = pd.read_csv(work / "outputs" / "tables" / "screening_by_year.csv")
    total = screening[screening["year"] == "all"].iloc[0]
    assert f"Of {int(total['n_screened']):,} decisions screened, {int(total['n_damp_flagged']):,}" in text
    sev = pd.read_csv(work / "outputs" / "tables" / "severe_by_year.csv").iloc[-1]
    assert f"Cohen's kappa {sev['kappa']:.2f}" in text


def test_report_degrades_with_missing_stages(tmp_path):
    work = tmp_path / "work"
    make_synthetic_work_dir(work, n_decisions=120, seed=2)
    for name in ["coded", "sample", "verified", "awaab_checks"]:
        (work / f"{name}.jsonl").unlink()
    summary = build_report(_settings(work))
    text = summary.read_text()
    assert (work / "outputs" / "tables" / "screening_by_year.csv").exists()
    assert (work / "outputs" / "tables" / "severe_by_year.csv").exists()
    assert not (work / "outputs" / "tables" / "hypothesis_prevalence.csv").exists()
    assert "coded.jsonl missing" in text and "nan%" not in text.lower() and "failed (" not in text


def test_report_includes_landlord_rates(tmp_path):
    work = tmp_path / "work"
    make_synthetic_work_dir(work, n_decisions=150, seed=4)
    stock = tmp_path / "SDR_2024.csv"
    from hos_pipeline.synthetic import ASSOCIATIONS, COUNCILS

    pd.DataFrame({"Provider name": ASSOCIATIONS + COUNCILS, "Registration number": range(25),
                  "Total social stock owned": [500 + 900 * i for i in range(25)]}).to_csv(stock, index=False)
    settings = _settings(work, rsh_sdr=stock)
    assert run_denominators(settings) is not None
    text = build_report(settings).read_text()
    assert (work / "outputs" / "tables" / "landlord_rates.csv").exists()
    assert "Landlord rates per 1,000 homes" in text and "nan" not in text.lower().replace("tenancies", "")
