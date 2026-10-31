from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hos_pipeline.config import Paths, Settings
from hos_pipeline.denominators.rsh import (
    garwood_ci,
    load_aliases,
    load_lahs,
    load_sdr,
    match_entities,
    normalise_name,
    rates_per_1000,
    run_denominators,
)
from hos_pipeline.synthetic import ASSOCIATIONS, COUNCILS, make_synthetic_work_dir

ALIASES = Path(__file__).resolve().parents[1] / "codebook" / "entity_aliases.yaml"


def _stock() -> pd.DataFrame:
    rows = [
        ("Metropolitan Thames Valley Housing", "L0001", "sdr", 57000),
        ("Southern Housing", "L0002", "sdr", 77000),
        ("Southern Midlands Homes Limited", "L0005", "sdr", 4000),
        ("Sovereign Network Group", "L0003", "sdr", 82000),
        ("Clarion Housing Association Limited", "L0004", "sdr", 125000),
        ("Peabody Trust", "L0006", "sdr", 60000),
        ("Peabody South East Limited", "L0007", "sdr", 9000),
        ("Nottingham", "E06000018", "lahs", 25000),
        ("Haringey", "E09000014", "lahs", 15000),
        ("Lambeth", "E09000022", "lahs", 23000),
        ("Bristol, City of", "E06000023", "lahs", 27000),
    ]
    return pd.DataFrame(rows, columns=["entity", "entity_id", "source", "stock"]).assign(year=2024)


def test_normalise_name():
    assert normalise_name("The Hyde Housing Association Limited") == "hyde"
    assert normalise_name("Hyde HA") == normalise_name("HYDE Housing Association Ltd.")
    assert normalise_name("London & Quadrant Housing Trust") == "london and quadrant housing trust"
    assert normalise_name("Nottingham City Council") == "nottingham city council"  # "council" kept
    assert normalise_name("Southern Housing") != normalise_name("Southern Housing Group")
    # a generic remainder keeps its suffix rather than collapsing to one common word
    assert normalise_name("Community Housing Association") == "community housing association"
    assert normalise_name("King’s Homes plc") == "kings homes"
    assert normalise_name(None) == ""


def test_alias_file_seeds_known_cases():
    aliases = load_aliases(ALIASES)
    for name in ["MTVH", "Metropolitan Housing Trust", "Thames Valley Housing Association", "Optivo",
                 "Southern Housing Group", "Network Homes", "Sovereign Housing Association", "Nottingham City Homes",
                 "Homes for Haringey", "Lambeth Living"]:
        recs = aliases[normalise_name(name)]
        assert recs and recs[0]["reason"], name
        assert "period_from" in recs[0]
    assert aliases[normalise_name("Nottingham City Homes")][0]["canonical"] == "Nottingham City Council"


def test_match_entities_alias_exact_fuzzy_and_review(tmp_path):
    names = ["Thames Valley Housing Association", "Optivo", "Nottingham City Homes", "Homes for Haringey",
             "Clarion Housing Association", "Bristol City Council", "Peabody", "Lambeth Living", "Nowhere Homes Ltd",
             "Southern Housing Group"]
    review = tmp_path / "review.csv"
    out = match_entities(names, _stock(), load_aliases(ALIASES), review_path=review)
    got = out.set_index("hos_name")
    assert got.loc["Thames Valley Housing Association", "entity"] == "Metropolitan Thames Valley Housing"
    assert got.loc["Thames Valley Housing Association", "method"] == "alias"
    assert got.loc["Optivo", "entity"] == "Southern Housing"
    assert got.loc["Southern Housing Group", "entity"] == "Southern Housing"
    assert got.loc["Nottingham City Homes", "entity"] == "Nottingham"
    assert got.loc["Homes for Haringey", "entity"] == "Haringey"
    assert got.loc["Lambeth Living", "entity"] == "Lambeth"
    assert got.loc["Clarion Housing Association", "method"] == "exact"
    assert got.loc["Bristol City Council", "entity"] == "Bristol, City of"
    assert got.loc["Bristol City Council", "method"] == "fuzzy"
    # "Peabody" is a subset of two providers' names: never auto-resolved
    assert got.loc["Peabody", "method"] == "ambiguous" and not got.loc["Peabody", "matched"]
    assert not got.loc["Nowhere Homes Ltd", "matched"]
    flagged = set(pd.read_csv(review)["hos_name"])
    assert flagged == {"Peabody", "Nowhere Homes Ltd"}


def test_alias_period_and_conflict():
    stock = _stock()
    aliases = {"entities": [
        {"canonical": "Haringey", "source": "lahs", "stock_names": ["Haringey"], "entity_id": "E09000022",
         "reason": "deliberately wrong id", "variants": ["Bad Alias Homes"]},
        {"canonical": "Lambeth", "source": "lahs", "stock_names": ["Lambeth"], "reason": "only before 2020",
         "period": {"from": None, "to": "2019-12-31"}, "variants": ["Period Homes"]},
    ]}
    from datetime import date

    out = match_entities(["Bad Alias Homes", "Period Homes", "Period Homes"], stock, aliases,
                         dates=[None, date(2019, 5, 1), date(2021, 5, 1)])
    assert out.loc[0, "method"] == "alias_conflict" and not out.loc[0, "matched"]
    assert out.loc[1, "entity"] == "Lambeth" and out.loc[1, "method"] == "alias"
    assert out.loc[2, "method"] != "alias"


def test_garwood_known_values():
    lo, hi = garwood_ci(0)
    assert lo == 0 and hi == pytest.approx(3.689, abs=1e-3)
    lo, hi = garwood_ci(10, exposure=2.0)
    assert lo == pytest.approx(4.795 / 2, abs=1e-3) and hi == pytest.approx(18.390 / 2, abs=1e-3)


def test_rates_per_1000_ranking_threshold():
    stock = pd.DataFrame({"entity": ["A", "B", "C"], "entity_id": ["1", "2", "3"], "source": "sdr",
                          "stock": [10000, 500, 2000], "year": 2024})
    counts = pd.DataFrame({"entity": ["A", "B", "C"], "source": "sdr", "n_damp_decisions": [20, 5, 0],
                           "n_severe_damp": [2, 1, 0]})
    out = rates_per_1000(counts, stock).set_index("entity")
    assert out.loc["A", "damp_decisions_per_1000"] == pytest.approx(2.0)
    assert not out.loc["B", "ranked"] and pd.isna(out.loc["B", "damp_decisions_rank"])
    assert out.loc["A", "damp_decisions_rank"] == 1 and out.loc["C", "damp_decisions_rank"] == 2
    assert out.loc["C", "damp_decisions_per_1000_hi"] == pytest.approx(3.689 / 2, abs=1e-3)


def _write_stock_files(tmp_path: Path) -> tuple[Path, Path]:
    sdr = pd.DataFrame({"Registration number": [f"L{i:04d}" for i in range(len(ASSOCIATIONS))],
                        "Organisation name": [f"{n} Limited" for n in ASSOCIATIONS],
                        "Total social stock owned": [f"{2000 + 700 * i:,}" for i in range(len(ASSOCIATIONS))]})
    sdr_path = tmp_path / "SDR_2023-24.xlsx"
    with pd.ExcelWriter(sdr_path) as xl:
        pd.DataFrame([["Notes sheet"]]).to_excel(xl, sheet_name="Notes", header=False, index=False)
        pd.DataFrame([["Statistical Data Return 2023-24"], [None]] + [list(sdr.columns)] + sdr.values.tolist()
                     ).to_excel(xl, sheet_name="Stock", header=False, index=False)
    lahs = pd.DataFrame({"ONS code": [f"E0700{i:04d}" for i in range(len(COUNCILS))],
                         "Local authority": [c.replace(" Council", "") for c in COUNCILS],
                         "Local authority owned dwelling stock": [800 + 1500 * i for i in range(len(COUNCILS))]})
    lahs_path = tmp_path / "lahs_2024.csv"
    lahs.to_csv(lahs_path, index=False)
    return sdr_path, lahs_path


def test_loaders_detect_columns_and_fail_clearly(tmp_path):
    sdr_path, lahs_path = _write_stock_files(tmp_path)
    sdr = load_sdr(sdr_path)
    assert list(sdr.columns) == ["entity", "entity_id", "source", "stock", "year"]
    assert len(sdr) == len(ASSOCIATIONS) and sdr["year"].iloc[0] == 2024 and sdr["stock"].iloc[1] == 2700
    lahs = load_lahs(lahs_path)
    assert lahs["entity_id"].iloc[0] == "E07000000" and lahs["source"].iloc[0] == "lahs"
    with pytest.raises(ValueError, match="Available columns"):
        load_sdr(lahs_path)
    with pytest.raises(ValueError, match="stock_cols"):
        load_lahs(lahs_path, stock_cols=["No such column"])


def test_run_denominators(tmp_path):
    work = tmp_path / "work"
    make_synthetic_work_dir(work, n_decisions=200, seed=5)
    assert run_denominators(Settings(paths=Paths(work_dir=work, entity_aliases=ALIASES))) is None
    assert not (work / "outputs" / "landlord_rates.csv").exists()

    sdr_path, lahs_path = _write_stock_files(tmp_path)
    settings = Settings(paths=Paths(work_dir=work, rsh_sdr=sdr_path, lahs=lahs_path, entity_aliases=ALIASES))
    out = run_denominators(settings)
    rates = pd.read_csv(out)
    assert out == work / "outputs" / "landlord_rates.csv"
    screen = pd.read_json(work / "screen.jsonl", lines=True)
    damp = screen[screen["damp_flag"] & screen["year"].between(2019, 2025)]
    assert len(rates) == damp["landlord"].nunique()  # every synthetic landlord matched
    assert rates["n_damp_decisions"].sum() == len(damp)
    assert {"damp_decisions_per_1000", "severe_damp_per_1000_lo", "ranked"} <= set(rates.columns)
    assert (rates.loc[rates["stock"] < 1000, "ranked"] == False).all()  # noqa: E712
    assert (work / "entity_match_review.csv").exists() and (work / "entity_matches.csv").exists()
