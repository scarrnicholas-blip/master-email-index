from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.multitest import multipletests

from hos_pipeline.config import Paths, Settings
from hos_pipeline.stats.core import cohens_kappa, holm, pooled_estimate, weighted_proportion, wilson_ci
from hos_pipeline.stats.models import cv_auc, fit_logistic
from hos_pipeline.stats.table import build_analysis_table
from hos_pipeline.stats.tests import compare_groups_holm, mann_whitney, permutation_cluster_test
from hos_pipeline.synthetic import make_synthetic_work_dir


def test_wilson_known_values():
    p, lo, hi = wilson_ci(5, 10)
    assert p == 0.5
    assert lo == pytest.approx(0.2366, abs=1e-4)
    assert hi == pytest.approx(0.7634, abs=1e-4)
    p, lo, hi = wilson_ci(0, 10)
    assert (p, lo) == (0.0, 0.0)
    assert hi == pytest.approx(0.2775, abs=1e-4)
    assert all(math.isnan(v) for v in wilson_ci(0, 0))


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.uniform(0, 0.2, size=rng.integers(1, 15))
        expected = multipletests(p, method="holm")[1]
        assert holm(p) == pytest.approx(expected.tolist())
    out = holm([0.01, float("nan"), 0.04])
    assert math.isnan(out[1])
    assert out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.04)
    assert holm([0.3, 0.2]) == pytest.approx([0.4, 0.4])  # step-down stays monotone
    assert holm([0.9, 0.8]) == [1.0, 1.0]  # capped at 1


def test_kappa_matches_sklearn_and_edge_cases():
    rng = np.random.default_rng(1)
    a = rng.choice(["yes", "no", "unclear"], 200)
    b = np.where(rng.random(200) < 0.7, a, rng.choice(["yes", "no", "unclear"], 200))
    assert cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    assert cohens_kappa(["x", None, "y", "x"], ["x", "y", float("nan"), "x"]) == 1.0  # two usable pairs, constant
    assert math.isnan(cohens_kappa(["x", None], ["x", "y"]))
    assert cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0


def test_weighted_proportion_equal_weights_is_plain_proportion():
    y = [1, 0, 1, 1, 0, 0, 0, 1, 0, 0]
    res = weighted_proportion(y, [3.0] * len(y))
    p, lo, hi = wilson_ci(4, 10)
    assert res["estimate"] == pytest.approx(p)
    assert res["n_eff"] == pytest.approx(10)
    assert (res["lo"], res["hi"]) == pytest.approx((lo, hi))


def test_weighted_proportion_unequal_weights_widen_interval():
    y = np.r_[np.ones(50), np.zeros(50)]
    w = np.r_[np.full(50, 1.0), np.full(50, 9.0)]
    res = weighted_proportion(y, w)
    assert res["estimate"] == pytest.approx(0.1)
    assert res["n_eff"] < 100 and res["deff"] > 1


def test_pooled_estimate_filters_years_and_recalibrates():
    df = pd.DataFrame({
        "year": [2019] * 4 + [2020] * 4 + [2026] * 2,
        "stratum": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
        "population_n": [40] * 4 + [8] * 4 + [10] * 2,
        "weight": [10.0] * 4 + [2.0] * 4 + [5.0] * 2,
        "H1": [1, 1, np.nan, 0, 0, 0, 0, 1, 1, 1],
    })
    plain = pooled_estimate(df, "H1", years=[2019, 2020])
    assert plain["n"] == 7 and plain["years"] == [2019, 2020]
    recal = pooled_estimate(df, "H1", years=[2019, 2020], stratum_col="stratum", population_col="population_n")
    # stratum a: 2/3 yes of 40; stratum b: 1/4 yes of 8 -> (40*2/3 + 8*1/4) / 48
    assert recal["estimate"] == pytest.approx((40 * 2 / 3 + 2) / 48)


def _cluster_frame(rng, strong: bool) -> pd.DataFrame:
    rows = []
    for g in range(20):
        rate = (0.9 if g % 2 else 0.1) if strong else 0.4
        for _ in range(15):
            rows.append({"landlord": f"L{g}", "year": int(rng.integers(2019, 2026)), "y": float(rng.random() < rate)})
    return pd.DataFrame(rows)


def test_permutation_cluster_detects_clustering_and_is_deterministic():
    rng = np.random.default_rng(2)
    strong = _cluster_frame(rng, True)
    r1 = permutation_cluster_test(strong, "y", n_perm=500, seed=7)
    assert r1["p_value"] < 0.01 and r1["dispersion_ratio"] > 3
    r2 = permutation_cluster_test(strong, "y", n_perm=500, seed=7)
    assert r1 == r2
    null = _cluster_frame(np.random.default_rng(3), False)
    r3 = permutation_cluster_test(null, "y", n_perm=500, seed=7, strata=None)
    assert r3["p_value"] > 0.05
    assert 1 / 501 <= r3["p_value"] <= 1


def test_permutation_cluster_min_group_n():
    df = pd.DataFrame({"landlord": ["a"] * 5 + ["b"] * 5 + ["c"] * 2, "y": [1, 0] * 6, "year": 2020})
    r = permutation_cluster_test(df, "y", n_perm=50, min_group_n=3)
    assert r["n_groups"] == 2 and r["n_excluded_rows"] == 2


def test_mann_whitney_matches_scipy():
    rng = np.random.default_rng(4)
    x, y = rng.normal(1, 1, 40), rng.normal(0, 1, 30)
    res = mann_whitney(list(x) + [float("nan")], y)
    ref = mannwhitneyu(x, y, alternative="two-sided")
    assert res["U"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
    assert res["n_x"] == 40
    assert res["rank_biserial"] == pytest.approx(2 * ref.statistic / (40 * 30) - 1)
    assert res["rank_biserial"] > 0


def test_compare_groups_holm():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"g": ["a"] * 50 + ["b"] * 50, "v": np.r_[rng.normal(2, 1, 50), rng.normal(0, 1, 50)],
                       "flag": np.r_[rng.random(50) < 0.6, rng.random(50) < 0.2].astype(float)})
    out = compare_groups_holm(df, [{"value": "v", "group": "g"},
                                   {"value": "flag", "group": "g", "test": "two_proportion"}])
    assert list(out["p_holm"]) == pytest.approx(holm(out["p"]))
    assert out["reject_holm"].all()


def test_fit_logistic_recovers_positive_coefficient():
    rng = np.random.default_rng(6)
    n = 800
    df = pd.DataFrame({"x": rng.normal(size=n), "grp": rng.choice(["a", "b", "c"], n)})
    df["y"] = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + 1.2 * df["x"])))).astype(int)
    out = fit_logistic(df, "y", ["x"], categorical=["grp"])
    row = out.set_index("term").loc["x"]
    assert row["coef"] == pytest.approx(1.2, abs=0.3)
    assert row["or_lo"] > 1 and row["p"] < 0.001
    assert {"grp=b", "grp=c"} <= set(out["term"]) and out["converged"].all()
    weighted = fit_logistic(df.assign(w=rng.uniform(1, 4, n)), "y", ["x"], weights="w")
    assert weighted.set_index("term").loc["x", "odds_ratio"] > 2
    assert weighted["method"].iloc[0].startswith("glm")


def test_fit_logistic_reports_separation():
    df = pd.DataFrame({"x": [0.0] * 20 + [1.0] * 20, "y": [0] * 20 + [1] * 20})
    out = fit_logistic(df, "y", ["x"])
    assert "separation" in out["warning"].iloc[0]
    assert not out["converged"].iloc[0]


def test_cv_auc_separable_and_guard():
    rng = np.random.default_rng(8)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "t": rng.choice(["u", "v"], n)})
    df["y"] = (df["x"] + rng.normal(0, 0.5, n) > 0).astype(int)
    res = cv_auc(df, "y", ["x"], categorical=["t"], folds=5, seed=1, repeats=2)
    assert res["auc_mean"] > 0.7 and len(res["auc_folds"]) == 10
    assert cv_auc(df, "y", ["x"], seed=1) == cv_auc(df, "y", ["x"], seed=1)
    tiny = pd.DataFrame({"x": range(10), "y": [1, 1] + [0] * 8})
    assert cv_auc(tiny, "y", ["x"], folds=5)["folds"] == 2


def test_analysis_table_from_synthetic(tmp_path):
    make_synthetic_work_dir(tmp_path, n_decisions=150, seed=3)
    settings = Settings(paths=Paths(work_dir=tmp_path))
    df = build_analysis_table(settings)
    assert (tmp_path / "analysis_table.csv").exists()
    assert len(df) == sum(1 for _ in open(tmp_path / "coded.jsonl"))
    for col in ["H1", "H12", "H1_A", "D13", "severe_damp", "finding_damp", "duration_days", "duration_precision",
                "n_awaab_breached", "total_compensation_ordered", "weight", "landlord_type"]:
        assert col in df.columns
    assert set(df["H1"].dropna().unique()) <= {0.0, 1.0}
    assert (df["H3_A"] <= df["H3"].fillna(0)).all()  # verified claims are a subset of coded "yes"
    assert (df["duration_days"].dropna() >= df["duration_days_min"].dropna()).all()
    assert df["landlord_type"].notna().all() and df["landlord_type"].nunique() >= 2
