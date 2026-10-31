"""Hypothesis tests: Mann-Whitney, two-proportion, landlord clustering by permutation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from hos_pipeline.config import Stats
from hos_pipeline.stats.core import holm

DEFAULT_SEED = Stats().seed
NAN = float("nan")


def _clean(v: Sequence[float]) -> np.ndarray:
    a = np.asarray(pd.to_numeric(pd.Series(list(v), dtype=object), errors="coerce"), dtype=float)
    return a[~np.isnan(a)]


def mann_whitney(x: Sequence[float], y: Sequence[float], alternative: str = "two-sided") -> dict:
    """Mann-Whitney U test (scipy, tie-corrected) with rank-biserial effect size.

    ``U`` is the statistic for ``x``; ``rank_biserial = 2U/(n_x n_y) - 1`` is
    positive when values in ``x`` tend to exceed those in ``y``.
    """
    xa, ya = _clean(x), _clean(y)
    out = {"U": NAN, "p": NAN, "n_x": int(xa.size), "n_y": int(ya.size),
           "median_x": float(np.median(xa)) if xa.size else NAN,
           "median_y": float(np.median(ya)) if ya.size else NAN,
           "rank_biserial": NAN, "alternative": alternative}
    if xa.size == 0 or ya.size == 0:
        return out
    res = sps.mannwhitneyu(xa, ya, alternative=alternative)
    u = float(res.statistic)
    out.update(U=u, p=float(res.pvalue), rank_biserial=2.0 * u / (xa.size * ya.size) - 1.0)
    return out


def two_proportion_test(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Compare two proportions: pooled z-test, or Fisher's exact test when any
    expected cell count is below 5 (where the normal approximation is poor)."""
    out = {"p1": k1 / n1 if n1 else NAN, "p2": k2 / n2 if n2 else NAN, "diff": NAN,
           "statistic": NAN, "p": NAN, "method": ""}
    if not n1 or not n2:
        return out
    out["diff"] = out["p1"] - out["p2"]
    table = np.array([[k1, n1 - k1], [k2, n2 - k2]], dtype=float)
    expected = table.sum(1, keepdims=True) * table.sum(0, keepdims=True) / table.sum()
    if (expected < 5).any():
        odds, p = sps.fisher_exact(table.astype(int))
        out.update(statistic=float(odds), p=float(p), method="fisher_exact")
        return out
    pooled = (k1 + k2) / (n1 + n2)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = out["diff"] / se if se > 0 else NAN
    out.update(statistic=float(z), p=float(2 * sps.norm.sf(abs(z))) if se > 0 else NAN, method="z_pooled")
    return out


def permutation_cluster_test(
    df: pd.DataFrame,
    outcome: str,
    group: str = "landlord",
    n_perm: int = 2000,
    seed: int = DEFAULT_SEED,
    strata: str | None = "year",
    min_group_n: int = 3,
) -> dict:
    """Does ``outcome`` cluster within ``group`` (landlords) more than chance?

    Statistic: ``S = sum_g (O_g - E_g)^2 / n_g`` where ``O_g`` is the group's
    outcome total and ``E_g`` its expectation from the stratum means. Without
    strata this is the n-weighted between-group variance of group proportions,
    ``sum_g n_g (p_g - p)^2``, i.e. Pearson's chi-square times ``p(1-p)``. The
    null distribution shuffles the outcome across decisions within strata
    (so year mix cannot masquerade as clustering);
    ``p = (1 + #{S_null >= S_obs}) / (1 + n_perm)``. Groups with fewer than
    ``min_group_n`` decisions are excluded. ``dispersion_ratio`` is
    ``S_obs / mean(S_null)``: about 1 under no clustering.
    """
    base = {"outcome": outcome, "group": group, "strata": strata, "n": 0, "n_groups": 0,
            "n_excluded_rows": 0, "statistic": NAN, "null_mean": NAN, "null_sd": NAN,
            "dispersion_ratio": NAN, "p_value": NAN, "n_perm": int(n_perm), "seed": int(seed),
            "overall_mean": NAN, "note": ""}
    cols = [outcome, group] + ([strata] if strata else [])
    d = df[cols].copy()
    d = d[d[outcome].notna() & d[group].notna()]
    sizes = d.groupby(group)[outcome].transform("size")
    n_before = len(d)
    d = d[sizes >= min_group_n]
    base["n_excluded_rows"] = n_before - len(d)
    base["n"] = len(d)
    if len(d) == 0:
        base["note"] = "no usable rows"
        return base
    y = d[outcome].astype(float).to_numpy()
    g_codes, g_levels = pd.factorize(d[group], sort=True)
    base["n_groups"] = len(g_levels)
    base["overall_mean"] = float(y.mean())
    if len(g_levels) < 2:
        base["note"] = "fewer than two groups with enough decisions"
        return base
    if strata:
        s_codes, _ = pd.factorize(d[strata].astype(object).where(d[strata].notna(), "__missing__"), sort=True)
    else:
        s_codes = np.zeros(len(d), dtype=int)
    s_mean = np.bincount(s_codes, weights=y) / np.bincount(s_codes)
    n_g = np.bincount(g_codes).astype(float)
    e_g = np.bincount(g_codes, weights=s_mean[s_codes], minlength=len(g_levels))

    def stat(values: np.ndarray) -> float:
        o_g = np.bincount(g_codes, weights=values, minlength=len(g_levels))
        return float(((o_g - e_g) ** 2 / n_g).sum())

    observed = stat(y)
    rng = np.random.default_rng(seed)
    by_stratum = np.argsort(s_codes, kind="stable")
    null = np.empty(n_perm)
    for i in range(n_perm):
        # lexsort by (stratum, random key): a uniform shuffle within each stratum.
        shuffled = np.lexsort((rng.random(len(y)), s_codes))
        perm = np.empty_like(y)
        perm[by_stratum] = y[shuffled]
        null[i] = stat(perm)
    tol = 1e-12 * max(1.0, abs(observed))
    base.update(
        statistic=observed,
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)) if n_perm > 1 else NAN,
        dispersion_ratio=observed / float(null.mean()) if null.mean() > 0 else NAN,
        p_value=(1 + int((null >= observed - tol).sum())) / (1 + n_perm),
    )
    return base


def compare_groups_holm(
    df: pd.DataFrame,
    comparisons: Sequence[Mapping[str, Any]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run a family of two-group comparisons and Holm-adjust their p-values.

    Each comparison is a mapping with keys ``value`` (column), ``group``
    (column), optional ``a``/``b`` (levels; default: the two sorted levels),
    ``test`` (``"mann_whitney"`` default, or ``"two_proportion"`` for a 0/1
    value), ``label`` and ``where`` ({column: required value}) for a subset.
    """
    rows = []
    for spec in comparisons:
        value, grp = spec["value"], spec["group"]
        test = spec.get("test", "mann_whitney")
        d = df
        for col, required in (spec.get("where") or {}).items():
            d = d[d[col] == required]
        d = d[d[grp].notna() & d[value].notna()]
        a, b = spec.get("a"), spec.get("b")
        if a is None or b is None:
            levels = sorted(d[grp].unique().tolist(), key=str)
            if len(levels) != 2:
                raise ValueError(f"{grp!r} has {len(levels)} levels; give 'a' and 'b' explicitly")
            a, b = levels
        xa = d.loc[d[grp] == a, value].astype(float)
        xb = d.loc[d[grp] == b, value].astype(float)
        row = {"label": spec.get("label") or f"{value} by {grp}: {a} vs {b}", "value": value,
               "group": grp, "a": a, "b": b, "test": test, "n_a": len(xa), "n_b": len(xb)}
        if test == "mann_whitney":
            r = mann_whitney(xa, xb)
            row.update(summary_a=r["median_x"], summary_b=r["median_y"], summary="median",
                       statistic=r["U"], effect=r["rank_biserial"], effect_name="rank_biserial", p=r["p"])
        elif test == "two_proportion":
            r = two_proportion_test(int(xa.sum()), len(xa), int(xb.sum()), len(xb))
            row.update(summary_a=r["p1"], summary_b=r["p2"], summary="proportion",
                       statistic=r["statistic"], effect=r["diff"], effect_name=f"difference ({r['method']})",
                       p=r["p"])
        else:
            raise ValueError(f"unknown test {test!r}")
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = holm(out["p"].tolist())
    out["reject_holm"] = out["p_holm"] < alpha
    return out
