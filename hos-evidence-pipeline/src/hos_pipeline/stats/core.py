"""Deterministic core statistics: proportions, intervals, multiplicity, agreement.

Weighting and intervals
-----------------------
The coded sample is stratified (year x landlord type) with unequal sampling
fractions, so each coded decision carries ``weight = population_n / sample_n``.
Population-weighted proportions are ratio estimates ``sum(w*y) / sum(w)``.
Their intervals use the Wilson score interval evaluated at Kish's effective
sample size ``n_eff = (sum w)^2 / sum(w^2)`` (Kish 1965; the approach Korn &
Graubard 1998 recommend for survey proportions). ``n_eff`` shrinks the nominal
n by the design effect of unequal weighting, so a sample dominated by a few
heavily weighted strata gets a correspondingly wider interval, while Wilson
keeps the interval inside [0, 1] and behaves well near 0% and 100%. No finite
population correction is applied, which makes the intervals slightly
conservative when a large share of a stratum is sampled.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

NAN = float("nan")


def _z(alpha: float) -> float:
    return float(norm.ppf(1.0 - alpha / 2.0))


def _wilson(p: float, n: float, alpha: float) -> tuple[float, float]:
    """Wilson score bounds for proportion ``p`` from (possibly non-integer) ``n``."""
    if not n or n <= 0 or not math.isfinite(p):
        return NAN, NAN
    z = _z(alpha)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(max(p * (1.0 - p) / n + z2 / (4.0 * n * n), 0.0)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Proportion ``k/n`` with its Wilson score interval: ``(p, lo, hi)``."""
    if n is None or n <= 0:
        return NAN, NAN, NAN
    if k < 0 or k > n:
        raise ValueError(f"k={k} must lie in [0, n={n}]")
    p = k / n
    lo, hi = _wilson(p, float(n), alpha)
    return p, lo, hi


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (monotone, capped at 1).

    NaNs are passed through and do not count towards the family size.
    """
    p = np.asarray(list(pvalues), dtype=float)
    out = np.full(p.shape, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    m = pv.size
    if m == 0:
        return out.tolist()
    order = np.argsort(pv, kind="mergesort")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pv[idx]))
        adj[idx] = running
    out[mask] = adj
    return out.tolist()


def _missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def cohens_kappa(rater_a: Sequence, rater_b: Sequence) -> float:
    """Cohen's kappa for two raters on nominal categories.

    Pairs where either rating is None/NaN are dropped. Returns NaN with fewer
    than two usable pairs, and 1.0 when both raters used one identical category
    throughout (kappa is otherwise 0/0 there; perfect agreement is the honest
    reading).
    """
    a_list, b_list = list(rater_a), list(rater_b)
    if len(a_list) != len(b_list):
        raise ValueError("rater_a and rater_b must have the same length")
    pairs = [(a, b) for a, b in zip(a_list, b_list) if not _missing(a) and not _missing(b)]
    n = len(pairs)
    if n < 2:
        return NAN
    po = sum(a == b for a, b in pairs) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb.get(c, 0) for c in ca) / (n * n)
    if math.isclose(pe, 1.0):
        return 1.0 if math.isclose(po, 1.0) else NAN
    return (po - pe) / (1.0 - pe)


def weighted_proportion(y: Sequence[float], w: Sequence[float], alpha: float = 0.05) -> dict:
    """Population-weighted proportion with a Wilson interval on Kish's n_eff.

    Pairs with a missing outcome or weight are dropped. Returns ``estimate``,
    ``lo``, ``hi``, ``n`` (unweighted count), ``k`` (unweighted positives),
    ``n_eff``, ``deff`` (n / n_eff) and ``sum_w`` (estimated population size).
    """
    ya = np.asarray(list(y), dtype=float)
    wa = np.asarray(list(w), dtype=float)
    if ya.shape != wa.shape:
        raise ValueError("y and w must have the same length")
    keep = ~(np.isnan(ya) | np.isnan(wa))
    ya, wa = ya[keep], wa[keep]
    if (wa < 0).any():
        raise ValueError("weights must be non-negative")
    n = int(ya.size)
    sw = float(wa.sum())
    out = {"estimate": NAN, "lo": NAN, "hi": NAN, "n": n, "k": int(np.nansum(ya > 0.5)),
           "n_eff": NAN, "deff": NAN, "sum_w": sw}
    if n == 0 or sw <= 0:
        return out
    p = float((wa * ya).sum() / sw)
    n_eff = sw * sw / float((wa * wa).sum())
    lo, hi = _wilson(p, n_eff, alpha)
    out.update(estimate=p, lo=lo, hi=hi, n_eff=n_eff, deff=n / n_eff)
    return out


def pooled_estimate(
    df: pd.DataFrame,
    outcome: str,
    weight: str = "weight",
    year_col: str = "year",
    years: Sequence[int] | None = None,
    alpha: float = 0.05,
    *,
    stratum_col: str | None = None,
    population_col: str | None = None,
) -> dict:
    """Population-weighted pooled estimate of ``outcome`` over ``years``.

    Rows with a missing outcome (e.g. "unclear") or weight are excluded. When
    ``stratum_col`` and ``population_col`` are given, weights are recomputed as
    ``population_n / n_available`` within each stratum, so strata where some
    sampled decisions are uncoded or unclear still count for their full
    population share (missing-at-random within stratum).
    """
    cols = [outcome, weight] + [c for c in (year_col, stratum_col, population_col) if c]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"pooled_estimate: missing columns {missing}")
    d = df
    if years is not None:
        d = d[d[year_col].isin(list(years))]
    d = d[d[outcome].notna() & d[weight].notna()]
    w = d[weight].astype(float)
    if stratum_col and population_col and len(d):
        n_avail = d.groupby(stratum_col)[outcome].transform("size").astype(float)
        w = d[population_col].astype(float) / n_avail
    res = weighted_proportion(d[outcome].astype(float).to_numpy(), w.to_numpy(), alpha=alpha)
    res.update(outcome=outcome, years=sorted({int(v) for v in d[year_col].dropna()}) if len(d) else [])
    return res
