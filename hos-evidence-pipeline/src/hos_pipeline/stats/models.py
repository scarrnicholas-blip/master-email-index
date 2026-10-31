"""Logistic regression with inference (statsmodels) and cross-validated AUC (sklearn).

Weighted fits use a Binomial GLM with ``var_weights`` and a robust HC0
(sandwich) covariance: the pseudo-likelihood estimator used for survey
weights. ``freq_weights`` would treat each weight as that many replicated
decisions and understate standard errors by orders of magnitude. When the
sampling strata are themselves covariates in the model, the unweighted fit is
also consistent and more efficient, which is what the report uses.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tools.sm_exceptions import (
    ConvergenceWarning,
    HessianInversionWarning,
    PerfectSeparationError,
    PerfectSeparationWarning,
)

from hos_pipeline.config import Stats

DEFAULT_SEED = Stats().seed
NAN = float("nan")
TERM_COLUMNS = ["term", "coef", "se", "odds_ratio", "or_lo", "or_hi", "p"]


def _design(
    df: pd.DataFrame, outcome: str, predictors: Sequence[str], categorical: Sequence[str], extra: Sequence[str] = ()
) -> tuple[pd.DataFrame, list[str], list[str]]:
    cat = list(dict.fromkeys(categorical))
    num = [c for c in dict.fromkeys(predictors) if c not in cat]
    cols = [outcome, *num, *cat, *[c for c in extra if c]]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"missing columns {missing}")
    return df[cols].dropna(), num, cat


def _model_frame(term_rows: list[dict], info: dict) -> pd.DataFrame:
    out = pd.DataFrame(term_rows, columns=TERM_COLUMNS)
    for key, value in info.items():
        out[key] = [value] * len(out) if len(out) else pd.Series(dtype=object)
    out.attrs["model"] = info
    return out


def fit_logistic(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    categorical: Sequence[str] = (),
    weights: str | None = None,
) -> pd.DataFrame:
    """Fit ``outcome ~ predictors + categorical`` on complete cases.

    Categorical columns (listed in ``categorical``, whether or not also in
    ``predictors``) are dummy-coded against their first sorted level. Returns
    one row per term (coef, se, odds ratio with 95% CI, p) with model-level
    columns repeated on every row: n, n_events, pseudo_r2 (McFadden),
    converged, method, reference, warning. Perfect separation or a failed fit
    is reported in ``warning`` rather than raised.
    """
    use, num, cat = _design(df, outcome, predictors, categorical, [weights] if weights else [])
    info: dict = {"n": len(use), "n_events": NAN, "pseudo_r2": NAN, "converged": False,
                  "method": "glm_binomial_var_weights_hc0" if weights else "logit_mle",
                  "reference": "", "warning": ""}
    if use.empty:
        info["warning"] = "no complete cases"
        return _model_frame([], info)
    y = use[outcome].astype(float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError(f"{outcome!r} must be coded 0/1")
    info["n_events"] = int(y.sum())
    if y.nunique() < 2:
        info["warning"] = "outcome has a single level"
        return _model_frame([], info)

    parts = [use[num].astype(float)]
    refs = []
    for c in cat:
        levels = sorted(use[c].astype(str).unique())
        refs.append(f"{c}={levels[0]}")
        parts.append(pd.get_dummies(use[c].astype(str), prefix=c, prefix_sep="=", drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    constant = [c for c in X.columns if X[c].nunique() < 2]
    notes = [f"dropped constant column(s): {', '.join(constant)}"] if constant else []
    X = sm.add_constant(X.drop(columns=constant), has_constant="add")
    info["reference"] = "; ".join(refs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            if weights:
                res = sm.GLM(y, X, family=sm.families.Binomial(), var_weights=use[weights].astype(float)).fit(
                    cov_type="HC0", maxiter=200
                )
                converged = bool(res.converged)
                pr2 = float(res.pseudo_rsquared(kind="mcf"))
            else:
                res = sm.Logit(y, X).fit(disp=0, maxiter=200)
                converged = bool(res.mle_retvals.get("converged", False))
                pr2 = float(res.prsquared)
        except (PerfectSeparationError, np.linalg.LinAlgError, ValueError) as exc:
            info["warning"] = "; ".join([*notes, f"fit failed: {type(exc).__name__}: {exc}"])
            rows = [{"term": t, **{k: NAN for k in TERM_COLUMNS[1:]}} for t in X.columns]
            return _model_frame(rows, info)
    kinds = {type(w.message) for w in caught}
    if PerfectSeparationWarning in kinds:
        notes.append("perfect or quasi-perfect separation: estimates for separated terms are not identified")
    if ConvergenceWarning in kinds or not converged:
        notes.append("did not converge")
    if HessianInversionWarning in kinds:
        notes.append("Hessian not invertible")
    with np.errstate(over="ignore"):  # separated terms can have astronomically large odds ratios
        ci = np.exp(res.conf_int(alpha=0.05))
        rows = [
            {"term": t, "coef": float(res.params[t]), "se": float(res.bse[t]),
             "odds_ratio": float(np.exp(res.params[t])), "or_lo": float(ci.loc[t, 0]), "or_hi": float(ci.loc[t, 1]),
             "p": float(res.pvalues[t])}
            for t in X.columns
        ]
    info.update(pseudo_r2=pr2, converged=converged and PerfectSeparationWarning not in kinds, warning="; ".join(notes))
    return _model_frame(rows, info)


def cv_auc(
    df: pd.DataFrame,
    outcome: str,
    predictors: Sequence[str],
    categorical: Sequence[str] = (),
    folds: int = 5,
    seed: int = DEFAULT_SEED,
    repeats: int = 1,
) -> dict:
    """Stratified k-fold cross-validated ROC AUC of an L2 logistic regression.

    Pipeline: one-hot (categorical) + standardise (numeric) -> LogisticRegression
    (default C=1; the mild penalty keeps folds with near-separation finite).
    ``folds`` is reduced to the minority-class count when that is smaller.
    """
    use, num, cat = _design(df, outcome, predictors, categorical)
    y = use[outcome].astype(int).to_numpy()
    counts = np.bincount(y, minlength=2) if y.size else np.array([0, 0])
    k = int(min(folds, counts.min()))
    out = {"auc_mean": NAN, "auc_sd": NAN, "auc_folds": [], "folds": k, "repeats": repeats, "seed": seed,
           "n": int(y.size), "n_events": int(counts[1]) if y.size else 0, "note": ""}
    if k < 2:
        out["note"] = "fewer than two decisions in the minority class"
        return out
    if k < folds:
        out["note"] = f"folds reduced from {folds} to {k} (minority class size)"
    X = use[num + cat].copy()
    for c in cat:
        X[c] = X[c].astype(str).astype(object)
    transformers = []
    if cat:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat))
    if num:
        transformers.append(("num", StandardScaler(), num))
    pipe = Pipeline([("pre", ColumnTransformer(transformers)), ("clf", LogisticRegression(max_iter=1000))])
    aucs = []
    for train, test in RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats, random_state=seed).split(X, y):
        pipe.fit(X.iloc[train], y[train])
        aucs.append(float(roc_auc_score(y[test], pipe.predict_proba(X.iloc[test])[:, 1])))
    out.update(auc_mean=float(np.mean(aucs)), auc_sd=float(np.std(aucs, ddof=1)) if len(aucs) > 1 else NAN,
               auc_folds=aucs)
    return out
