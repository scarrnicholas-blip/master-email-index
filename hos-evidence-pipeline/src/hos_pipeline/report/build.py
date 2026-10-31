"""Build the statistical tables, figures and plain-English summary.

``build_report(settings)`` reads whatever stage outputs exist in the work
directory and writes, under ``work/outputs``:

    tables/*.csv     one file per analysis
    figures/*.png    prevalence, severe-by-year, duration box plot
    summary.md       methods, results, limitations, reproducibility block

Missing inputs skip the tables that need them; the summary says which. Every
number in summary.md is read from the tables computed here.
"""

from __future__ import annotations

import hashlib
import logging
import math
import platform
import re
import shutil
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

import hos_pipeline
from hos_pipeline.config import Settings
from hos_pipeline.io_utils import read_jsonl
from hos_pipeline.stats.core import cohens_kappa, holm, pooled_estimate, wilson_ci
from hos_pipeline.stats.models import cv_auc, fit_logistic
from hos_pipeline.stats.table import build_analysis_table, natural_ids
from hos_pipeline.stats.tests import compare_groups_holm, permutation_cluster_test

log = logging.getLogger(__name__)

INPUTS = ["decisions", "screen", "sample", "coded", "verified", "outcomes", "awaab_checks"]
PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "rapidfuzz", "pydantic", "PyYAML"]
AWAAB_STATUSES = ["met", "breached", "possibly_breached", "not_yet_due", "no_action_recorded", "not_applicable"]

# Logistic model eligibility: an H code enters only if it is determinable
# (yes/no) in at least this share of decisions and its rarer level has at
# least MODEL_MIN_LEVEL decisions. The rule looks at predictors only, never at
# the outcome, so selection cannot inflate the associations reported.
MODEL_MAX_UNCLEAR = 0.20
MODEL_MIN_LEVEL = 10
MIN_GROUP_N = 3

# Okabe-Ito: distinguishable with colour-vision deficiency; the pair differs
# in lightness, and series also differ by hatch / line style / marker.
BLUE, ORANGE, INK, MUTED, GRID, FILL = "#0072B2", "#E69F00", "#1A1A1A", "#5F5F5F", "#D9D9D9", "#CFE3F1"


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def _finite(x: Any) -> bool:
    try:
        return x is not None and not pd.isna(x) and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _pct(x: Any, nd: int = 1) -> str:
    return f"{100 * float(x):.{nd}f}%" if _finite(x) else "n/a"


def _ci(lo: Any, hi: Any, nd: int = 1) -> str:
    return f"{100 * float(lo):.{nd}f}–{100 * float(hi):.{nd}f}%" if _finite(lo) and _finite(hi) else "n/a"


def _num(x: Any, nd: int = 0) -> str:
    if not _finite(x):
        return "n/a"
    return f"{float(x):,.{nd}f}"


def _pval(p: Any) -> str:
    if not _finite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _md_table(df: pd.DataFrame, columns: Sequence[tuple[str, str, Callable[[Any], str]]]) -> str:
    """Render ``df`` as a Markdown table: columns = [(header, column, formatter)]."""
    if df is None or df.empty:
        return "_No rows._\n"
    head = "| " + " | ".join(h for h, _, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(f(r.get(c)) for _, c, f in columns) + " |" for r in df.to_dict("records")]
    return "\n".join([head, sep, *body]) + "\n"


def _txt(x: Any) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _inventory(work: Path) -> pd.DataFrame:
    rows = []
    for name in INPUTS:
        p = work / f"{name}.jsonl"
        if p.exists():
            with p.open("rb") as fh:
                n = sum(1 for line in fh if line.strip())
            rows.append({"file": p.name, "present": True, "rows": n, "bytes": p.stat().st_size, "sha256": _sha256(p)})
        else:
            rows.append({"file": p.name, "present": False, "rows": 0, "bytes": 0, "sha256": ""})
    return pd.DataFrame(rows)


def _load(work: Path, name: str, keep: Sequence[str] | None = None) -> pd.DataFrame | None:
    p = work / f"{name}.jsonl"
    if not p.exists():
        return None
    recs = ({k: r.get(k) for k in keep} for r in read_jsonl(p)) if keep else read_jsonl(p)
    return pd.DataFrame(list(recs))


def _codebook_info(path: Path | None) -> tuple[dict[str, str], str, str]:
    """(labels by H/D id, codebook file version, sha256) from the codebook YAML, best effort."""
    if path is None or not Path(path).exists():
        return {}, "", ""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 - labels are cosmetic
        return {}, "", _sha256(Path(path))
    labels: dict[str, str] = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ident = node.get("id")
            if isinstance(ident, str) and re.fullmatch(r"[HD]\d+", ident):
                for key in ("label", "short", "name", "title", "question", "description"):
                    if isinstance(node.get(key), str):
                        labels.setdefault(ident, " ".join(node[key].split()))
                        break
            for k, v in node.items():
                if isinstance(k, str) and re.fullmatch(r"[HD]\d+", k) and isinstance(v, (str, dict)):
                    text = v if isinstance(v, str) else next(
                        (v[x] for x in ("label", "short", "name", "title", "question") if isinstance(v.get(x), str)), None)
                    if text:
                        labels.setdefault(k, " ".join(text.split()))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    version = str(data.get("version", "")) if isinstance(data, dict) else ""
    return labels, version, _sha256(Path(path))


# --------------------------------------------------------------------------
# Report state
# --------------------------------------------------------------------------


@dataclass
class _Report:
    settings: Settings
    out: Path
    notes: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[str] = field(default_factory=list)

    @property
    def alpha(self) -> float:
        return self.settings.stats.alpha

    @property
    def level(self) -> str:
        return f"{100 * (1 - self.alpha):g}%"

    @property
    def years(self) -> list[int]:
        return list(self.settings.stats.pooled_years)

    @property
    def period(self) -> str:
        return f"{min(self.years)}–{max(self.years)}" if self.years else "all years"

    def save(self, name: str, df: pd.DataFrame) -> None:
        self.tables[name] = df
        df.to_csv(self.out / "tables" / f"{name}.csv", index=False)

    def skip(self, what: str, why: str) -> None:
        self.notes.append(f"{what}: not produced ({why}).")

    @contextmanager
    def guard(self, what: str):
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - one failed table must not sink the report
            log.exception("report section %s failed", what)
            self.notes.append(f"{what}: failed ({type(exc).__name__}: {exc}).")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def screening_by_year(screen: pd.DataFrame, alpha: float) -> pd.DataFrame:
    s = screen.copy()
    s["year"] = s["year"].astype(object).where(s["year"].notna(), "unknown")
    rows = []
    for year, g in s.groupby("year", sort=False):
        k, n = int(g["damp_flag"].sum()), len(g)
        p, lo, hi = wilson_ci(k, n, alpha)
        rows.append({"year": year, "n_screened": n, "n_damp_flagged": k, "pct_damp": p, "lo": lo, "hi": hi,
                     "n_complaint_definition_hit": int(g["complaint_definition_hit"].sum()),
                     "n_determination_hit": int(g["determination_hit"].sum()),
                     "n_keyword_severe_damp": int(g.get("keyword_severe_damp", pd.Series(dtype=bool)).fillna(False).sum())})
    out = pd.DataFrame(rows)
    out["_k"] = out["year"].map(lambda y: (1, 0) if y == "unknown" else (0, int(y)))
    out = out.sort_values("_k").drop(columns="_k")
    k, n = int(out["n_damp_flagged"].sum()), int(out["n_screened"].sum())
    p, lo, hi = wilson_ci(k, n, alpha)
    total = {"year": "all", "n_screened": n, "n_damp_flagged": k, "pct_damp": p, "lo": lo, "hi": hi,
             **{c: int(out[c].sum()) for c in ("n_complaint_definition_hit", "n_determination_hit",
                                               "n_keyword_severe_damp")}}
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)


def sample_strata(sample: pd.DataFrame, table: pd.DataFrame | None) -> pd.DataFrame:
    g = sample.groupby("stratum").agg(population_n=("population_n", "first"), sample_n=("sample_n", "first"),
                                      weight=("weight", "first"), n_drawn=("case_id", "size")).reset_index()
    coded = table.groupby("stratum").size() if table is not None and len(table) else pd.Series(dtype=int)
    g["n_coded"] = g["stratum"].map(coded).fillna(0).astype(int)
    g["sampling_fraction"] = g["sample_n"] / g["population_n"]
    total = {"stratum": "all", "population_n": int(g["population_n"].sum()), "sample_n": int(g["sample_n"].sum()),
             "weight": np.nan, "n_drawn": int(g["n_drawn"].sum()), "n_coded": int(g["n_coded"].sum())}
    total["sampling_fraction"] = total["sample_n"] / total["population_n"] if total["population_n"] else np.nan
    return pd.concat([g, pd.DataFrame([total])], ignore_index=True)


def _weighted_frame(table: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Table with usable weights; falls back to weight 1 when no sample weights exist."""
    if table["weight"].notna().any():
        return table, True
    t = table.copy()
    t["weight"] = 1.0
    return t, False


def prevalence(
    table: pd.DataFrame, ids: Sequence[str], labels: dict[str, str], rep: _Report, grades: Sequence[str] = ("B",),
    checked: set[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Unweighted Wilson and population-weighted pooled prevalence per code and grade."""
    t, _ = _weighted_frame(table)
    recal = t["stratum"].notna().all() and t["population_n"].notna().all()
    in_years = t[t["year"].isin(rep.years)]
    rows = []
    for code in ids:
        for grade in grades:
            col = code if grade == "B" else f"{code}_A"
            if col not in t.columns or t[col].notna().sum() == 0:
                continue
            y = t[col]
            k, n = int((y == 1).sum()), int(y.notna().sum())
            p, lo, hi = wilson_ci(k, n, rep.alpha)
            w = pooled_estimate(in_years, col, "weight", "year", rep.years, rep.alpha,
                                stratum_col="stratum" if recal else None,
                                population_col="population_n" if recal else None)
            row = {"id": code, "label": labels.get(code, ""), "grade": grade, "n_coded": len(t), "k": k, "n": n,
                   "n_unclear": len(t) - n, "p": p, "lo": lo, "hi": hi,
                   "p_unclear_as_no": k / len(t) if len(t) else np.nan,
                   "w_p": w["estimate"], "w_lo": w["lo"], "w_hi": w["hi"], "w_n": w["n"], "w_n_eff": w["n_eff"],
                   "w_deff": w["deff"], "w_years": rep.period}
            if grade == "A" and checked is not None:
                yes = t.loc[t[code] == 1, "case_id"]
                row["verification_coverage"] = (
                    sum((c, code) in checked for c in yes) / len(yes) if len(yes) else np.nan)
            rows.append(row)
    return pd.DataFrame(rows)


def severe_by_year(screen: pd.DataFrame | None, outcomes: pd.DataFrame | None, years_of: dict[str, int],
                   rep: _Report) -> pd.DataFrame:
    if screen is not None:
        base = screen.loc[screen["damp_flag"].astype(bool), ["case_id", "year", "keyword_severe_damp"]].copy()
    else:
        base = pd.DataFrame({"case_id": outcomes["case_id"], "year": outcomes["case_id"].map(years_of),
                             "keyword_severe_damp": np.nan})
    if outcomes is not None:
        base = base.merge(outcomes[["case_id", "severe_damp"]], on="case_id", how="left")
    else:
        base["severe_damp"] = np.nan
    base = base[base["year"].notna()]
    base["year"] = base["year"].astype(int)

    def summarise(label: str, g: pd.DataFrame) -> dict:
        kw = g["keyword_severe_damp"].dropna().astype(float)
        clf = g["severe_damp"].dropna().astype(float)
        both = g.dropna(subset=["keyword_severe_damp", "severe_damp"])
        kp, klo, khi = wilson_ci(int(kw.sum()), len(kw), rep.alpha)
        cp, clo, chi_ = wilson_ci(int(clf.sum()), len(clf), rep.alpha)
        a, b = both["keyword_severe_damp"].astype(bool).tolist(), both["severe_damp"].astype(bool).tolist()
        return {"period": label, "n_damp_flagged": len(g), "kw_k": int(kw.sum()), "kw_n": len(kw), "kw_p": kp,
                "kw_lo": klo, "kw_hi": khi, "clf_k": int(clf.sum()), "clf_n": len(clf), "clf_p": cp, "clf_lo": clo,
                "clf_hi": chi_, "n_both": len(both),
                "agreement": float(np.mean([x == y for x, y in zip(a, b)])) if a else np.nan,
                "kappa": cohens_kappa(a, b)}

    rows = [summarise(str(y), g) for y, g in base.groupby("year")]
    rows.append(summarise(rep.period, base[base["year"].isin(rep.years)]))
    return pd.DataFrame(rows)


def duration_comparisons(table: pd.DataFrame, severe_col: str, rep: _Report) -> pd.DataFrame:
    d = table[table["duration_days"].notna()]
    specs = [
        {"label": "Duration: severe vs not severe", "value": "duration_days", "group": severe_col, "a": 1.0, "b": 0.0},
        {"label": "Duration (works completed only): severe vs not severe", "value": "duration_days",
         "group": severe_col, "a": 1.0, "b": 0.0, "where": {"duration_end": "works_completed"}},
    ]
    types = sorted(t for t in d["landlord_type"].dropna().unique())
    for i, a in enumerate(types):
        for b in types[i + 1:]:
            specs.append({"label": f"Duration: {a} vs {b}", "value": "duration_days", "group": "landlord_type",
                          "a": a, "b": b})
    usable = []
    for s in specs:
        sub = d
        for c, v in (s.get("where") or {}).items():
            sub = sub[sub[c] == v]
        na, nb = int((sub[s["group"]] == s["a"]).sum()), int((sub[s["group"]] == s["b"]).sum())
        if min(na, nb) >= MIN_GROUP_N:
            usable.append(s)
        else:
            rep.notes.append(f"Duration comparison '{s['label']}' skipped (n = {na} and {nb}; need {MIN_GROUP_N}+).")
    return compare_groups_holm(d, usable, alpha=rep.alpha) if usable else pd.DataFrame()


def landlord_clustering(table: pd.DataFrame, ids: Sequence[str], rep: _Report) -> pd.DataFrame:
    strata = "stratum" if table["stratum"].notna().all() else "year"
    rows = []
    for code in ids:
        if table[code].notna().sum() == 0:
            continue
        r = permutation_cluster_test(table, code, "landlord", n_perm=rep.settings.stats.permutations,
                                     seed=rep.settings.stats.seed, strata=strata, min_group_n=MIN_GROUP_N)
        rows.append({"id": code, **r})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_holm"] = holm(out["p_value"].tolist())
        out["reject_holm"] = out["p_holm"] < rep.alpha
    return out


def logistic_model(table: pd.DataFrame, outcome: str, ids: Sequence[str], rep: _Report
                   ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    d = table[table[outcome].notna() & table["year"].isin(rep.years)].copy()
    year0 = min(rep.years)
    ycol = f"years_since_{year0}"
    d[ycol] = d["year"].astype(float) - year0
    chosen = []
    for code in ids:
        unclear = d[code].isna().mean() if len(d) else 1.0
        filled = d[code].fillna(0.0)
        if unclear <= MODEL_MAX_UNCLEAR and min((filled == 1).sum(), (filled == 0).sum()) >= MODEL_MIN_LEVEL:
            chosen.append(code)
        d[code] = filled
    cat = ["landlord_type"] if d["landlord_type"].nunique() > 1 else []
    preds = [*chosen, ycol]
    terms = fit_logistic(d, outcome, preds, cat)
    auc = cv_auc(d, outcome, preds, cat, folds=rep.settings.stats.cv_folds, seed=rep.settings.stats.seed)
    info = terms.attrs.get("model", {})
    n_params = max(len(terms) - 1, 1)
    auc_row = {k: v for k, v in auc.items() if k != "auc_folds"}
    auc_row.update(outcome=outcome, predictors="; ".join(preds + cat), n_params=n_params,
                   events_per_parameter=(info.get("n_events", np.nan) / n_params) if n_params else np.nan,
                   auc_folds="; ".join(f"{a:.3f}" for a in auc["auc_folds"]))
    return terms, pd.DataFrame([auc_row]), chosen


def awaab_summary(awaab: pd.DataFrame, rep: _Report) -> pd.DataFrame:
    rows = []

    def summarise(duty: str, in_force: str, g: pd.DataFrame) -> dict:
        counts = g["status"].value_counts()
        row = {"duty": duty, "in_force": in_force, "n_checks": len(g), "n_cases": g["case_id"].nunique(),
               **{s: int(counts.get(s, 0)) for s in AWAAB_STATUSES}}
        assessable = row["met"] + row["breached"] + row["possibly_breached"]
        p, lo, hi = wilson_ci(row["breached"], assessable, rep.alpha)
        row.update(assessable=assessable, breach_rate=p, breach_lo=lo, breach_hi=hi,
                   breach_rate_incl_possible=(row["breached"] + row["possibly_breached"]) / assessable
                   if assessable else np.nan)
        return row

    for (duty, force), g in awaab.groupby(["duty", "in_force"]):
        rows.append(summarise(duty, "in force" if force else "before commencement", g))
    rows.append(summarise("all duties", "all", awaab))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def _axes(width: float, height: float, ncols: int = 1):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, ncols, squeeze=False)[0]
    for ax in axes:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK, labelsize=8)
    return fig, axes


def fig_prevalence(prev: pd.DataFrame, path: Path, rep: _Report) -> None:
    b = prev[prev["grade"] == "B"].set_index("id")
    a = prev[prev["grade"] == "A"].set_index("id")
    ids = list(b.index)
    fig, (ax,) = _axes(7.5, 0.45 * len(ids) + 1.6)
    y = np.arange(len(ids))[::-1].astype(float)
    series = [("Coded (grade B)", b, BLUE, None)]
    if not a.empty and a["w_p"].notna().any():
        series.append(("Verified (grade A)", a.reindex(ids), ORANGE, "///"))
    h = 0.8 / len(series)
    for j, (name, s, colour, hatch) in enumerate(series):
        yy = y + (len(series) - 1) * h / 2 - j * h
        p = s["w_p"].to_numpy(float) * 100
        lo, hi = s["w_lo"].to_numpy(float) * 100, s["w_hi"].to_numpy(float) * 100
        ax.barh(yy, np.nan_to_num(p), height=h * 0.92, color=colour, edgecolor=INK, linewidth=0.6, hatch=hatch,
                label=name)
        ax.errorbar(p, yy, xerr=[p - lo, hi - p], fmt="none", ecolor=INK, elinewidth=0.9, capsize=2)
        for yi, pi, hi_i in zip(yy, p, hi):
            if np.isfinite(pi):
                ax.text((hi_i if np.isfinite(hi_i) else pi) + 1.2, yi, f"{pi:.0f}%", va="center", fontsize=7, color=INK)
        if len(series) > 1 and np.isfinite(hi[0]):
            ax.text(hi[0] + 7, yy[0], name, va="center", fontsize=7, color=MUTED, style="italic")
    ax.set_yticks(y, [f"{i} {b.at[i, 'label'][:38]}".strip() for i in ids])
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel(f"Population-weighted share of damp/mould decisions, {rep.period} ({rep.level} CI)", fontsize=8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=7, loc="lower right")
    ax.set_title("Failure hypotheses recorded in decisions", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path)


def fig_severe_by_year(sev: pd.DataFrame, path: Path) -> None:
    d = sev[sev["period"].str.fullmatch(r"\d{4}")].copy()
    d["year"] = d["period"].astype(int)
    fig, (ax,) = _axes(7, 3.8)
    lines = [("Classifier", "clf_p", BLUE, "-", "o"), ("Keyword screen (grade D)", "kw_p", ORANGE, "--", "s")]
    ends = []
    for name, col, colour, ls, marker in lines:
        if d[col].notna().any():
            ax.plot(d["year"], d[col] * 100, color=colour, linestyle=ls, marker=marker, markersize=6, linewidth=2,
                    label=name)
            last = d[d[col].notna()].iloc[-1]
            ends.append([last["year"], last[col] * 100, name])
    if len(ends) == 2 and abs(ends[0][1] - ends[1][1]) < 4:
        hi_i = 0 if ends[0][1] >= ends[1][1] else 1
        ends[hi_i][1] += 2
        ends[1 - hi_i][1] -= 2
    for x, yv, name in ends:
        ax.text(x + 0.15, yv, name, va="center", fontsize=7, color=INK)
    ax.set_xlim(d["year"].min() - 0.3, d["year"].max() + 2.2)
    ax.set_xticks(d["year"])
    ax.set_ylim(0, max(5.0, float(np.nanmax(d[["clf_p", "kw_p"]].to_numpy(float)) * 100) * 1.25)
                if d[["clf_p", "kw_p"]].notna().any().any() else 100)
    ax.set_ylabel("Damp-flagged decisions with severe\nmaladministration on damp/mould", fontsize=8)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    ax.set_title("Severe maladministration on damp and mould, by decision year", fontsize=10, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path)


def fig_duration(table: pd.DataFrame, severe_col: str, path: Path) -> None:
    d = table[table["duration_days"].notna()]
    panels = []
    sev = [("Not severe", d.loc[d[severe_col] == 0, "duration_days"]), ("Severe", d.loc[d[severe_col] == 1, "duration_days"])]
    panels.append(("By severe maladministration finding", [s for s in sev if len(s[1])]))
    types = [(t.replace("_", " ").capitalize(), d.loc[d["landlord_type"] == t, "duration_days"])
             for t in sorted(d["landlord_type"].dropna().unique())]
    panels.append(("By landlord type", [s for s in types if len(s[1])]))
    panels = [p for p in panels if p[1]]
    fig, axes = _axes(7.5, 4, ncols=len(panels))
    for ax, (title, groups) in zip(axes, panels):
        ax.boxplot([g.to_numpy(float) for _, g in groups], tick_labels=[f"{n}\n(n={len(g)})" for n, g in groups],
                   widths=0.5, patch_artist=True,
                   boxprops={"facecolor": FILL, "edgecolor": INK, "linewidth": 0.9},
                   medianprops={"color": INK, "linewidth": 2}, whiskerprops={"color": INK, "linewidth": 0.9},
                   capprops={"color": INK, "linewidth": 0.9},
                   flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none", "markeredgecolor": MUTED})
        for i, (_, g) in enumerate(groups, start=1):
            ax.text(i + 0.3, float(np.median(g)), f"{np.median(g):,.0f}", va="center", fontsize=7, color=INK)
        ax.set_title(title, fontsize=9, loc="left", color=INK)
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Days from landlord aware to works completed\n(or determination); median labelled", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "hos_pipeline": hos_pipeline.__version__}
    for pkg in PACKAGES:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not installed"
    return out


def _methods(rep: _Report, weighted: bool, severe_source: str, model_rule: str) -> str:
    s = rep.settings.stats
    weight_text = (
        "Each coded decision stands for the damp-flagged decisions in its sampling stratum (decision year and "
        "landlord type), so population-weighted estimates weight it by the stratum's population size divided by "
        "the number of decisions with a usable answer in that stratum."
        if weighted else
        "No sampling weights were available (sample.jsonl missing), so 'weighted' estimates treat every coded "
        "decision equally."
    )
    return f"""## Methods in plain English

**Screening and sampling.** Every published decision was screened for damp, mould and condensation. Screening
(evidence grade D) flags a decision when the complaint definition or a determination names damp or mould, or
the terms recur. A stratified random sample of flagged decisions was then coded in full.

**Coding and evidence grades.** Each sampled decision was coded against the codebook (grade B): every failure
hypothesis (H codes) and remediation dimension (D codes) is "yes", "no" or "unclear". A strict second reader
then checked each "yes" against the cited paragraphs; a confirmed claim is grade A. Grade A figures count only
confirmed claims, so they are a lower bound.

**Prevalence.** For each code we report the simple share of coded decisions answering "yes" among those with a
yes/no answer, with a Wilson {rep.level} confidence interval, and a population-weighted pooled estimate for
{rep.period}. {weight_text} Its interval is a Wilson interval on the Kish effective sample size, which widens the
interval to reflect unequal weights. "Unclear" answers are left out of both; the column `p_unclear_as_no` shows
the share if every unclear answer were a "no".

**Severe maladministration.** Severe maladministration on a damp/mould element is measured two ways: the grade D
keyword screen and the outcome classifier that reads the Determination section. Agreement between them is
reported as Cohen's kappa. Model and duration analyses use the {severe_source}.

**Durations.** Duration runs from the earliest date the landlord could have known of the damp or mould to the
latest date works could have been completed, or the determination date if completion is not recorded. Where a
decision gives only a month or year, this is the longest duration consistent with the text. Groups are compared
with Mann–Whitney tests (rank-biserial effect size), Holm-adjusted across the comparisons.

**Landlord clustering.** For each H code, a permutation test asks whether "yes" answers bunch up within particular
landlords more than chance would produce. The statistic is the between-landlord spread of each landlord's rate;
the null distribution reshuffles answers {s.permutations:,} times within sampling strata, so differences between
years and landlord types are not mistaken for clustering. Landlords with fewer than {MIN_GROUP_N} coded decisions
are left out. P-values are Holm-adjusted across the H codes.

**Model.** A logistic regression relates the severe damp/mould finding to the H codes, decision year (linear) and
landlord type. {model_rule} "Unclear" is treated as "not established" (0). The model is unweighted because the
sampling strata (year, landlord type) are themselves in the model. Discrimination is the ROC AUC from
{s.cv_folds}-fold stratified cross-validation.

**Awaab's Law.** Event dates from each decision are compared with the duty time limits. Most decisions predate
commencement, so those rows are a benchmark of what the duties would have required, not findings of breach.
"Possibly breached" means the date precision allows either answer.

**Landlord rates.** Damp-flagged decisions in {rep.period} and severe damp/mould findings per 1,000 homes owned,
using RSH SDR (housing associations) and LAHS (councils) stock, with exact Poisson (Garwood) intervals. Names are
matched through the alias table, exact normalised names, then fuzzy matching; landlords below the minimum stock
are listed but not ranked.
"""


def _limitations(rep: _Report) -> str:
    return f"""## Limitations

- Published decisions are complaints that reached the Ombudsman; they are not a random sample of tenancies or of
  landlords' damp cases, and publication practice changed over time.
- Codes describe what a decision records. "No" can mean the decision is silent, not that the failure did not occur.
- Grade B codes are model- or coder-assigned; only grade A figures have been checked against the cited text.
- Weighted intervals use the Kish effective sample size and no finite population correction, so they are
  approximate and somewhat conservative; the strata themselves are not modelled in the variance.
- Durations from month- or year-precision dates are upper bounds (see `duration_days_min` in the analysis table);
  durations ending at the determination date understate how long problems lasted where works were never finished.
- The logistic model shows associations, not causes, and its estimates are unstable when there are few severe
  findings per parameter (see `events_per_parameter`).
- Landlord rates divide a {rep.period} count by one stock snapshot; mergers, stock transfers and name matching
  (see `entity_match_review.csv`) affect them.
"""


def _summary(rep: _Report, inv: pd.DataFrame, meta: dict) -> str:
    t = rep.tables
    lines: list[str] = ["# Housing Ombudsman damp and mould decisions: statistical summary", ""]
    lines.append(f"Generated {meta['timestamp']} by hos_pipeline {hos_pipeline.__version__}.")
    if meta["synthetic"]:
        lines += ["", "> **SYNTHETIC DATA.** These inputs were generated by `hos_pipeline.synthetic` for testing. "
                      "Nothing here describes real landlords or decisions."]
    lines += ["", "## Inputs", "", _md_table(inv, [("File", "file", str), ("Present", "present", lambda v: "yes" if v else "no"),
                                              ("Rows", "rows", _num)])]
    lines.append(meta["methods"])
    lines.append("## Results\n")

    if "screening_by_year" in t:
        df = t["screening_by_year"]
        tot = df[df["year"] == "all"].iloc[0]
        lines += ["### Screening (tables/screening_by_year.csv)", "",
                  f"Of {_num(tot['n_screened'])} decisions screened, {_num(tot['n_damp_flagged'])} "
                  f"({_pct(tot['pct_damp'])}; {rep.level} CI {_ci(tot['lo'], tot['hi'])}) were damp-flagged.", "",
                  _md_table(df, [("Year", "year", str), ("Screened", "n_screened", _num),
                                 ("Damp-flagged", "n_damp_flagged", _num), ("Share", "pct_damp", _pct),
                                 ("Severe (keyword)", "n_keyword_severe_damp", _num)])]

    if "sample_strata" in t:
        df = t["sample_strata"]
        tot = df[df["stratum"] == "all"].iloc[0]
        lines += ["### Sample (tables/sample_strata.csv)", "",
                  f"{_num(tot['sample_n'])} of {_num(tot['population_n'])} damp-flagged decisions were sampled across "
                  f"{len(df) - 1} strata ({_pct(tot['sampling_fraction'])}); {_num(tot['n_coded'])} have been coded.", "",
                  _md_table(df, [("Stratum", "stratum", str), ("Population", "population_n", _num),
                                 ("Sampled", "sample_n", _num), ("Coded", "n_coded", _num),
                                 ("Weight", "weight", lambda v: _num(v, 2))])]

    for name, title in (("hypothesis_prevalence", "Failure hypotheses"), ("remediation_prevalence", "Remediation dimensions")):
        if name not in t:
            continue
        df = t[name].copy()
        df["code"] = (df["id"] + " " + df["label"].fillna("")).str.strip()
        lines += [f"### {title} (tables/{name}.csv)", "",
                  f"Shares of coded decisions answering \"yes\". 'Weighted' is the population-weighted pooled "
                  f"{rep.period} estimate with its {rep.level} CI.", ""]
        df["ci"] = [_ci(a, b) for a, b in zip(df["lo"], df["hi"])]
        df["w_ci"] = [_ci(a, b) for a, b in zip(df["w_lo"], df["w_hi"])]
        df["yes_n"] = [f"{_num(a)} / {_num(b)}" for a, b in zip(df["k"], df["n"])]
        cols = [("Code", "code", str), ("Grade", "grade", str), ("Yes / answered", "yes_n", str), ("Share", "p", _pct),
                (f"{rep.level} CI", "ci", str), ("Weighted", "w_p", _pct), (f"Weighted {rep.level} CI", "w_ci", str),
                ("Unclear", "n_unclear", _num)]
        top = df[df["grade"] == "B"].sort_values("w_p", ascending=False).head(3)
        if len(top):
            lines.append("Most common (weighted, grade B): " + "; ".join(
                f"{r.code} {_pct(r.w_p)} ({_ci(r.w_lo, r.w_hi)})" for r in top.itertuples()) + ".\n")
        if "verification_coverage" in df.columns:
            cov = df.loc[df["grade"] == "A", "verification_coverage"].dropna()
            if len(cov):
                lines.append(f"The second reader has checked between {_pct(cov.min())} and {_pct(cov.max())} of "
                             "\"yes\" claims per hypothesis; unchecked claims count as unverified in grade A.\n")
        lines.append(_md_table(df, cols))

    if "severe_by_year" in t:
        df = t["severe_by_year"]
        pooled = df.iloc[-1]
        lines += ["### Severe maladministration on damp/mould (tables/severe_by_year.csv)", "",
                  f"Pooled {rep.period}: keyword screen {_pct(pooled['kw_p'])} ({_ci(pooled['kw_lo'], pooled['kw_hi'])}, "
                  f"n = {_num(pooled['kw_n'])}); classifier {_pct(pooled['clf_p'])} "
                  f"({_ci(pooled['clf_lo'], pooled['clf_hi'])}, n = {_num(pooled['clf_n'])}). Where both exist "
                  f"(n = {_num(pooled['n_both'])}) they agree on {_pct(pooled['agreement'])} of decisions, "
                  f"Cohen's kappa {_num(pooled['kappa'], 2)}.", "",
                  _md_table(df, [("Period", "period", str), ("Damp-flagged", "n_damp_flagged", _num),
                                 ("Keyword", "kw_p", _pct), ("Classifier", "clf_p", _pct),
                                 ("Classified", "clf_n", _num), ("Kappa", "kappa", lambda v: _num(v, 2))])]

    if "duration_comparisons" in t and not t["duration_comparisons"].empty:
        df = t["duration_comparisons"]
        lines += ["### Durations (tables/duration_comparisons.csv)", "",
                  f"Median days in each group, Mann–Whitney p-values and Holm-adjusted p-values across the "
                  f"{len(df)} comparisons. Rank-biserial r runs from -1 to 1; positive means the first group "
                  "waited longer.", "",
                  _md_table(df, [("Comparison", "label", str), ("n (first)", "n_a", _num), ("n (second)", "n_b", _num),
                                 ("Median (first)", "summary_a", _num), ("Median (second)", "summary_b", _num),
                                 ("r", "effect", lambda v: _num(v, 2)), ("p", "p", _pval), ("p (Holm)", "p_holm", _pval)])]

    if "landlord_clustering" in t and not t["landlord_clustering"].empty:
        df = t["landlord_clustering"]
        hits = df[df["reject_holm"]]
        verdict = (", ".join(f"{r.id} (spread {_num(r.dispersion_ratio, 1)}× chance, Holm p {_pval(r.p_holm)})"
                             for r in hits.itertuples()) if len(hits) else "none of the H codes")
        lines += ["### Landlord clustering (tables/landlord_clustering.csv)", "",
                  f"Clustered within landlords beyond chance after Holm adjustment (alpha {rep.alpha:g}): {verdict}. "
                  f"Tests used {_num(df['n_perm'].iloc[0])} permutations and {_num(df['n_groups'].max())} landlords "
                  f"with at least {MIN_GROUP_N} coded decisions.", "",
                  _md_table(df, [("Code", "id", str), ("Decisions", "n", _num), ("Landlords", "n_groups", _num),
                                 ("Spread vs chance", "dispersion_ratio", lambda v: _num(v, 2)),
                                 ("p", "p_value", _pval), ("p (Holm)", "p_holm", _pval)])]

    if "logistic_severe_damp" in t:
        terms, auc = t["logistic_severe_damp"], t["logistic_cv_auc"].iloc[0]
        n = terms["n"].iloc[0] if len(terms) else auc["n"]
        events = terms["n_events"].iloc[0] if len(terms) else auc["n_events"]
        pr2 = terms["pseudo_r2"].iloc[0] if len(terms) else np.nan
        warn = terms["warning"].iloc[0] if len(terms) and isinstance(terms["warning"].iloc[0], str) else ""
        lines += ["### Model of severe damp/mould findings (tables/logistic_severe_damp.csv, logistic_cv_auc.csv)", "",
                  f"Outcome: {auc['outcome']}. {_num(n)} decisions, {_num(events)} with the finding "
                  f"({_num(auc['events_per_parameter'], 1)} per parameter). McFadden pseudo-R² {_num(pr2, 3)}; "
                  f"cross-validated AUC {_num(auc['auc_mean'], 3)} (SD {_num(auc['auc_sd'], 3)}, "
                  f"{_num(auc['folds'])} folds).", ""]
        if warn:
            lines += [f"Model warning: {warn}.", ""]
        if auc.get("note"):
            lines += [f"Cross-validation note: {auc['note']}.", ""]
        body = terms[terms["term"] != "const"] if len(terms) else terms
        lines.append(_md_table(body, [("Term", "term", str), ("Odds ratio", "odds_ratio", lambda v: _num(v, 2)),
                                      (f"{rep.level} CI low", "or_lo", lambda v: _num(v, 2)),
                                      (f"{rep.level} CI high", "or_hi", lambda v: _num(v, 2)), ("p", "p", _pval)]))

    if "awaab_summary" in t:
        df = t["awaab_summary"]
        allrow = df[df["duty"] == "all duties"].iloc[0]
        lines += ["### Awaab's Law duty benchmark (tables/awaab_summary.csv)", "",
                  f"{_num(allrow['n_checks'])} duty checks across {_num(allrow['n_cases'])} decisions. Of "
                  f"{_num(allrow['assessable'])} assessable checks, {_pct(allrow['breach_rate'])} "
                  f"({_ci(allrow['breach_lo'], allrow['breach_hi'])}) exceeded the time limit outright, and "
                  f"{_pct(allrow['breach_rate_incl_possible'])} including possible breaches.", "",
                  _md_table(df, [("Duty", "duty", str), ("Status", "in_force", str), ("Checks", "n_checks", _num),
                                 ("Met", "met", _num), ("Breached", "breached", _num),
                                 ("Possibly", "possibly_breached", _num), ("Breach rate", "breach_rate", _pct)])]

    if "landlord_rates" in t:
        df = t["landlord_rates"]
        ranked = df[df["ranked"].astype(str).str.lower() == "true"]
        lines += ["### Landlord rates per 1,000 homes (tables/landlord_rates.csv)", "",
                  f"{_num(len(df))} matched landlords, {_num(len(ranked))} with enough stock to rank. "
                  "Highest damp-flagged decision rates:", "",
                  _md_table(ranked.head(10), [("Landlord", "entity", str), ("Homes", "stock", _num),
                                              ("Decisions", "n_damp_decisions", _num),
                                              ("Per 1,000", "damp_decisions_per_1000", lambda v: _num(v, 2)),
                                              ("Low", "damp_decisions_per_1000_lo", lambda v: _num(v, 2)),
                                              ("High", "damp_decisions_per_1000_hi", lambda v: _num(v, 2)),
                                              ("Severe", "n_severe_damp", _num)])]

    if rep.figures:
        lines += ["### Figures", ""] + [f"- figures/{f}" for f in rep.figures] + [""]
    if rep.notes:
        lines += ["## Not produced or skipped", ""] + [f"- {n}" for n in rep.notes] + [""]
    lines.append(_limitations(rep))

    s = rep.settings.stats
    repro = [f"generated_utc: {meta['timestamp']}", f"seed: {s.seed}", f"permutations: {s.permutations}",
             f"cv_folds: {s.cv_folds}", f"alpha: {s.alpha}", f"pooled_years: {rep.period}",
             f"codebook_versions_in_coding: {meta['codebook_versions'] or 'n/a'}",
             f"codebook_file_version: {meta['codebook_file_version'] or 'n/a'}",
             f"codebook_file_sha256: {meta['codebook_sha256'] or 'n/a'}", "packages:"]
    repro += [f"  {k}: {v}" for k, v in _versions().items()]
    repro.append("inputs:")
    for r in inv.itertuples():
        repro.append(f"  {r.file}: rows={r.rows} sha256={r.sha256 or 'missing'}")
    lines += ["## Reproducibility", "", "```", *repro, "```", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def build_report(settings: Settings) -> Path:
    """Compute every table and figure the available inputs allow; return summary.md's path."""
    work = settings.paths.work_dir
    out = settings.work("outputs", "summary.md").parent
    for sub in ("tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rep = _Report(settings, out)
    inv = _inventory(work)
    present = dict(zip(inv["file"].str.removesuffix(".jsonl"), inv["present"]))
    labels, cb_version, cb_sha = _codebook_info(settings.paths.codebook)

    screen = _load(work, "screen", ["case_id", "landlord", "year", "damp_flag", "complaint_definition_hit",
                                    "determination_hit", "keyword_severe_damp"])
    sample = _load(work, "sample")
    outcomes = _load(work, "outcomes", ["case_id", "severe_damp", "any_severe", "severe_complaint_handling"])
    verified = _load(work, "verified", ["case_id", "claim_id", "grade", "verdict"])
    awaab = _load(work, "awaab_checks", ["case_id", "duty", "status", "in_force"])
    decisions = _load(work, "decisions", ["case_id", "year"])
    years_of = dict(zip(decisions["case_id"], decisions["year"])) if decisions is not None else {}

    table = None
    if present["coded"]:
        with rep.guard("Analysis table"):
            table = build_analysis_table(settings)
            if table.empty:
                table = None
    for name in INPUTS:
        if not present[name]:
            rep.notes.append(f"{name}.jsonl missing: tables that need it were skipped.")

    if screen is not None:
        with rep.guard("Screening by year"):
            rep.save("screening_by_year", screening_by_year(screen, rep.alpha))
    if sample is not None:
        with rep.guard("Sample strata"):
            rep.save("sample_strata", sample_strata(sample, table))

    h_ids = d_ids = []
    if table is not None:
        h_ids = [c for c in natural_ids(table.columns, "H", 0) if table[c].notna().any()]
        d_ids = [c for c in natural_ids(table.columns, "D", 0) if table[c].notna().any()]
        _, weighted = _weighted_frame(table)
        with rep.guard("Hypothesis prevalence"):
            checked = set(zip(verified["case_id"], verified["claim_id"])) if verified is not None else None
            grades = ("B", "A") if verified is not None else ("B",)
            rep.save("hypothesis_prevalence", prevalence(table, h_ids, labels, rep, grades, checked))
        with rep.guard("Remediation prevalence"):
            rep.save("remediation_prevalence", prevalence(table, d_ids, labels, rep))
    else:
        weighted = False
        rep.skip("Prevalence, durations, clustering and model tables", "no coded decisions")

    if screen is not None or outcomes is not None:
        with rep.guard("Severe by year"):
            rep.save("severe_by_year", severe_by_year(screen, outcomes, years_of, rep))

    severe_col, severe_source = "severe_damp", "outcome classifier's severe damp/mould finding"
    if table is not None and table["severe_damp"].notna().sum() == 0:
        severe_col, severe_source = "severe_damp_coded", "coded damp/mould determination (classifier output missing)"

    if table is not None:
        with rep.guard("Duration comparisons"):
            if table["duration_days"].notna().any():
                rep.save("duration_comparisons", duration_comparisons(table, severe_col, rep))
            else:
                rep.skip("Duration comparisons", "no decision has both a landlord-aware date and an end date")
        with rep.guard("Landlord clustering"):
            rep.save("landlord_clustering", landlord_clustering(table, h_ids, rep))
        with rep.guard("Logistic model"):
            if table[severe_col].notna().sum() and table[severe_col].nunique() == 2:
                terms, auc, _ = logistic_model(table, severe_col, h_ids, rep)
                rep.save("logistic_severe_damp", terms)
                rep.save("logistic_cv_auc", auc)
            else:
                rep.skip("Logistic model", f"{severe_col} has fewer than two observed levels")

    if awaab is not None and len(awaab):
        with rep.guard("Awaab duty summary"):
            rep.save("awaab_summary", awaab_summary(awaab, rep))

    rates_path = settings.work("outputs", "landlord_rates.csv")
    if rates_path.exists():
        with rep.guard("Landlord rates"):
            shutil.copyfile(rates_path, out / "tables" / "landlord_rates.csv")
            rep.tables["landlord_rates"] = pd.read_csv(rates_path)
    else:
        rep.skip("Landlord rates", "run denominators.rsh.run_denominators with paths.rsh_sdr / paths.lahs set")

    if "hypothesis_prevalence" in rep.tables and not rep.tables["hypothesis_prevalence"].empty:
        with rep.guard("Prevalence figure"):
            fig_prevalence(rep.tables["hypothesis_prevalence"], out / "figures" / "prevalence.png", rep)
            rep.figures.append("prevalence.png")
    if "severe_by_year" in rep.tables:
        with rep.guard("Severe-by-year figure"):
            fig_severe_by_year(rep.tables["severe_by_year"], out / "figures" / "severe_by_year.png")
            rep.figures.append("severe_by_year.png")
    if table is not None and table["duration_days"].notna().any():
        with rep.guard("Duration figure"):
            fig_duration(table, severe_col, out / "figures" / "duration_box.png")
            rep.figures.append("duration_box.png")

    model_rule = (f"An H code enters the model when at least {100 * (1 - MODEL_MAX_UNCLEAR):.0f}% of decisions "
                  f"answer it yes/no and its rarer answer occurs at least {MODEL_MIN_LEVEL} times; this rule looks "
                  "only at the predictors, never the outcome.")
    coders = set(table["coder"].dropna()) if table is not None else set()
    versions = sorted(set(table["codebook_version"].dropna())) if table is not None else []
    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "synthetic": "synthetic" in coders or any(str(v).startswith("synthetic") for v in versions),
        "codebook_versions": ", ".join(versions),
        "codebook_file_version": cb_version,
        "codebook_sha256": cb_sha,
        "methods": _methods(rep, weighted, severe_source, model_rule),
    }
    summary = out / "summary.md"
    summary.write_text(_summary(rep, inv, meta), encoding="utf-8")
    return summary
