"""Landlord stock denominators: RSH Statistical Data Return (SDR) and LAHS.

Housing associations' stock comes from the Regulator of Social Housing's SDR
provider-level file; local authorities' own stock from the DLUHC/MHCLG Local
Authority Housing Statistics (LAHS). Column names change between releases,
so the loaders find columns by keyword and fail loudly, listing what they saw.

Matching a landlord as named in a decision to a stock entity goes:
alias table (codebook/entity_aliases.yaml) -> exact normalised name ->
rapidfuzz ``token_set_ratio``. Anything below the threshold, tied, or in
conflict is left unmatched and written out for manual review.

Rates are pooled-period counts per 1,000 homes against one stock snapshot
(the latest year in the stock file), with exact Poisson (Garwood) intervals.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rapidfuzz import fuzz, process
from scipy.stats import chi2

from hos_pipeline.config import Settings
from hos_pipeline.io_utils import read_jsonl

log = logging.getLogger(__name__)

STOCK_COLUMNS = ["entity", "entity_id", "source", "stock", "year"]
DEFAULT_MIN_STOCK = 1000

SDR_KEYWORDS = {
    "name": ["provider name", "organisation name", "organization name", "registered provider", "landlord name",
             "rp name", "provider", "organisation", "name"],
    "id": ["registration number", "registration no", "reg no", "reg. no", "rp code", "provider code",
           "registration", "rp number"],
    "stock": ["total social stock owned", "total social housing stock owned", "total social housing units owned",
              "total social stock", "total stock owned", "social stock owned", "total units owned", "total owned",
              "units owned"],
}
LAHS_KEYWORDS = {
    "name": ["local authority name", "authority name", "la name", "local authority", "organisation name",
             "area name", "council", "name"],
    "id": ["ons code", "gss code", "la code", "area code", "e-code", "ecode", "ons", "code"],
    "stock": ["local authority owned dwelling stock", "la owned dwelling stock", "local authority owned",
              "la owned", "total dwelling stock", "dwelling stock", "total stock", "stock"],
}
_EXCLUDE = {
    "name": ["code", "number", "type", "status", "size", "date"],
    "id": ["name", "date"],
    "stock": ["%", "percent", "proportion", "change", "vacant"],
}

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _cell(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def _find_col(columns: Sequence[str], keywords: Sequence[str], role: str) -> str | None:
    lowered = {c: re.sub(r"\s+", " ", c.lower()) for c in columns}
    for kw in keywords:
        hits = [c for c, low in lowered.items() if kw in low and not any(x in low for x in _EXCLUDE[role])]
        if hits:
            exact = [c for c in hits if lowered[c] == kw]
            return (exact or hits)[0]
    return None


def _header_score(row: Sequence[Any], keywords: Mapping[str, list[str]]) -> int:
    cells = [_cell(v).lower() for v in row]
    return sum(any(kw in c for c in cells for kw in kws) for kws in keywords.values())


def _read_table(path: Path, sheet: str | None, keywords: Mapping[str, list[str]]) -> tuple[pd.DataFrame, str]:
    """Read csv/xlsx and locate the header row (title rows above it are common)."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tsv"}:
        raw = {"csv": pd.read_csv(path, header=None, dtype=object, sep=None, engine="python",
                                  encoding_errors="replace")}
    elif suffix in {".xlsx", ".xlsm", ".xls", ".ods"}:
        got = pd.read_excel(path, sheet_name=sheet if sheet is not None else None, header=None, dtype=object)
        raw = got if isinstance(got, dict) else {str(sheet): got}
    else:
        raise ValueError(f"unsupported stock file type {path.suffix!r}: use .csv or .xlsx")
    best = (-1, "", 0)
    for name, frame in raw.items():
        for i in range(min(30, len(frame))):
            score = _header_score(frame.iloc[i].tolist(), keywords)
            if score > best[0]:
                best = (score, name, i)
    _, name, row = best
    frame = raw[name]
    header, seen = [], {}
    for v in frame.iloc[row].tolist():
        h = _cell(v) or "unnamed"
        seen[h] = seen.get(h, 0) + 1
        header.append(h if seen[h] == 1 else f"{h}.{seen[h] - 1}")
    body = frame.iloc[row + 1:].copy()
    body.columns = header
    return body.reset_index(drop=True), name


def _year_from(path: Path) -> int | None:
    m = re.search(r"(20\d{2})[-_ /](\d{2})(?!\d)", path.name)
    if m:
        return 2000 + int(m.group(2))
    m = re.search(r"20\d{2}", path.name)
    return int(m.group(0)) if m else None


def _to_number(s: pd.Series) -> pd.Series:
    cleaned = s.astype(object).map(lambda v: re.sub(r"[,\s£]", "", _cell(v)))
    return pd.to_numeric(cleaned.replace({"": None, "-": None, "..": None, "x": None}), errors="coerce")


def _load_stock(
    path: str | Path,
    source: str,
    keywords: Mapping[str, list[str]],
    name_col: str | None,
    id_col: str | None,
    stock_cols: list[str] | None,
    sheet: str | None,
    year: int | None,
) -> pd.DataFrame:
    path = Path(path)
    df, sheet_used = _read_table(path, sheet, keywords)
    cols = list(df.columns)

    def need(found: str | None, role: str, param: str) -> str:
        if found is None or found not in cols:
            raise ValueError(
                f"{source.upper()} file {path.name} (sheet {sheet_used!r}): could not identify the {role} column. "
                f"Available columns: {cols}. Pass {param}= explicitly."
            )
        return found

    name = need(name_col or _find_col(cols, keywords["name"], "name"), "entity name", "name_col")
    ident = id_col or _find_col(cols, keywords["id"], "id")
    if ident is not None and ident not in cols:
        need(None, "entity id", "id_col")
    if stock_cols:
        for c in stock_cols:
            need(c, f"stock ({c})", "stock_cols")
        stock = sum(_to_number(df[c]).fillna(0) for c in stock_cols)
        stock = stock.where(pd.concat([_to_number(df[c]) for c in stock_cols], axis=1).notna().any(axis=1))
    else:
        stock = _to_number(df[need(_find_col(cols, keywords["stock"], "stock"), "stock", "stock_cols")])
    out = pd.DataFrame({
        "entity": df[name].map(_cell),
        "entity_id": df[ident].map(_cell) if ident else "",
        "source": source,
        "stock": stock,
        "year": year if year is not None else _year_from(path),
    })
    total_row = out["entity"].str.lower().str.fullmatch(r"(grand )?totals?|england|all .*")
    out = out[(out["entity"] != "") & out["stock"].notna() & ~total_row]
    out["year"] = out["year"].astype("Int64")
    return out[STOCK_COLUMNS].reset_index(drop=True)


def load_sdr(
    path: str | Path,
    name_col: str | None = None,
    id_col: str | None = None,
    stock_cols: list[str] | None = None,
    sheet: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """RSH SDR provider-level stock -> [entity, entity_id, source='sdr', stock, year].

    Columns are auto-detected (provider name, registration number, total social
    stock owned) unless given; ``stock_cols`` with several columns are summed.
    ``year`` defaults to the as-at year in the file name ("2023-24" -> 2024).
    """
    return _load_stock(path, "sdr", SDR_KEYWORDS, name_col, id_col, stock_cols, sheet, year)


def load_lahs(
    path: str | Path,
    name_col: str | None = None,
    id_col: str | None = None,
    stock_cols: list[str] | None = None,
    sheet: str | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    """LAHS local-authority-owned dwelling stock -> [entity, entity_id, source='lahs', stock, year]."""
    return _load_stock(path, "lahs", LAHS_KEYWORDS, name_col, id_col, stock_cols, sheet, year)


# --------------------------------------------------------------------------
# Names and aliases
# --------------------------------------------------------------------------

_LEGAL_SUFFIXES = [
    ("community", "interest", "company"), ("community", "benefit", "society"), ("industrial", "and", "provident",
                                                                                "society"),
    ("limited",), ("ltd",), ("plc",), ("llp",), ("cic",), ("inc",), ("incorporated",),
]
_SECTOR_SUFFIXES = [("housing", "association"), ("ha",)]
_FILLER = {"the", "of"}
# Tokens too generic to identify a landlord on their own: a name is never
# reduced to only these, so e.g. "Community Housing Association" keeps its suffix.
_GENERIC = {
    "community", "city", "county", "district", "borough", "council", "housing", "homes", "home", "group", "trust",
    "and", "north", "south", "east", "west", "central", "new", "national", "regional", "local", "united", "royal",
    "rural", "urban", "first", "society", "partnership", "living", "london", "english", "england", "association",
}
_COUNCIL_RE = re.compile(r"\b(council|borough|city of|district|county|corporation of|royal borough)\b", re.I)


def normalise_name(name: str | None) -> str:
    """Canonical form for matching landlord names.

    Lowercases, strips accents and punctuation, "&"/"+" -> "and", drops
    "the"/"of", trailing legal forms (limited, ltd, plc, CIC, ...) and a
    trailing "housing association"/"HA" when a distinctive word remains.
    "council", "trust", "group" and "homes" are kept, so different landlords
    sharing a stem (e.g. "Southern Housing" vs "Southern Housing Group") stay distinct.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['’`]", "", s)
    s = s.replace("&", " and ").replace("+", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [t for t in s.split() if t not in _FILLER]
    changed = True
    while changed and tokens:
        changed = False
        for suffix in _LEGAL_SUFFIXES:
            if len(tokens) > len(suffix) and tuple(tokens[-len(suffix):]) == suffix:
                tokens = tokens[: -len(suffix)]
                changed = True
    for suffix in _SECTOR_SUFFIXES:
        rest = tokens[: -len(suffix)]
        if tuple(tokens[-len(suffix):]) == suffix and any(t not in _GENERIC for t in rest):
            tokens = rest
            break
    return " ".join(tokens)


def _is_council(name: str) -> bool:
    return bool(_COUNCIL_RE.search(name or ""))


def _period(value: Any) -> tuple[date | None, date | None]:
    if not value:
        return None, None
    f, t = value.get("from"), value.get("to")
    as_date = lambda v: v if isinstance(v, date) or v is None else date.fromisoformat(str(v))  # noqa: E731
    return as_date(f), as_date(t)


def load_aliases(path: str | Path | None) -> dict[str, list[dict]]:
    """Read entity_aliases.yaml -> {normalised HOS name: [alias record, ...]}.

    Each record: canonical, source, stock_names, entity_id, reason, verify,
    period_from, period_to. Several records for one name are allowed when
    their periods differ.
    """
    if path is None or not Path(path).exists():
        return {}
    return _coerce_aliases(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _coerce_aliases(aliases: Mapping[str, Any]) -> dict[str, list[dict]]:
    """Accept the YAML structure, an already-flattened map, or a plain {name: canonical} dict."""
    out: dict[str, list[dict]] = {}

    def add(name: str, rec: dict) -> None:
        out.setdefault(normalise_name(name), []).append(rec)

    if "entities" in aliases:
        for ent in aliases.get("entities") or []:
            base = {
                "canonical": ent["canonical"],
                "source": ent.get("source"),
                "stock_names": list(ent.get("stock_names") or [ent["canonical"]]),
                "entity_id": _cell(ent.get("entity_id")) or None,
                "reason": ent.get("reason", ""),
                "verify": bool(ent.get("verify", False)),
            }
            pf, pt = _period(ent.get("period"))
            add(ent["canonical"], {**base, "period_from": pf, "period_to": pt})
            for v in ent.get("variants") or []:
                v = {"name": v} if isinstance(v, str) else v
                vf, vt = _period(v.get("period")) if v.get("period") else (pf, pt)
                add(v["name"], {**base, "reason": v.get("reason", base["reason"]),
                                "verify": bool(v.get("verify", base["verify"])), "period_from": vf, "period_to": vt})
        return out
    for name, value in aliases.items():
        recs = value if isinstance(value, list) else [value]
        for rec in recs:
            if isinstance(rec, str):
                rec = {"canonical": rec}
            rec = {"source": None, "stock_names": [rec["canonical"]], "entity_id": None, "reason": "",
                   "verify": False, "period_from": None, "period_to": None, **rec}
            add(name, rec)
    return out


def _pick_alias(records: list[dict], when: date | None) -> dict | None:
    for rec in records:
        f, t = rec.get("period_from"), rec.get("period_to")
        if when is None or ((f is None or when >= f) and (t is None or when <= t)):
            return rec
    return None


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

MATCH_COLUMNS = ["hos_name", "decision_date", "normalised", "entity", "entity_id", "source", "score", "method",
                 "matched", "canonical", "alias_reason", "verify", "candidates"]


class _Matcher:
    def __init__(self, stock: pd.DataFrame, threshold: int):
        cands = stock[["entity", "entity_id", "source"]].drop_duplicates(["entity", "source"]).reset_index(drop=True)
        cands["norm"] = cands["entity"].map(normalise_name)
        self.c = cands
        self.threshold = threshold
        self.by_norm: dict[str, list[int]] = {}
        for i, n in enumerate(cands["norm"]):
            self.by_norm.setdefault(n, []).append(i)
        self.by_id = {_cell(v): i for i, v in enumerate(cands["entity_id"]) if _cell(v)}

    def _row(self, idx: int | None, score: float, method: str, cands: str = "") -> dict:
        if idx is None:
            return {"entity": None, "entity_id": None, "source": None, "score": score, "method": method,
                    "matched": False, "candidates": cands}
        r = self.c.iloc[idx]
        return {"entity": r["entity"], "entity_id": r["entity_id"] or None, "source": r["source"], "score": score,
                "method": method, "matched": True, "candidates": cands}

    def _pool(self, source: str | None) -> list[int]:
        return [i for i in range(len(self.c)) if source is None or self.c.at[i, "source"] == source]

    def name(self, norm: str, prefer: str | None, only: str | None = None) -> dict:
        if not norm:
            return self._row(None, 0.0, "unmatched")
        exact = [i for i in self.by_norm.get(norm, []) if only is None or self.c.at[i, "source"] == only]
        if exact:
            preferred = [i for i in exact if self.c.at[i, "source"] == prefer] or exact
            if len({self.c.at[i, "entity"] for i in preferred}) == 1:
                return self._row(preferred[0], 100.0, "exact")
            return self._row(None, 100.0, "ambiguous", "; ".join(self.c.loc[preferred, "entity"]))
        for source in ([prefer] if prefer and only is None else []) + [only]:
            pool = self._pool(source)
            if not pool:
                continue
            choices = {i: self.c.at[i, "norm"] for i in pool}
            hits = process.extract(norm, choices, scorer=fuzz.token_set_ratio, limit=5)
            shown = "; ".join(f"{self.c.at[i, 'entity']} ({s:.0f})" for _, s, i in hits[:3])
            if not hits or hits[0][1] < self.threshold:
                if source == only:
                    return self._row(None, hits[0][1] if hits else 0.0, "unmatched", shown)
                continue
            top = [i for _, s, i in hits if s == hits[0][1]]
            if len({self.c.at[i, "entity"] for i in top}) > 1:
                return self._row(None, hits[0][1], "ambiguous", shown)
            return self._row(top[0], float(hits[0][1]), "fuzzy", shown)
        return self._row(None, 0.0, "unmatched")

    def alias(self, rec: dict) -> dict:
        by_name = None
        for sn in rec["stock_names"]:
            r = self.name(normalise_name(sn), rec.get("source"), only=rec.get("source"))
            if r["matched"]:
                by_name = r
                break
        id_idx = self.by_id.get(_cell(rec.get("entity_id"))) if rec.get("entity_id") else None
        if by_name and id_idx is not None and self.c.at[id_idx, "entity"] != by_name["entity"]:
            return self._row(None, 0.0, "alias_conflict",
                             f"name -> {by_name['entity']}; id {rec['entity_id']} -> {self.c.at[id_idx, 'entity']}")
        if by_name:
            return {**by_name, "method": "alias"}
        if id_idx is not None:
            return self._row(id_idx, 100.0, "alias_id")
        return self._row(None, 0.0, "alias_unresolved", f"alias -> {rec['canonical']} not found in stock")


def match_entities(
    hos_names: list[str],
    stock: pd.DataFrame,
    aliases: Mapping[str, Any] | None,
    threshold: int = 90,
    *,
    dates: Sequence[date | None] | None = None,
    review_path: str | Path | None = None,
) -> pd.DataFrame:
    """Match HOS landlord names to stock entities; one output row per input name.

    Order: alias table (honouring each alias's ``period`` when ``dates`` are
    given), exact normalised name, then rapidfuzz token_set_ratio >= threshold.
    Council-like names prefer LAHS entities, others SDR. Ties between
    different entities and alias id/name conflicts are never auto-resolved.
    Unmatched names are written to ``review_path`` (one row per name).
    """
    alias_map = _coerce_aliases(aliases or {})
    m = _Matcher(stock, threshold)
    cache: dict[str, dict] = {}
    rows = []
    for i, name in enumerate(hos_names):
        when = dates[i] if dates is not None else None
        norm = normalise_name(name)
        rec = _pick_alias(alias_map.get(norm, []), when)
        if rec is not None:
            res = {**m.alias(rec), "canonical": rec["canonical"], "alias_reason": rec.get("reason", ""),
                   "verify": rec.get("verify", False)}
        else:
            if norm not in cache:
                cache[norm] = {**m.name(norm, "lahs" if _is_council(name) else "sdr"), "canonical": None,
                               "alias_reason": "", "verify": False}
            res = cache[norm]
        rows.append({"hos_name": name, "decision_date": when, "normalised": norm, **res})
    out = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    if review_path is not None:
        unmatched = out[~out["matched"].astype(bool)]
        review = (
            unmatched.groupby(["hos_name", "normalised", "method", "candidates"], dropna=False)
            .size().rename("n_decisions").reset_index().sort_values("n_decisions", ascending=False)
        )
        Path(review_path).parent.mkdir(parents=True, exist_ok=True)
        review.to_csv(review_path, index=False)
    return out


# --------------------------------------------------------------------------
# Rates
# --------------------------------------------------------------------------


def garwood_ci(k: float, exposure: float = 1.0, alpha: float = 0.05) -> tuple[float, float]:
    """Exact Poisson (Garwood 1936) interval for a rate ``k / exposure``."""
    if not exposure or exposure <= 0 or np.isnan(k):
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else chi2.ppf(alpha / 2, 2 * k) / 2
    hi = chi2.ppf(1 - alpha / 2, 2 * k + 2) / 2
    return float(lo / exposure), float(hi / exposure)


def rates_per_1000(
    counts: pd.DataFrame,
    stock: pd.DataFrame,
    min_stock: int = DEFAULT_MIN_STOCK,
    count_cols: Sequence[str] = ("n_damp_decisions", "n_severe_damp"),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Counts per 1,000 homes with Garwood 95% CIs.

    ``counts`` has ``entity`` (and optionally ``source``) plus the count
    columns. Stock is the latest year per entity. Entities with stock below
    ``min_stock`` stay in the output but are excluded from the rankings
    (``ranked`` False), since tiny denominators give unstable rates.
    """
    keys = ["entity", "source"] if "source" in counts.columns else ["entity"]
    latest = stock.sort_values("year", na_position="first").drop_duplicates(keys, keep="last")
    df = counts.merge(latest[[*keys, "entity_id", "stock", "year"]].rename(columns={"year": "stock_year"}),
                      on=keys, how="left")
    df["ranked"] = df["stock"].fillna(0) >= min_stock
    for col in count_cols:
        if col not in df.columns:
            continue
        base = col.removeprefix("n_")
        df[f"{base}_per_1000"] = df[col] / df["stock"] * 1000
        ci = [garwood_ci(k, s / 1000, alpha) if pd.notna(s) else (np.nan, np.nan) for k, s in zip(df[col], df["stock"])]
        df[f"{base}_per_1000_lo"] = [c[0] for c in ci]
        df[f"{base}_per_1000_hi"] = [c[1] for c in ci]
        rank = df.loc[df["ranked"], f"{base}_per_1000"].rank(ascending=False, method="min")
        df[f"{base}_rank"] = rank.reindex(df.index).astype("Int64")
    first = f"{count_cols[0].removeprefix('n_')}_per_1000"
    return df.sort_values(["ranked", first], ascending=[False, False]).reset_index(drop=True)


def run_denominators(settings: Settings, min_stock: int = DEFAULT_MIN_STOCK, threshold: int = 90) -> Path | None:
    """Match damp-flagged decisions to stock and write outputs/landlord_rates.csv.

    Counts use the screened population (screen.jsonl damp_flag) for the pooled
    years; severe findings come from outcomes.jsonl (classifier) when present,
    else the screening keyword flag. Returns None, writing nothing, when no
    stock file is configured.
    """
    frames = []
    for path, loader in ((settings.paths.rsh_sdr, load_sdr), (settings.paths.lahs, load_lahs)):
        if path is None:
            continue
        if not Path(path).exists():
            log.warning("stock file %s does not exist; skipped", path)
            continue
        frames.append(loader(path))
    if not frames:
        log.warning("No RSH SDR or LAHS stock file configured (paths.rsh_sdr / paths.lahs): landlord rates not produced.")
        return None
    stock = pd.concat(frames, ignore_index=True)
    work = settings.paths.work_dir
    screen_path = work / "screen.jsonl"
    if not screen_path.exists():
        log.warning("screen.jsonl not found in %s: landlord rates not produced.", work)
        return None
    years = set(settings.stats.pooled_years)
    damp = [r for r in read_jsonl(screen_path) if r.get("damp_flag") and r.get("year") in years]
    outcomes_path = work / "outcomes.jsonl"
    if outcomes_path.exists():
        severe = {r["case_id"]: bool(r["severe_damp"]) for r in read_jsonl(outcomes_path)}
        severe_source = "outcome classifier"
    else:
        severe = {r["case_id"]: bool(r.get("keyword_severe_damp")) for r in damp}
        severe_source = "screening keywords (grade D)"
    dates: dict[str, date | None] = {}
    if (work / "decisions.jsonl").exists():
        for r in read_jsonl(work / "decisions.jsonl"):
            d = r.get("decision_date")
            dates[r["case_id"]] = date.fromisoformat(d[:10]) if d else None

    matches = match_entities(
        [r["landlord"] for r in damp], stock, load_aliases(settings.paths.entity_aliases), threshold,
        dates=[dates.get(r["case_id"]) for r in damp], review_path=settings.work("entity_match_review.csv"),
    )
    matches.insert(0, "case_id", [r["case_id"] for r in damp])
    matches["severe_damp"] = [severe.get(r["case_id"], False) for r in damp]
    matches.to_csv(settings.work("entity_matches.csv"), index=False)
    hit = matches[matches["matched"].astype(bool)]
    counts = (
        hit.groupby(["entity", "source"])
        .agg(n_damp_decisions=("case_id", "size"), n_severe_damp=("severe_damp", "sum"),
             hos_names=("hos_name", lambda s: "; ".join(sorted(set(s)))))
        .reset_index()
    )
    rates = rates_per_1000(counts, stock, min_stock=min_stock)
    rates["years"] = f"{min(years)}-{max(years)}"
    rates["severe_source"] = severe_source
    rates["min_stock_for_ranking"] = min_stock
    out = settings.work("outputs", "landlord_rates.csv")
    rates.to_csv(out, index=False)
    n_un = int((~matches["matched"].astype(bool)).sum())
    log.info("landlord rates: %d entities; %d of %d decisions unmatched (see entity_match_review.csv)",
             len(rates), n_un, len(matches))
    return out
