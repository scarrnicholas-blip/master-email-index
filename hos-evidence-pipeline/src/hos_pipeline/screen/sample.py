"""Stage 2: stratified random sample of damp-flagged decisions.

Allocation is proportional to stratum size with a per-stratum floor, rounded by
largest remainder so the sample size is exact. Within a stratum the draw is a
seeded permutation of the sorted case ids, so it is reproducible and a top-up
takes the next ids in ``draw_order``.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..config import Settings
from ..io_utils import read_jsonl, write_jsonl
from ..models import SampleRecord, ScreenResult

LANDLORD_TYPES = ("local_authority", "housing_association", "almo_tmo", "other")

# Arm's-length management organisations, current or former. Only names known to
# have been ALMOs; most would otherwise read as housing associations ("Homes").
KNOWN_ALMOS = (
    "aire valley homes", "ascham homes", "ashfield homes", "barnet homes", "berneslai homes",
    "blackpool coastal housing", "brent housing partnership", "cheltenham borough homes", "citywest homes",
    "colchester borough homes", "dale and valley homes", "derby homes", "durham city homes",
    "ealing homes", "east durham homes", "east north east homes", "eastbourne homes", "enfield homes",
    "gateshead housing company", "hackney homes", "homes for haringey", "homes for islington",
    "homes for northumberland", "homes in sedgemoor", "hounslow homes", "kirklees neighbourhood housing",
    "lambeth living", "lewisham homes", "newark and sherwood homes", "northwards housing",
    "nottingham city homes", "sheffield homes", "six town housing", "solihull community housing",
    "south essex homes", "st leger homes", "stockport homes", "tower hamlets homes",
    "west north west homes", "wolverhampton homes", "your homes newcastle",
)
_ALMO_RE = re.compile(r"\b(?:tmo|almo|tenant management organisation|arm'?s[- ]length management)\b")
_LA_RE = re.compile(r"\b(?:council|borough of|corporation of|city of london corporation|county of)\b")
_HA_RE = re.compile(
    r"\b(?:housing|homes?|living|group|limited|ltd|plc|association|trust|partnership|society"
    r"|co-?operative|almshouses?)\b"
)


def landlord_type(name: str) -> str:
    """local_authority, housing_association, almo_tmo or other, from the landlord's name."""
    n = re.sub(r"\s+", " ", (name or "").lower().replace("’", "'").replace("&", " and ")).strip()
    if not n:
        return "other"
    if _ALMO_RE.search(n) or any(a in n for a in KNOWN_ALMOS):
        return "almo_tmo"
    if _LA_RE.search(n):
        return "local_authority"
    if _HA_RE.search(n):
        return "housing_association"
    return "other"


def stratum_values(result: ScreenResult, strata: list[str]) -> list[str]:
    values = []
    for name in strata:
        if name == "landlord_type":
            value = landlord_type(result.landlord)
        elif name in ScreenResult.model_fields:
            value = getattr(result, name)
        else:
            raise ValueError(f"unknown stratum field {name!r}; use landlord_type or a ScreenResult field")
        values.append("unknown" if value is None else str(value))
    return values


def allocate(sizes: dict[str, int], target: int, floor: int = 0) -> dict[str, int]:
    """Split ``target`` across strata in proportion to size, each getting at least
    min(floor, size) and at most its size. Sums to min(target, total) exactly."""
    target = max(0, min(target, sum(sizes.values())))
    floors = {k: min(floor, n) for k, n in sizes.items()}
    if sum(floors.values()) >= target:
        # Too many strata for every floor: share the target in proportion to the floors.
        total = sum(floors.values()) or 1
        return _largest_remainder({k: Fraction(target * f, total) for k, f in floors.items()}, target, floors)

    # Pin strata whose proportional share breaks a cap, then a floor, and re-share the rest.
    fixed: dict[str, int] = {}
    while True:
        free = [k for k in sizes if k not in fixed]
        rem = target - sum(fixed.values())
        pop = sum(sizes[k] for k in free)
        quotas = {k: Fraction(rem * sizes[k], pop) if pop else Fraction(0) for k in free}
        over = {k: sizes[k] for k in free if quotas[k] > sizes[k]}
        under = {k: floors[k] for k in free if quotas[k] < floors[k]}
        if not (over or under):
            break
        fixed.update(over or under)
    quotas.update({k: Fraction(v) for k, v in fixed.items()})
    return _largest_remainder(quotas, target, sizes)


def stratified_sample(
    results: list[ScreenResult],
    strata: list[str],
    target_n: int,
    seed: int,
    min_per_stratum: int,
) -> list[SampleRecord]:
    """Seeded stratified sample of the damp-flagged results."""
    groups = group_strata(results, strata)
    alloc = allocate({k: len(v) for k, v in groups.items()}, target_n, min_per_stratum)
    records = []
    for key, ids in groups.items():
        n = alloc[key]
        for rank, case_id in enumerate(_draw(ids, seed, key)[:n]):
            records.append(SampleRecord(case_id=case_id, stratum=key, population_n=len(ids), sample_n=n,
                                        weight=len(ids) / n, draw_order=rank))
    return records


def irr_subsample(sample: list[SampleRecord], fraction: float = 0.10, seed: int = 20261030) -> list[str]:
    """Case ids for the second human reader: round(fraction x sample), spread across strata
    in proportion to each stratum's share of the sample."""
    by_stratum: dict[str, list[str]] = {}
    for r in sample:
        by_stratum.setdefault(r.stratum, []).append(r.case_id)
    k = math.floor(fraction * len(sample) + 0.5)
    alloc = allocate({s: len(ids) for s, ids in by_stratum.items()}, k)
    chosen: list[str] = []
    for s in sorted(by_stratum):
        chosen += _draw(sorted(by_stratum[s]), seed, f"irr|{s}")[: alloc[s]]
    return sorted(chosen)


def group_strata(results: Iterable[ScreenResult], strata: list[str]) -> dict[str, list[str]]:
    """Damp-flagged case ids by stratum key, keys and ids sorted."""
    flagged = {r.case_id: r for r in results if r.damp_flag}
    groups: dict[str, list[str]] = {}
    for case_id in sorted(flagged):
        groups.setdefault("|".join(stratum_values(flagged[case_id], strata)), []).append(case_id)
    return dict(sorted(groups.items()))


def run_sample(settings: Settings) -> Path:
    """screen.jsonl -> sample.jsonl, sample_strata.csv and irr_sample.txt.

    Only decisions whose year is in ``screening.years`` are sampled.
    """
    cfg = settings.sampling
    years = set(settings.screening.years)
    results = [r for r in read_jsonl(settings.work("screen.jsonl"), ScreenResult) if not years or r.year in years]
    records = stratified_sample(results, cfg.strata, cfg.target_n, cfg.seed, cfg.min_per_stratum)
    out = settings.work("sample.jsonl")
    write_jsonl(out, records)
    _write_strata(settings.work("sample_strata.csv"), group_strata(results, cfg.strata), records, cfg.strata)
    irr = irr_subsample(records, 0.10, cfg.seed)
    settings.work("irr_sample.txt").write_text("".join(f"{cid}\n" for cid in irr), encoding="utf-8")
    return out


# --------------------------------------------------------------------------


def _largest_remainder(quotas: dict[str, Fraction], target: int, caps: dict[str, int]) -> dict[str, int]:
    out = {k: math.floor(q) for k, q in quotas.items()}
    short = target - sum(out.values())
    for k in sorted(quotas, key=lambda k: (out[k] - quotas[k], k)):  # largest remainder first
        if short <= 0:
            break
        if quotas[k] > out[k] and out[k] < caps[k]:
            out[k] += 1
            short -= 1
    return out


def _draw(ids: list[str], seed: int, key: str) -> list[str]:
    """``ids`` in a seeded random order. The seed mixes in the stratum key, stably."""
    stratum_seed = int.from_bytes(hashlib.sha256(f"{seed}|{key}".encode()).digest()[:8], "big")
    rng = np.random.default_rng(stratum_seed)
    return [ids[i] for i in rng.permutation(len(ids))]


def _write_strata(path: Path, groups: dict[str, list[str]], records: list[SampleRecord], strata: list[str]) -> None:
    sampled: dict[str, int] = {}
    for r in records:
        sampled[r.stratum] = sampled.get(r.stratum, 0) + 1
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["stratum", *strata, "population_n", "sample_n", "weight", "sampling_fraction"])
        for key, ids in groups.items():
            n = sampled.get(key, 0)
            writer.writerow([key, *key.split("|"), len(ids), n,
                             round(len(ids) / n, 6) if n else "", round(n / len(ids), 6)])
