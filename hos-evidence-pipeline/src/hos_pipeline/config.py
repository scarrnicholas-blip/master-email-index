"""Pipeline settings, loaded from a YAML file (see config.example.yaml).

Every path can be absolute or relative to the config file. Defaults match the
NoMug HOS archive layout written by archive.sh:

    ~/NoMug HOS Archive/
        html/<year>/<case id>.html.gz
        manifest.tsv      case_id, url, http_status, bytes, sha256, fetched_utc
        not-found.tsv
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class Paths(BaseModel):
    archive_dir: Path = Path("~/NoMug HOS Archive")
    rulings_tsv: Path | None = Field(default=None, description="Optional: the Check Your Landlord list")
    work_dir: Path = Path("./work")
    codebook: Path = Path("./codebook/codebook.yaml")
    rsh_sdr: Path | None = None
    lahs: Path | None = None
    entity_aliases: Path | None = Path("./codebook/entity_aliases.yaml")
    bank_holidays_json: Path | None = Field(
        default=None, description="Optional gov.uk bank-holidays.json; built-in calendar used if absent"
    )


class Screening(BaseModel):
    # Term groups counted across the full text. Regexes, case-insensitive.
    terms: dict[str, list[str]] = {
        "damp": [r"\bdamp(?:ness)?\b"],
        "mould": [r"\bmou?ld(?:y|s)?\b"],
        "condensation": [r"\bcondensation\b"],
    }
    # A decision is damp-flagged when the complaint definition or a determination
    # names damp/mould, or the combined damp+mould count reaches this threshold.
    min_mentions: int = 3
    years: list[int] = Field(default_factory=lambda: list(range(2019, 2027)))


class Sampling(BaseModel):
    seed: int = 20261030
    # Stratify by these ScreenResult/Decision fields. "landlord_type" is derived
    # from the landlord name (council / housing association / other).
    strata: list[str] = ["year", "landlord_type"]
    target_n: int = 1667
    min_per_stratum: int = 5


class LLM(BaseModel):
    model: str = "claude-opus-5-5"
    verifier_model: str = "claude-opus-5-5"
    classifier_model: str = "claude-opus-5-5"
    effort_coding: str = "high"
    effort_verify: str = "high"
    effort_classify: str = "medium"
    max_tokens: int = 48000
    concurrency: int = 8
    use_batches: bool = Field(default=True, description="Message Batches API: half price, results within 24h")
    server_side_fallback: bool = True


class Awaab(BaseModel):
    commencement: str = "2025-10-27"
    duties_file: Path | None = Field(default=None, description="Override the built-in SI 2025/1042 duty table")


class Stats(BaseModel):
    pooled_years: list[int] = Field(default_factory=lambda: list(range(2019, 2026)))
    permutations: int = 2000
    cv_folds: int = 5
    seed: int = 20261030
    alpha: float = 0.05


class Settings(BaseModel):
    paths: Paths = Paths()
    screening: Screening = Screening()
    sampling: Sampling = Sampling()
    llm: LLM = LLM()
    awaab: Awaab = Awaab()
    stats: Stats = Stats()

    def resolve(self, base: Path) -> "Settings":
        """Expand ~ and make relative paths relative to ``base``."""
        for name, value in self.paths:
            if isinstance(value, Path):
                p = value.expanduser()
                setattr(self.paths, name, p if p.is_absolute() else (base / p).resolve())
        if self.awaab.duties_file is not None:
            p = self.awaab.duties_file.expanduser()
            self.awaab.duties_file = p if p.is_absolute() else (base / p).resolve()
        return self

    def work(self, *parts: str) -> Path:
        path = self.paths.work_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


def load_settings(path: str | Path | None) -> Settings:
    if path is None:
        return Settings().resolve(Path.cwd())
    path = Path(path).expanduser().resolve()
    data = yaml.safe_load(path.read_text()) or {}
    return Settings.model_validate(data).resolve(path.parent)
