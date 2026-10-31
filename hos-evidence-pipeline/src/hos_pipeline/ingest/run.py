"""Stage 1: stream the archive into ``decisions.jsonl`` and ``ingest_report.csv``.

Resumable: case ids already in decisions.jsonl are skipped and each decision is
appended as soon as it is parsed. A page that fails to load or parse gets a row
in the report and is retried on the next run.
"""

from __future__ import annotations

import csv
import logging
import zlib
from pathlib import Path

from ..config import Settings
from ..io_utils import append_jsonl, done_ids
from ..models import Decision
from .archive import ArchiveEntry, load_html_with_sha, read_manifest, sha_matches
from .parse import parse_decision, section_kind

log = logging.getLogger(__name__)

REPORT_FIELDS = ["case_id", "n_paragraphs", "n_numbered", "has_determination", "sha256_ok", "warnings"]


def ingest_archive(
    settings: Settings,
    limit: int | None = None,
    case_ids: set[str] | None = None,
    verify_sha: bool = True,
) -> Path:
    out = settings.work("decisions.jsonl")
    report = settings.work("ingest_report.csv")
    entries = read_manifest(settings.paths.archive_dir)
    _drop_partial_line(out)
    done = done_ids(out)
    todo = [e for cid, e in entries.items() if cid not in done and (case_ids is None or cid in case_ids)]
    if limit is not None:
        todo = todo[:limit]
    log.info("%d archive pages, %d already ingested, %d to do", len(entries), len(done), len(todo))

    new_report = not report.exists()
    with report.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        if new_report:
            writer.writeheader()
        for i, entry in enumerate(todo, 1):
            decision, row = _ingest_one(entry, verify_sha)
            if decision is not None:
                append_jsonl(out, decision)
            writer.writerow(row)
            if i % 500 == 0:
                fh.flush()
                log.info("ingested %d/%d", i, len(todo))
    _dedupe_report(report)
    return out


def _ingest_one(entry: ArchiveEntry, verify_sha: bool) -> tuple[Decision | None, dict]:
    row = {"case_id": entry.case_id, "n_paragraphs": 0, "n_numbered": 0, "has_determination": False,
           "sha256_ok": "", "warnings": ""}
    try:
        html, digest = load_html_with_sha(entry)
    except (OSError, EOFError, zlib.error) as exc:
        row["warnings"] = f"error: cannot read {entry.path}: {exc}"
        return None, row
    ok = sha_matches(entry, digest, verify_sha)
    row["sha256_ok"] = "" if ok is None else ok
    try:
        decision = parse_decision(html, entry.case_id, entry.url)
    except Exception as exc:  # noqa: BLE001 - one bad page must not stop 18k
        row["warnings"] = f"error: parse failed: {type(exc).__name__}: {exc}"
        return None, row

    warnings = list(decision.parse_warnings)
    if ok is False:
        warnings.append("sha256 does not match manifest")
    decision = decision.model_copy(update={
        "source_path": str(entry.path),
        "source_sha256": digest,
        "sha256_ok": ok,
        "parse_warnings": warnings,
    })
    row.update(
        n_paragraphs=len(decision.paragraphs),
        n_numbered=len({p.number for p in decision.paragraphs if p.number is not None}),
        has_determination=any(section_kind(p.section) == "determination" for p in decision.paragraphs),
        warnings="; ".join(warnings),
    )
    return decision, row


def _drop_partial_line(path: Path) -> None:
    """Truncate a last line left half-written by an interrupted run."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as fh:
        fh.seek(-1, 2)
        if fh.read(1) == b"\n":
            return
        log.warning("dropping a half-written last line from %s", path)
        end = fh.tell()
        while end > 0:
            start = max(0, end - 65536)
            fh.seek(start)
            cut = fh.read(end - start).rfind(b"\n")
            if cut >= 0:
                fh.truncate(start + cut + 1)
                return
            end = start
        fh.truncate(0)


def _dedupe_report(report: Path) -> None:
    """Keep the latest row per case id (a retried failure appends a second row)."""
    with report.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    latest = {r["case_id"]: r for r in rows}
    if len(latest) == len(rows):
        return
    tmp = report.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(sorted(latest.values(), key=lambda r: r["case_id"]))
    tmp.replace(report)
