"""Read the local NoMug HOS archive: ``manifest.tsv`` plus ``html/<id[:4]>/<id>.html.gz``.

The manifest's sha256 is of the uncompressed page. A case fetched twice has two
manifest rows; the last one wins. Without a manifest the html folder is globbed.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

_GZIP_MAGIC = b"\x1f\x8b"


@dataclass
class ArchiveEntry:
    case_id: str
    url: str | None
    path: Path
    sha256: str | None
    fetched_utc: str | None


def html_path(archive_dir: Path, case_id: str) -> Path:
    return Path(archive_dir) / "html" / case_id[:4] / f"{case_id}.html.gz"


def read_manifest(archive_dir: Path) -> dict[str, ArchiveEntry]:
    """Entries keyed by case id, sorted by case id.

    Pages on disk that the manifest does not list (an interrupted fetch) are
    included with no url or checksum.
    """
    archive_dir = Path(archive_dir).expanduser()
    entries: dict[str, ArchiveEntry] = {}
    manifest = archive_dir / "manifest.tsv"
    if manifest.exists():
        with manifest.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh, delimiter="\t"):
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                case_id = row.get("case_id", "")
                if not case_id:
                    continue
                entries[case_id] = ArchiveEntry(
                    case_id=case_id,
                    url=row.get("url") or None,
                    path=html_path(archive_dir, case_id),
                    sha256=(row.get("sha256") or "").lower() or None,
                    fetched_utc=row.get("fetched_utc") or None,
                )
    for path in (archive_dir / "html").glob("*/*.html.gz"):
        case_id = path.name.removesuffix(".html.gz")
        if case_id not in entries:
            entries[case_id] = ArchiveEntry(case_id, None, path, None, None)
    return dict(sorted(entries.items()))


def read_html_bytes(entry: ArchiveEntry) -> bytes:
    data = entry.path.read_bytes()
    # Tolerate a page saved uncompressed under the .gz name.
    return gzip.decompress(data) if data[:2] == _GZIP_MAGIC else data


def load_html_with_sha(entry: ArchiveEntry) -> tuple[str, str]:
    """The decoded page and the sha256 of its uncompressed bytes."""
    data = read_html_bytes(entry)
    return data.decode("utf-8", errors="replace"), hashlib.sha256(data).hexdigest()


def load_html(entry: ArchiveEntry, verify_sha: bool = True) -> tuple[str, bool | None]:
    """The decoded page, and whether it matches the manifest checksum (None if unchecked)."""
    html, digest = load_html_with_sha(entry)
    return html, sha_matches(entry, digest, verify_sha)


def sha_matches(entry: ArchiveEntry, digest: str, verify_sha: bool = True) -> bool | None:
    if not verify_sha or not entry.sha256:
        return None
    return digest == entry.sha256
