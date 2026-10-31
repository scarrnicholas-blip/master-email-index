"""JSON Lines and table helpers shared by every stage."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

M = TypeVar("M", bound=BaseModel)


def write_jsonl(path: str | Path, records: Iterable[BaseModel | dict]) -> int:
    """Write records atomically (temp file then rename). Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            line = rec.model_dump_json() if isinstance(rec, BaseModel) else json.dumps(rec, default=str)
            fh.write(line + "\n")
            n += 1
    tmp.replace(path)
    return n


def append_jsonl(path: str | Path, rec: BaseModel | dict) -> None:
    """Append one record; used by resumable stages so a crash loses at most one record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = rec.model_dump_json() if isinstance(rec, BaseModel) else json.dumps(rec, default=str)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def read_jsonl(path: str | Path, model: type[M] | None = None) -> Iterator[M] | Iterator[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield model.model_validate_json(line) if model else json.loads(line)


def done_ids(path: str | Path, key: str = "case_id") -> set[str]:
    """IDs already present in a JSONL output, so a re-run skips them."""
    path = Path(path)
    if not path.exists():
        return set()
    return {rec[key] for rec in read_jsonl(path) if key in rec}
