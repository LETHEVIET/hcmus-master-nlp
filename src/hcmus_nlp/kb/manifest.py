"""KBManifest — deterministic metadata cho external KB.

Mỗi KB khi ingest sẽ ghi 1 manifest với:
- name, version, source_url, license
- file_sha256, file_size, row_counts

KHÔNG có `built_at` (timestamp) để reproducible. Identity là
`name + version + file_sha256 + file_size`.

Cache format khuyến nghị:
- seed: JSONL.gz (human-readable)
- cbdb / chgis: SQLite (query + audit)
KHÔNG dùng pickle cho external data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class KBManifest:
    name: str
    version: str
    source_url: str | None
    license: str
    file_sha256: str
    file_size: int
    row_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        # sort_keys=True để deterministic giữa các lần ghi.
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    def write(self, path: Path) -> None:
        path.write_text(self.to_json() + "\n", encoding="utf-8")


def sha256_of_file(path: Path) -> tuple[str, int]:
    """Trả (sha256_hex, file_size). Đọc 1 lần, không load vào RAM."""
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def load_manifest(path: Path) -> KBManifest:
    """Đọc manifest từ file JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return KBManifest(
        name=data["name"],
        version=data["version"],
        source_url=data.get("source_url"),
        license=data["license"],
        file_sha256=data["file_sha256"],
        file_size=data["file_size"],
        row_counts=data.get("row_counts", {}),
    )


__all__ = ["KBManifest", "load_manifest", "sha256_of_file"]
