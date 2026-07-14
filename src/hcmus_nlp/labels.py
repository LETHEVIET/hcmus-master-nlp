"""Mapping config loader — single source of truth cho internal → submission.

`config/mapping.toml` đọc bằng `tomllib` (stdlib 3.11+). Không YAML, không
fallback dict. Có `confirmed` flag để chặn `--mode final` cho tới khi giảng
viên duyệt mapping.

Loader: `hcmus_nlp.labels.load_mapping(path)`. Cache theo path để tránh đọc
lại nhiều lần.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
from pathlib import Path

# Tập label hợp lệ ở mỗi phía.
INTERNAL_LABELS: frozenset[str] = frozenset(
    {
        "PERSON",
        "LOCATION",
        "POLITY",
        "DYNASTY",
        "OFFICIAL_TITLE",
        "BOOK",
        "TIME",
        "NUMBER",
    }
)

SUBMISSION_LABELS: frozenset[str] = frozenset({"PER", "LOC", "ORG", "TITLE", "TME", "NUM"})


class MappingError(ValueError):
    """Lỗi đọc/validate mapping TOML."""


@dataclass(frozen=True)
class Mapping:
    version: str
    confirmed: bool
    internal_to_submission: dict[str, str]
    unresolved_conflict_policy: str  # "exclude" | "include_first" | "include_all"
    compatible_groups: dict[str, list[str]] = field(default_factory=dict)
    priority_label_order: dict[str, int] = field(default_factory=dict)

    def is_confirmed(self) -> bool:
        return self.confirmed

    def to_submission(self, internal_label: str) -> str | None:
        return self.internal_to_submission.get(internal_label)

    def labels_compatible(self, a: str, b: str) -> bool:
        """Hai internal label compatible không? (cùng nhóm policy).

        Cấu trúc compatible_groups: `{primary: [others...]}`. Hai label
        compatible nếu `a` là primary và `b` nằm trong list, hoặc ngược lại.
        """
        if a == b:
            return True
        if b in self.compatible_groups.get(a, ()):
            return True
        if a in self.compatible_groups.get(b, ()):
            return True
        return False

    def preferred_label(self, a: str, b: str) -> str:
        """Hai label compatible → trả label có priority thấp hơn."""
        pa = self.priority_label_order.get(a, 99)
        pb = self.priority_label_order.get(b, 99)
        return a if pa <= pb else b


# Cache đơn giản theo path; mtime reset khi re-read.
_CACHE: dict[str, tuple[float, Mapping]] = {}


def load_mapping(path: str | Path) -> Mapping:
    """Load mapping TOML. Validate đầy đủ.

    Raises:
        FileNotFoundError: file không tồn tại.
        MappingError: cấu trúc TOML không hợp lệ, label không match schema,
            internal label chưa khai báo, vv.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(
            f"Mapping file not found: {p}. "
            "Tạo mới từ config/mapping.toml.example hoặc copy từ git history."
        )

    mtime = p.stat().st_mtime
    cache_key = str(p)
    cached = _CACHE.get(cache_key)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    with p.open("rb") as f:
        data = tomllib.load(f)

    if not isinstance(data, MappingABC):
        raise MappingError(f"Mapping root must be a table, got {type(data).__name__}")

    version = data.get("version")
    if not isinstance(version, str):
        raise MappingError("`version` must be a string")
    confirmed = data.get("confirmed")
    if not isinstance(confirmed, bool):
        raise MappingError("`confirmed` must be a boolean")

    raw_mapping = data.get("mapping")
    if not isinstance(raw_mapping, MappingABC):
        raise MappingError("`[mapping]` table is required")
    if not raw_mapping:
        raise MappingError("`[mapping]` table is empty")

    internal_to_submission: dict[str, str] = {}
    for internal, target in raw_mapping.items():
        if internal not in INTERNAL_LABELS:
            raise MappingError(
                f"Internal label {internal!r} is not in INTERNAL_LABELS. "
                f"Known: {sorted(INTERNAL_LABELS)}"
            )
        if target not in SUBMISSION_LABELS:
            raise MappingError(
                f"Submission label {target!r} for {internal!r} is not in "
                f"SUBMISSION_LABELS. Known: {sorted(SUBMISSION_LABELS)}"
            )
        internal_to_submission[internal] = target

    missing = INTERNAL_LABELS - set(internal_to_submission.keys())
    if missing:
        raise MappingError(
            f"Missing internal labels in [mapping]: {sorted(missing)}. "
            "Mọi INTERNAL_LABELS phải có target trong SUBMISSION_LABELS."
        )

    policy = data.get("policy", {})
    if not isinstance(policy, MappingABC):
        raise MappingError("`[policy]` must be a table")
    unresolved = policy.get("unresolved_conflict", "exclude")
    if unresolved not in {"exclude", "include_first", "include_all"}:
        raise MappingError(
            f"`policy.unresolved_conflict` must be one of "
            "exclude | include_first | include_all, got "
            f"{unresolved!r}"
        )

    label_groups = data.get("label_groups", {})
    if not isinstance(label_groups, MappingABC):
        raise MappingError("`[label_groups]` must be a table")
    compatible_groups: dict[str, list[str]] = {}
    if "compatible" in label_groups:
        compatible = label_groups["compatible"]
        if not isinstance(compatible, MappingABC):
            raise MappingError("`[label_groups.compatible]` must be a table")
        for group_name, members in compatible.items():
            if not isinstance(members, list):
                raise MappingError(f"`label_groups.compatible.{group_name}` must be a list")
            for m in members:
                if m not in INTERNAL_LABELS:
                    raise MappingError(
                        f"label_groups.compatible.{group_name} contains "
                        f"{m!r} which is not in INTERNAL_LABELS"
                    )
            compatible_groups[group_name] = list(members)

    priority_label_order: dict[str, int] = {}
    if "priority_label_order" in label_groups:
        prio = label_groups["priority_label_order"]
        if not isinstance(prio, MappingABC):
            raise MappingError("`[label_groups.priority_label_order]` must be a table")
        for label, rank in prio.items():
            if label not in INTERNAL_LABELS:
                raise MappingError(
                    f"label_groups.priority_label_order.{label!r} is not in INTERNAL_LABELS"
                )
            if not isinstance(rank, int):
                raise MappingError(
                    f"label_groups.priority_label_order.{label} must be int, got {type(rank).__name__}"
                )
            priority_label_order[label] = rank

    mapping = Mapping(
        version=version,
        confirmed=confirmed,
        internal_to_submission=internal_to_submission,
        unresolved_conflict_policy=unresolved,
        compatible_groups=compatible_groups,
        priority_label_order=priority_label_order,
    )
    _CACHE[cache_key] = (mtime, mapping)
    return mapping


def reset_cache() -> None:
    """Xóa cache. Dùng cho test hoặc khi cần reload sau khi file đổi ngoài
    ý muốn (vd build script modify file)."""
    _CACHE.clear()


__all__ = [
    "INTERNAL_LABELS",
    "SUBMISSION_LABELS",
    "Mapping",
    "MappingError",
    "load_mapping",
    "reset_cache",
]
