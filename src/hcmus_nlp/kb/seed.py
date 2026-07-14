"""Seed gazetteer — curated small lists cho LOCATION / POLITY / DYNASTY.

Dùng để bootstrap trước khi có CBDB/CHGIS. Có thể sử dụng ngay không cần
external download. Mỗi seed có `priority_score = SEED_GAZETTEER` (0.70).

Format cache: JSONL.gz, mỗi dòng `{term, label, alias}`.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hcmus_nlp._weights import SEED_GAZETTEER
from hcmus_nlp.candidates import Trie
from hcmus_nlp.kb.manifest import KBManifest, sha256_of_file
from hcmus_nlp.source_base import AnnotationContext, Candidate, SourceKind


@dataclass(frozen=True)
class SeedEntry:
    term: str
    label: str
    alias: str | None = None


# Seed mặc định — corpus hiện tại (8 tác phẩm lịch sử Trung Quốc).
# Mở rộng khi cần bằng ingest file JSONL.gz.
DEFAULT_LOCATION_SEEDS: tuple[SeedEntry, ...] = (
    SeedEntry("長安", "LOCATION"),
    SeedEntry("洛陽", "LOCATION"),
    SeedEntry("沛", "LOCATION"),
    SeedEntry("南陽郡", "LOCATION"),
    SeedEntry("成都", "LOCATION"),
    SeedEntry("建康", "LOCATION"),
    SeedEntry("汴京", "LOCATION"),
    SeedEntry("臨安", "LOCATION"),
    SeedEntry("大都", "LOCATION"),
    SeedEntry("滄州", "LOCATION"),
)

DEFAULT_POLITY_SEEDS: tuple[SeedEntry, ...] = (
    SeedEntry("漢", "DYNASTY"),
    SeedEntry("魏", "POLITY"),
    SeedEntry("吳", "POLITY"),
    SeedEntry("蜀", "POLITY"),
    SeedEntry("秦", "DYNASTY"),
    SeedEntry("楚", "POLITY"),
    SeedEntry("齊", "POLITY"),
    SeedEntry("梁", "POLITY"),
    SeedEntry("陳", "POLITY"),
    SeedEntry("周", "DYNASTY"),
    SeedEntry("晉", "DYNASTY"),
    SeedEntry("隋", "DYNASTY"),
    SeedEntry("唐", "DYNASTY"),
    SeedEntry("宋", "DYNASTY"),
    SeedEntry("遼", "DYNASTY"),
    SeedEntry("金", "DYNASTY"),
    SeedEntry("元", "DYNASTY"),
    SeedEntry("明", "DYNASTY"),
    SeedEntry("清", "DYNASTY"),
)

DEFAULT_TITLE_SEEDS: tuple[SeedEntry, ...] = (
    SeedEntry("太守", "OFFICIAL_TITLE"),
    SeedEntry("刺史", "OFFICIAL_TITLE"),
    SeedEntry("將軍", "OFFICIAL_TITLE"),
    SeedEntry("大將軍", "OFFICIAL_TITLE"),
    SeedEntry("司馬", "OFFICIAL_TITLE"),
    SeedEntry("尚書", "OFFICIAL_TITLE"),
    SeedEntry("侍郎", "OFFICIAL_TITLE"),
    SeedEntry("丞相", "OFFICIAL_TITLE"),
    SeedEntry("御史", "OFFICIAL_TITLE"),
    SeedEntry("博士", "OFFICIAL_TITLE"),
    SeedEntry("校尉", "OFFICIAL_TITLE"),
    SeedEntry("中郎將", "OFFICIAL_TITLE"),
    SeedEntry("令史", "OFFICIAL_TITLE"),
    SeedEntry("大夫", "OFFICIAL_TITLE"),
    SeedEntry("太子", "OFFICIAL_TITLE"),
    SeedEntry("公主", "OFFICIAL_TITLE"),
    SeedEntry("皇帝", "OFFICIAL_TITLE"),
    SeedEntry("皇后", "OFFICIAL_TITLE"),
    SeedEntry("侯國", "OFFICIAL_TITLE"),
    SeedEntry("縣令", "OFFICIAL_TITLE"),
)


def all_default_seeds() -> tuple[SeedEntry, ...]:
    return DEFAULT_LOCATION_SEEDS + DEFAULT_POLITY_SEEDS + DEFAULT_TITLE_SEEDS


def write_seed_cache(
    entries: Iterable[SeedEntry],
    cache_path: Path,
    *,
    source_url: str = "internal://seed-default",
    version: str = "0.1.0",
    license: str = "internal-cc0",
) -> KBManifest:
    """Ghi seed entries ra JSONL.gz, kèm manifest.

    Ghi header gzip mtime=0 để reproducible. Sort entries theo (label, term)
    trước khi ghi.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(entries, key=lambda e: (e.label, e.term))

    # Tạo file tạm để tính SHA-256 trước khi ghi manifest.
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with (
        gzip.GzipFile(filename=str(tmp), mode="wb", mtime=0) as raw,
        __import__("io").TextIOWrapper(raw, encoding="utf-8") as f,
    ):
        for e in sorted_entries:
            f.write(
                json.dumps(
                    {"term": e.term, "label": e.label, "alias": e.alias or e.term},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    sha, size = sha256_of_file(tmp)
    row_counts: dict[str, int] = {}
    for e in sorted_entries:
        row_counts[e.label] = row_counts.get(e.label, 0) + 1

    manifest = KBManifest(
        name="seed",
        version=version,
        source_url=source_url,
        license=license,
        file_sha256=sha,
        file_size=size,
        row_counts=row_counts,
    )

    tmp.replace(cache_path)  # atomic
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    manifest.write(manifest_path)
    return manifest


def load_seed_cache(cache_path: Path) -> tuple[list[SeedEntry], KBManifest]:
    """Đọc seed cache và verify SHA-256 với manifest cùng tên."""
    from hcmus_nlp.kb.manifest import load_manifest

    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)

    sha, size = sha256_of_file(cache_path)
    if sha != manifest.file_sha256 or size != manifest.file_size:
        raise ValueError(
            f"Seed cache SHA-256 mismatch: expected {manifest.file_sha256}, got {sha}. "
            f"Hoặc re-ingest hoặc xóa cache cũ."
        )

    entries: list[SeedEntry] = []
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            entries.append(SeedEntry(term=d["term"], label=d["label"], alias=d.get("alias")))
    return entries, manifest


class SeedSource:
    """Source adapter cho seed gazetteer."""

    name = "seed"
    kind = SourceKind.SEED

    def __init__(self, entries: list[SeedEntry]):
        self._entries = entries
        self._trie = Trie()
        for e in entries:
            self._trie.insert(e.term, e.label, alias=e.alias or e.term)

    def candidates(self, text: str, ctx: AnnotationContext) -> Iterable[Candidate]:
        # Context gate 1-char dynasty names: chỉ emit nếu có cue ngữ cảnh
        # (lookbehind 1 + lookahead 4). Áp dụng cho entry đơn ký tự.
        matches = self._trie.find_all(text)
        for start, end, term, label, alias in matches:
            if len(term) == 1 and label in {"DYNASTY", "POLITY"}:
                if not self._has_dynasty_context(text, start, end):
                    continue
            yield Candidate(
                text=term,
                label=label,
                start=start,
                end=end,
                source=self.name,
                source_id=f"seed:{term}",
                priority_score=SEED_GAZETTEER,
                matched_alias=alias,
            )

    @staticmethod
    def _has_dynasty_context(text: str, start: int, end: int) -> bool:
        """1-char dynasty chỉ emit nếu có cue rõ ràng.

        Cues chấp nhận:
        - lookbehind ∈ {入 伐 據 都 建 興 亡}: động từ quân sự/chính trị.
        - lookahead ∈ {朝 亡 初 末 中 代 國 時}: từ nối tiếp dynasty.
        - lookbehind ∈ {。 ！ ？ \n}: đầu câu mới.

        Nếu start == 0, vẫn chấp nhận nếu lookahead có cue ("唐初").
        Nếu start > 0 nhưng lookbehind là chữ Hán thường và lookahead
        không có cue → từ chối.
        """
        lookahead = text[end : min(len(text), end + 4)]
        cues_after = ("朝", "亡", "初", "末", "中", "代", "國", "時")
        if any(lookahead.startswith(c) for c in cues_after):
            return True

        if start == 0:
            # Không có lookbehind để xét, đã thử lookahead → từ chối.
            return False

        lookbehind = text[start - 1]
        cues_before = {"入", "伐", "據", "都", "建", "興", "亡"}
        if lookbehind in cues_before:
            return True
        if lookbehind in {"。", "！", "？", "\n"}:
            return True
        return False

    def available(self) -> bool:
        return len(self._entries) > 0


__all__ = [
    "DEFAULT_LOCATION_SEEDS",
    "DEFAULT_POLITY_SEEDS",
    "DEFAULT_TITLE_SEEDS",
    "SeedEntry",
    "SeedSource",
    "all_default_seeds",
    "load_seed_cache",
    "write_seed_cache",
]
