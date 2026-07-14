"""Volume heading parser + injectivity guard.

Heading dạng `卷<N><part>` trong markdown heading (H1/H2/...) được parse thành
`VolumeId(number, part, raw)`. Canonical key là tuple `(number, part)` — không
phải string `{number:03d}{part}` vì zero-padding chỉ là cách viết.

Quyết định canonical cho heading không có số (諸蕃志 chỉ có `卷上`/`卷下`):
    卷上 → (0, "a")
    卷中 → (0, "b")
    卷下 → (0, "c")

Nếu sau này phát sinh cả `卷001上` thì `(0,"a")` và `(1,"a")` vẫn hai entry —
không collision vì khác number. Edge case lưu comment ở đây.

Collision semantics:
- `卷十五`, `卷15`, `卷015` → cùng key `(15, "")`, OK.
- `卷015b` → `(15, "b")`.
- `卷019b` → `(19, "b")`.
- Hai canonical key khác nhau phải tạo hai output path khác nhau; nếu formatter
  trùng, `assert_output_paths_injective()` raise.

Heading event collapse:
- Hai heading giống canonical key xuất hiện liền nhau do raw có cả `## 卷01`
  và `# 卷01` → collapse thành một heading event (không sinh collision giả).
- Cùng canonical key tái xuất hiện không liên tiếp ghi audit warning
  `repeated_volume_heading` kèm source lines; không tự kết luận collision.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

Part = Literal["", "a", "b", "c"]


@dataclass(frozen=True)
class VolumeId:
    """Canonical identity của một quyển trong tác phẩm.

    - `number`: int (canonical; `卷015` và `卷15` đều → 15).
    - `part`: "" (không có phần) / "a" (卷上) / "b" (卷中) / "c" (卷下).
    - `raw`: chuỗi heading gốc để audit.
    """

    number: int
    part: Part
    raw: str

    @property
    def canonical_key(self) -> tuple[int, str]:
        return (self.number, self.part)

    def canonical_id(self) -> str:
        """String canonical dùng cho JSON record (vd `015b`, `099c`, `000a`).

        Zero-pad tối thiểu 3 chữ số cho readability. Đây KHÔNG phải identity —
        identity là tuple `(number, part)`. Hai raw `卷15` và `卷015` cho cùng
        canonical_id `015` và cùng canonical key `(15, "")` → OK.
        """
        num_str = f"{self.number:03d}"
        return f"{num_str}{self.part}"


# --- Numeric parsing ----------------------------------------------------------

_ARABIC_RE = re.compile(r"\d+")
# Đơn giản cho corpus lịch sử: hỗ trợ 1-99 bằng chữ Hán. Mở rộng khi cần.
_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
    "千": 1000,
    "万": 10000,
    "兩": 2,
}


def _chinese_to_int(s: str) -> int | None:
    """Convert chuỗi chữ số Hán sang int. Trả None nếu không parse được.

    Hỗ trợ pattern đơn giản: 十 = 10, 二十 = 20, 二十一 = 21, 一百 = 100,
    一百二十三 = 123. Không hỗ trợ 万/億 phức tạp (corpus lịch sử hiếm dùng
    cho số quyển).
    """
    if not s:
        return None
    if s == "十":
        return 10
    total = 0
    current_digit = 0
    for char in s:
        v = _CHINESE_DIGITS.get(char)
        if v is None:
            return None
        if v >= 10:  # 十/百/千
            if current_digit == 0:
                current_digit = 1
            total += current_digit * v
            current_digit = 0
        else:
            current_digit = v
    total += current_digit
    return total


def _parse_number(token: str) -> int | None:
    """Parse một token số (Arabic hoặc chữ Hán) sang int."""
    if not token:
        return None
    if token.isdigit():
        return int(token)
    # Strip leading zeros trước khi convert; nếu toàn chữ số thì OK.
    return _chinese_to_int(token)


# --- Heading parser -----------------------------------------------------------

# Heading match: `卷` + số (Arabic hoặc chữ Hán) + optional part suffix.
# Số có thể trống (諸蕃志 dùng `卷上`/`卷下` không có số).
_PART_MAP = {
    "上": "a",
    "中": "b",
    "下": "c",
    "a": "a",
    "b": "b",
    "c": "c",
}

# Match cả `卷十五`, `卷15`, `卷015`, `卷099下`, `卷015b`, `卷上`, `卷下`.
_HEADING_RE = re.compile(
    r"^卷\s*"
    r"([0-9０-９]+|[零〇一二三四五六七八九十百千]+)?"
    r"(上|中|下|[abc])?$"
)


def parse_volume_heading(raw: str | None) -> VolumeId | None:
    """Parse heading markdown thành `VolumeId`. Trả None nếu không phải volume.

    Ví dụ:
        parse_volume_heading("卷099下")  → VolumeId(99, "c", "卷099下")
        parse_volume_heading("卷上")    → VolumeId(0,  "a", "卷上")
        parse_volume_heading("卷15")    → VolumeId(15, "",  "卷15")
        parse_volume_heading("卷015")   → VolumeId(15, "",  "卷015")
        parse_volume_heading("卷十六")  → VolumeId(16, "",  "卷十六")
        parse_volume_heading("卷015b")  → VolumeId(15, "b", "卷015b")
        parse_volume_heading("高帝紀")  → None
        parse_volume_heading(None)      → None
    """
    if raw is None:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    m = _HEADING_RE.match(cleaned)
    if not m:
        return None
    num_token, part_char = m.group(1), m.group(2)

    if num_token is None and part_char is None:
        return None

    if num_token is None:
        # Heading không có số (chỉ có `卷上`/`卷中`/`卷下`).
        number = 0
    else:
        number = _parse_number(num_token)
        if number is None:
            return None

    part: Part = ""
    if part_char is not None:
        mapped = _PART_MAP.get(part_char)
        if mapped is None:
            return None
        part = mapped  # type: ignore[assignment]

    return VolumeId(number=number, part=part, raw=cleaned)


# --- Injectivity guard --------------------------------------------------------


def format_volume_output_id(volume_id: VolumeId) -> str:
    """Format canonical path id cho folder/file. Single source of truth.

    Exporter phải dùng hàm này để tạo tên folder/file. Hai canonical key khác
    nhau phải cho hai output path khác nhau.
    """
    return volume_id.canonical_id()


def assert_output_paths_injective(
    volume_ids: Iterable[VolumeId],
) -> None:
    """Raise ValueError nếu hai canonical key map cùng output path.

    Gọi ở cuối prepare_corpus.py sau khi build xong toàn bộ volume_id set,
    trước khi exporter thật sự ghi file. Đây là guard bắt buộc cho tính đúng
    của submission folder layout.
    """
    seen: dict[str, tuple[int, str]] = {}
    for vid in volume_ids:
        path_id = format_volume_output_id(vid)
        key = vid.canonical_key
        if path_id in seen and seen[path_id] != key:
            raise ValueError(
                f"Output path collision: {path_id} produced by both "
                f"{seen[path_id]!r} and {key!r}. Formatter is not injective."
            )
        seen[path_id] = key


def detect_collisions(
    volume_events: Iterable[tuple[str, tuple[int, str]]],
) -> list[tuple[tuple[int, str], list[str]]]:
    """Phát hiện canonical key xuất hiện >1 lần KHÔNG liên tiếp **trong cùng
    source_file**.

    `volume_events` là iterable các `(source_file, canonical_key)`. Phân
    nhóm theo source_file, sau đó với mỗi file kiểm tra sequence:
    - Hai event liên tiếp cùng key → collapse (không tính).
    - Hai event không liên tiếp cùng key → flag repeated_volume_heading.

    Cùng canonical key xuất hiện ở 2 FILE khác nhau KHÔNG phải repeated
    heading — đó là 2 tác phẩm có cùng số quyển, hoàn toàn hợp lệ. Collision
    cứng (2 key khác nhau cùng path) đã được `assert_output_paths_injective`
    bắt.
    """
    per_file: dict[str, list[tuple[int, str]]] = {}
    for source, key in volume_events:
        per_file.setdefault(source, []).append(key)

    out: list[tuple[tuple[int, str], list[str]]] = []
    for source, keys in per_file.items():
        collapsed: list[tuple[int, str]] = []
        for key in keys:
            if not collapsed or collapsed[-1] != key:
                collapsed.append(key)
        counts: dict[tuple[int, str], int] = {}
        for key in collapsed:
            counts[key] = counts.get(key, 0) + 1
        for key, n in counts.items():
            if n > 1:
                out.append((key, [source]))
    return out


def canonical_key_to_dict(vid: VolumeId) -> dict:
    """Serialize VolumeId thành dict tương thích JSON record.

    Lưu ý: tuple canonical key không lưu vào JSON (sẽ bị đổi thành list).
    Caller lưu `volume_number`, `volume_part`, `volume_id` (string canonical)
    riêng; tuple chỉ dùng nội bộ khi parse.
    """
    return {
        "volume_number": vid.number,
        "volume_part": vid.part,
        "volume_id": vid.canonical_id(),
        "volume_raw": vid.raw,
    }


def from_canonical_dict(d: Mapping[str, object]) -> VolumeId | None:
    """Reconstruct VolumeId từ dict (ngược lại canonical_key_to_dict)."""
    number = d.get("volume_number")
    part = d.get("volume_part", "")
    raw = d.get("volume_raw", "")
    if not isinstance(number, int):
        return None
    if part not in ("", "a", "b", "c"):
        return None
    return VolumeId(number=number, part=part, raw=str(raw))
