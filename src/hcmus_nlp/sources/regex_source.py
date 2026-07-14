"""Regex source adapter — pre-annotation heuristic.

Adapter Phase C5. Hành vi phải khớp byte-for-byte với snapshot baseline
`tests/fixtures/regex_baseline/sample_paragraph.ner.json` (trừ field mới).

Patterns (giữ nguyên từ annotate_corpus.py cũ):
- BOOK: `《...》`
- TIME: tên niên hiệu + `元年`/năm
- OFFICIAL_TITLE: các chức quan phổ biến
- POLITY: triều đại + `朝`/`國`/`氏`
- LOCATION: chuỗi Hán 1-6 ký tự + suffix hành chính (`郡 縣 州 邑 城 關 鄉 鎮`)

Priority order (mức ưu tiên khi overlap): BOOK > TIME > OFFICIAL_TITLE >
POLITY > LOCATION. Source này dùng cách chọn non-overlap "first match wins"
giống code cũ.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from hcmus_nlp._weights import priority_for_source
from hcmus_nlp.source_base import (
    AnnotationContext,
    Candidate,
    SourceKind,
)

ERA_NAMES = (
    "建武|永平|建安|元嘉|太和|孝昌|武德|貞觀|開元|天寶|乾隆|寶慶|更始|"
    "地皇|天鳳|永興|正始|太康|咸和|大業|顯慶|神龍|天成|同光|廣順|乾祐|顯德"
)

TIME_RE = re.compile(rf"(?:{ERA_NAMES})?(?:元年|[一二三四五六七八九十百千万〇零０-９0-9]{{1,4}}年)")
BOOK_RE = re.compile(r"《[^》\n]{1,40}》")
OFFICIAL_RE = re.compile(
    r"(?:太守|刺史|將軍|大將軍|司馬|尚書|侍郎|丞相|御史|博士|校尉|中郎將|令史|大夫|太子|公主|皇帝|皇后|侯國|縣令)"
)
LOCATION_RE = re.compile(r"[\u3400-\u9fff]{1,6}(?:郡|縣|州|邑|城|關|鄉|鎮)")
POLITY_RE = re.compile(r"(?:漢|魏|吳|蜀|秦|楚|齊|梁|陳|周|晉|隋|唐|宋|遼|金|元|明|清)(?:朝|國|氏)")


# Order matters: pattern có priority cao hơn được thêm vào occupied list trước.
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("BOOK", BOOK_RE),
    ("TIME", TIME_RE),
    ("OFFICIAL_TITLE", OFFICIAL_RE),
    ("POLITY", POLITY_RE),
    ("LOCATION", LOCATION_RE),
)


class RegexSource:
    """Source adapter cho regex heuristic (Phase C5)."""

    name = "regex"
    kind = SourceKind.REGEX

    def candidates(self, text: str, ctx: AnnotationContext) -> Iterable[Candidate]:
        occupied: list[tuple[int, int]] = []

        def overlaps(start: int, end: int) -> bool:
            return any(
                start < other_end and end > other_start for other_start, other_end in occupied
            )

        for label, pattern in PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                if overlaps(start, end):
                    continue
                occupied.append((start, end))
                yield Candidate(
                    text=match.group(0),
                    label=label,
                    start=start,
                    end=end,
                    source=self.name,
                    source_id=None,
                    priority_score=priority_for_source(self.name),
                    matched_alias=match.group(0),
                )

    def available(self) -> bool:
        return True
