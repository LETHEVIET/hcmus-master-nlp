"""Source adapter protocol + AnnotationContext + Candidate.

Mọi source (regex, gazetteer, KB, model) implement `SourceAdapter`. Hàm
`candidates(text, ctx)` nhận text + AnnotationContext và trả về iterable
`Candidate`. `available()` kiểm tra xem source đã sẵn sàng (vd model đã
download hay chưa).

`Candidate` là frozen dataclass để có thể dùng trong set và đảm bảo
deterministic JSON shape.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple, Protocol, runtime_checkable


class Span(NamedTuple):
    start: int
    end: int


@dataclass(frozen=True)
class AnnotationContext:
    """Context bất biến truyền cho mọi source adapter.

    - `record_id`: id của record trong corpus (vd `hanshu-000123`).
    - `title`: tên tác phẩm (`漢書`).
    - `period`: chuỗi period (`東漢`, `唐`, v.v.) — dùng làm prior, không phải
      filter cứng.
    - `volume_id`: tuple (number, part) hoặc None nếu chưa xác định.
    - `source_file`: tên file raw.
    - `sentence_spans`: tuple Span của toàn record (để source có thể biết
      sentence boundary mà không cần tự parse).
    """

    record_id: str
    title: str
    period: str | None
    volume_id: tuple[int, str] | None
    source_file: str
    sentence_spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class Candidate:
    """Candidate NER do một source sinh ra.

    Tất cả offset là global trong record text.
    """

    text: str
    label: str
    start: int
    end: int
    source: str
    source_id: str | None = None
    priority_score: float = 0.5
    matched_alias: str | None = None
    # Entity-linking (nhiều KB record cho cùng span/label) ≠ NER conflict.
    linking_candidates: tuple[str, ...] = ()
    linking_status: str | None = None


class SourceKind:
    """Tag cho source để merger có thể ưu tiên / debug."""

    REGEX = "regex"
    SEED = "seed"
    KB = "kb"
    MODEL = "model"
    HUMAN = "human"


@runtime_checkable
class SourceAdapter(Protocol):
    """Protocol cho mọi NER source.

    Implement tối thiểu:
    - `name`: tên adapter (string).
    - `kind`: một trong SourceKind.*.
    - `candidates(text, ctx)`: iterable Candidate.
    - `available()`: True nếu adapter có thể chạy (vd model đã sync).
    """

    name: str
    kind: str

    def candidates(self, text: str, ctx: AnnotationContext) -> Iterable[Candidate]: ...

    def available(self) -> bool: ...


__all__ = [
    "AnnotationContext",
    "Candidate",
    "SourceAdapter",
    "SourceKind",
    "Span",
]
