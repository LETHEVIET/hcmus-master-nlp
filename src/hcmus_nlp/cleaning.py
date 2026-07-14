r"""Cleaning audit cho corpus Hán cổ lịch sử.

Phân loại từng dòng raw thành một trong ba `Decision`:
- KEEP: giữ làm content
- DROP: loại bỏ (boilerplate, blank, work title lặp)
- NEEDS_REVIEW: cần người xem (inline OCR placeholder, URL leak, ellipsis dài)

Quyết định:
- Boilerplate cũ (`姊妹计划`, `Public domain`, `本作品在全世界都属于`,
  `本作品 原文没有標點`) → DROP/BOILERPLATE.
- Heading markdown → đã được `prepare_corpus` xử lý riêng; cleaning chỉ áp
  dụng cho phần text.
- Inline OCR placeholder giữa Hán (`愚F8D5小生`) → NEEDS_REVIEW.
- URL `https?://\S+` và `Referenced:` → NEEDS_REVIEW/URL_LEAK.
- Wiki footer `{{footer|...}}` → NEEDS_REVIEW/WIKI_FOOTER.
- Ellipsis `…` KHÔNG tự động = OCR. Chỉ khi 5+ ký tự liên tiếp và line không
  có chữ Hán khác mới tính NEEDS_REVIEW.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field


class Decision(enum.Enum):
    KEEP = "keep"
    DROP = "drop"
    NEEDS_REVIEW = "needs_review"


class Reason(enum.Enum):
    BOILERPLATE = "boilerplate"
    WORK_TITLE = "work_title"
    BLANK = "blank"
    WIKISOURCE_TAG = "wikisource_tag"
    URL_LEAK = "url_leak"
    LICENSE_LEAK = "license_leak"
    INLINE_OCR_PLACEHOLDER = "inline_ocr_placeholder"
    WIKI_FOOTER = "wiki_footer"
    ELLIPSIS_RUN = "ellipsis_run"
    OTHER = "other"


@dataclass(frozen=True)
class AuditResult:
    decision: Decision
    reasons: tuple[Reason, ...] = field(default_factory=tuple)

    @property
    def is_drop(self) -> bool:
        return self.decision is Decision.DROP

    @property
    def is_needs_review(self) -> bool:
        return self.decision is Decision.NEEDS_REVIEW


# --- Patterns --------------------------------------------------------------

# Boilerplate cũ (giữ nguyên hành vi cũ).
_BOILERPLATE_PREFIXES = (
    "姊妹计划",
    "Public domain",
    "本作品在全世界都属于",
    "本作品 原文没有標點",
)

# Hex marker dạng `[CF3D]` (2-8 hex digits trong ngoặc vuông).
_INLINE_HEX_BRACKET_RE = re.compile(r"\[[0-9A-F]{2,8}\]")

# Hex marker giữa hai ký tự Hán: `愚F8D5小生`, `至CF3D楡`.
# Tương tự uppercase: cho phép một phía là Han, phía kia là đầu/cuối string.
_INLINE_HEX_BETWEEN_HAN_RE = re.compile(
    r"(?:(?<=[\u3400-\u9fff])|^)[A-F][0-9A-F]{2,7}(?:(?=[\u3400-\u9fff])|$)"
)

# Marker in hoa 2-8 chars ở một phía là Hán: `AT逆`, `弟MH`, `姓MS氏`.
# Dùng alternation: hoặc lookbehind Han, hoặc đầu string; lookahead Han,
# hoặc cuối string. Không yêu cầu cả hai phía vì OCR marker có thể dính sát
# đầu/cuối dòng.
_INLINE_UPPER_BETWEEN_HAN_RE = re.compile(
    r"(?:(?<=[\u3400-\u9fff])|^)[A-Z]{2,8}(?:(?=[\u3400-\u9fff])|$)"
)

_URL_RE = re.compile(r"https?://\S+")
_REFERENCED_RE = re.compile(r"\bReferenced:\s*", re.IGNORECASE)
_WIKI_FOOTER_RE = re.compile(r"\{\{footer\b", re.IGNORECASE)
_ELLIPSIS_RUN_RE = re.compile(r"…{5,}")

# Phát hiện chữ Hán trong line (để phân biệt ellipsis dài trong line Hán vs
# line chỉ toàn dấu câu).
_HAN_CHAR_RE = re.compile(r"[\u3400-\u9fff]")


# --- Public API ------------------------------------------------------------


def is_boilerplate_line(line: str) -> bool:
    """Kiểm tra line có phải Wikisource boilerplate cũ không."""
    stripped = line.strip()
    if not stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in _BOILERPLATE_PREFIXES)


def audit(line: str, *, work_title: str | None = None) -> AuditResult:
    """Audit một dòng raw. Trả `AuditResult(decision, reasons)`.

    Tham số `work_title` để phát hiện work title lặp ở đầu file. Nếu không
    truyền, mặc định coi line đầu tiên không phải work title lặp (caller
    truyền đúng ngữ cảnh).
    """
    stripped = line.strip()

    # 1. Boilerplate / blank cũ → DROP.
    if not stripped:
        return AuditResult(Decision.DROP, (Reason.BLANK,))
    for prefix in _BOILERPLATE_PREFIXES:
        if stripped.startswith(prefix):
            return AuditResult(Decision.DROP, (Reason.BOILERPLATE,))

    # 2. Work title lặp đầu file (caller quyết định dựa trên context).
    if work_title is not None and stripped == work_title:
        return AuditResult(Decision.DROP, (Reason.WORK_TITLE,))

    reasons: list[Reason] = []

    # 3. Inline OCR placeholder.
    if (
        _INLINE_HEX_BRACKET_RE.search(stripped)
        or _INLINE_HEX_BETWEEN_HAN_RE.search(stripped)
        or _INLINE_UPPER_BETWEEN_HAN_RE.search(stripped)
    ):
        reasons.append(Reason.INLINE_OCR_PLACEHOLDER)

    # 4. URL leak.
    if _URL_RE.search(stripped) or _REFERENCED_RE.search(stripped):
        reasons.append(Reason.URL_LEAK)

    # 5. Wiki footer.
    if _WIKI_FOOTER_RE.search(stripped):
        reasons.append(Reason.WIKI_FOOTER)

    # 6. Ellipsis run dài không có chữ Hán khác.
    if _ELLIPSIS_RUN_RE.search(stripped) and not _HAN_CHAR_RE.search(stripped):
        reasons.append(Reason.ELLIPSIS_RUN)

    if reasons:
        return AuditResult(Decision.NEEDS_REVIEW, tuple(reasons))

    return AuditResult(Decision.KEEP, ())


def normalize_text(line: str) -> str:
    """Chuẩn hóa whitespace; bảo toàn chữ Hán và dấu câu."""
    return re.sub(r"[ \t\u00a0]+", " ", line.strip())


def audit_record_metadata(
    text: str,
    *,
    source_file: str,
    source_line: int,
) -> AuditResult:
    """Audit `text` (đã normalize) cho mọi artifact inline.

    Dùng sau khi `normalize_text` để kiểm tra record giữ có còn mang artifact
    không (ví dụ heading markdown nào đó lọt vào text). Trả AuditResult.
    """
    return audit(text, work_title=None)


__all__ = [
    "AuditResult",
    "Decision",
    "Reason",
    "audit",
    "audit_record_metadata",
    "is_boilerplate_line",
    "normalize_text",
]
