"""Rule-based sentence segmentation.

Tách câu dựa trên dấu mạnh (`。！？!?｡`) nằm ngoài cặp markup
(`〈〉`, `《》`, `（ ）`, `「」`, `『』`, `【】`, `()`).

Không tự ý tách sau `；`/`：` vì đó là ranh giới mệnh đề hoặc dấu dẫn nhập.
Offset dùng `start` inclusive, `end` exclusive theo chỉ số Unicode Python.
"""

from __future__ import annotations

from collections.abc import Iterable

from hcmus_nlp.source_base import Span

OPEN_TO_CLOSE = {
    "〈": "〉",
    "《": "》",
    "（": "）",
    "(": ")",
    "「": "」",
    "『": "』",
    "【": "】",
    "[": "]",
}
CLOSE_CHARS = set(OPEN_TO_CLOSE.values())
SENTENCE_ENDS = set("。！？!?｡")


def segment(text: str) -> list[Span]:
    """Trả list Span (start, end) của các câu.

    Hàm cũ trả list dict; phiên bản này trả Span tuple cho nhẹ và dễ test.
    Span theo quy ước start inclusive / end exclusive, đã trim leading/trailing
    whitespace của từng câu.
    """
    sentences: list[Span] = []
    stack: list[str] = []
    start = 0

    def emit(end: int) -> None:
        nonlocal start
        raw = text[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        s = start + left_trim
        e = start + right_trim
        if s < e:
            sentences.append(Span(s, e))
        start = end

    for index, char in enumerate(text):
        if char in OPEN_TO_CLOSE:
            stack.append(OPEN_TO_CLOSE[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char in CLOSE_CHARS and stack:
            if char in stack:
                stack = stack[: len(stack) - 1 - stack[::-1].index(char)]
        if char in SENTENCE_ENDS and not stack:
            emit(index + 1)

    emit(len(text))
    return sentences


def segment_with_text(text: str) -> list[dict]:
    """Trả list dict có start/end/text để tương thích annotate_corpus.py cũ."""
    spans = segment(text)
    return [{"start": s.start, "end": s.end, "text": text[s.start : s.end]} for s in spans]


def sentence_spans_to_dicts(spans: Iterable[Span]) -> list[dict]:
    return [{"start": s.start, "end": s.end} for s in spans]
