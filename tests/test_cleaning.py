"""Test cleaning audit — inline OCR placeholder + URL leak + ellipsis.

Regression bảo vệ:
- Fixtures từ plan v5: `愚F8D5小生`, `至[CF3D]楡`, URL `https://...`,
  `Referenced: ...`, `{{footer|...}}`, ellipsis ngắn giữ, ellipsis dài review.
- Record giữ có `cleaning_status == "needs_review"` khi có reason.
- Boilerplate cũ vẫn DROP.
- Work title lặp đầu file DROP khi truyền `work_title`.
"""

from __future__ import annotations

import pytest

from hcmus_nlp.cleaning import (
    Decision,
    Reason,
    audit,
    is_boilerplate_line,
    normalize_text,
)


class TestBoilerplate:
    """Boilerplate cũ vẫn DROP."""

    def test_boilerplate_jiemei(self):
        result = audit("姊妹计划: blah blah")
        assert result.decision is Decision.DROP
        assert Reason.BOILERPLATE in result.reasons

    def test_boilerplate_public_domain(self):
        result = audit("Public domain notice here")
        assert result.decision is Decision.DROP
        assert Reason.BOILERPLATE in result.reasons

    def test_boilerplate_chinese(self):
        result = audit("本作品在全世界都属于公有领域")
        assert result.decision is Decision.DROP
        assert Reason.BOILERPLATE in result.reasons

    def test_boilerplate_punctuation_notice(self):
        result = audit("本作品 原文没有標點")
        assert result.decision is Decision.DROP
        assert Reason.BOILERPLATE in result.reasons

    def test_blank_line_drop(self):
        result = audit("")
        assert result.decision is Decision.DROP
        assert Reason.BLANK in result.reasons

    def test_whitespace_only_drop(self):
        result = audit("   \t  ")
        assert result.decision is Decision.DROP
        assert Reason.BLANK in result.reasons

    def test_is_boilerplate_helper(self):
        assert is_boilerplate_line("姊妹计划: ...") is True
        assert is_boilerplate_line("") is True
        assert is_boilerplate_line("高祖，沛豐邑中陽里人也。") is False


class TestInlineOcrPlaceholder:
    """Marker hex inline giữa Hán → NEEDS_REVIEW."""

    def test_hex_between_han_F8D5(self):
        # 愚F8D5小生 — hex F8D5 nằm giữa 愚 và 小 (đều là chữ Hán).
        result = audit("愚F8D5小生")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons

    def test_hex_between_han_CF3D(self):
        result = audit("至CF3D楡")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons

    def test_hex_bracket_CF3D(self):
        # 至[CF3D]楡 — hex trong ngoặc vuông.
        result = audit("至[CF3D]楡")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons

    def test_uppercase_marker_AT(self):
        # AT逆 — chuỗi in hoa 2+ chars giữa Hán.
        result = audit("AT逆")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons

    def test_uppercase_marker_MS(self):
        result = audit("姓MS氏")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons

    def test_uppercase_marker_MH(self):
        result = audit("弟MH")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons


class TestUrlLeak:
    """URL inline hoặc Referenced → NEEDS_REVIEW."""

    def test_https_url(self):
        result = audit("Xem thêm tại https://zh.wikisource.org/wiki/漢書")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.URL_LEAK in result.reasons

    def test_http_url(self):
        result = audit("Link: http://example.com/foo")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.URL_LEAK in result.reasons

    def test_referenced_line(self):
        result = audit("Referenced: https://...")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.URL_LEAK in result.reasons

    def test_referenced_case_insensitive(self):
        result = audit("referenced: https://...")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.URL_LEAK in result.reasons


class TestWikiFooter:
    """Wiki footer template → NEEDS_REVIEW."""

    def test_footer_template(self):
        result = audit("{{footer|something}}")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.WIKI_FOOTER in result.reasons

    def test_footer_case_insensitive(self):
        result = audit("{{FOOTER|...}}")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.WIKI_FOOTER in result.reasons


class TestEllipsis:
    """Ellipsis ngắn giữ; ellipsis dài không có chữ Hán → REVIEW."""

    def test_short_ellipsis_kept(self):
        # 3 ký tự ellipsis, có chữ Hán khác → KEEP.
        result = audit("高祖……沛豐邑人也。")
        assert result.decision is Decision.KEEP

    def test_long_ellipsis_no_han_review(self):
        # 20 ký tự ellipsis liên tiếp, không có chữ Hán → NEEDS_REVIEW.
        result = audit("…" * 20)
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.ELLIPSIS_RUN in result.reasons

    def test_long_ellipsis_with_han_kept(self):
        # Ellipsis dài nhưng có chữ Hán khác trong line → coi như dấu câu.
        result = audit("…" * 10 + "高祖沛人也。" + "…" * 10)
        assert result.decision is Decision.KEEP


class TestNormalChinese:
    """Text Hán bình thường → KEEP."""

    def test_clean_sentence(self):
        result = audit("高祖，沛豐邑中陽里人也。")
        assert result.decision is Decision.KEEP
        assert result.reasons == ()

    def test_long_clean_paragraph(self):
        text = "昔在帝堯，聰明文思，光宅天下。以義制法，以仁成俗，刑措不用，囹圄空虛。"
        result = audit(text)
        assert result.decision is Decision.KEEP


class TestWorkTitleRepeat:
    """Work title lặp đầu file → DROP khi caller truyền work_title."""

    def test_work_title_match(self):
        result = audit("諸蕃志", work_title="諸蕃志")
        assert result.decision is Decision.DROP
        assert Reason.WORK_TITLE in result.reasons

    def test_work_title_not_match(self):
        result = audit("高祖沛人也", work_title="諸蕃志")
        assert result.decision is Decision.KEEP


class TestMultipleReasons:
    """Một line có thể có nhiều reason."""

    def test_url_plus_footer(self):
        # `{{footer|...}}` + URL cùng line → cả hai reason.
        result = audit("{{footer}} xem https://x.com")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.URL_LEAK in result.reasons
        assert Reason.WIKI_FOOTER in result.reasons

    def test_hex_plus_url(self):
        result = audit("愚F8D5小生 https://x.com")
        assert result.decision is Decision.NEEDS_REVIEW
        assert Reason.INLINE_OCR_PLACEHOLDER in result.reasons
        assert Reason.URL_LEAK in result.reasons


class TestNormalizeText:
    """normalize_text chỉ chuẩn hóa whitespace; bảo toàn chữ Hán."""

    def test_collapse_whitespace(self):
        assert normalize_text("  高祖\t  沛人也  ") == "高祖 沛人也"

    def test_preserve_punctuation(self):
        assert normalize_text("高祖，沛人也。") == "高祖，沛人也。"

    def test_strip_leading_trailing(self):
        assert normalize_text("\t  高祖沛人也。  \n") == "高祖沛人也。"
