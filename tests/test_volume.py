"""Test volume heading parser — 10 edge case từ plan v5.

Regression bảo vệ:
- 10 case heading (`卷上`/`卷下`/`卷099下`/`卷015`/`卷15`/`卷十六`/
  `卷015b`/`卷019b`/collision/injective).
- Heading không phải volume (`高帝紀`) → None.
- injectivity: hai canonical key khác nhau → hai path khác nhau.
- Injection formatter lỗi → raise.
"""

from __future__ import annotations

import pytest

from hcmus_nlp.volume import (
    VolumeId,
    assert_output_paths_injective,
    canonical_key_to_dict,
    detect_collisions,
    format_volume_output_id,
    from_canonical_dict,
    parse_volume_heading,
)


class TestParseVolumeHeading:
    """10 case parse từ plan v5 + một số edge case phụ."""

    def test_arabic_with_part_lower(self):
        assert parse_volume_heading("卷099下") == VolumeId(99, "c", "卷099下")

    def test_chinese_only_upper(self):
        assert parse_volume_heading("卷上") == VolumeId(0, "a", "卷上")

    def test_chinese_only_lower(self):
        assert parse_volume_heading("卷下") == VolumeId(0, "c", "卷下")

    def test_chinese_only_middle(self):
        assert parse_volume_heading("卷中") == VolumeId(0, "b", "卷中")

    def test_arabic_15_and_015_same_canonical(self):
        v1 = parse_volume_heading("卷015")
        v2 = parse_volume_heading("卷15")
        assert v1 is not None and v2 is not None
        assert v1.canonical_key == v2.canonical_key == (15, "")
        assert v1.canonical_id() == v2.canonical_id() == "015"

    def test_arabic_with_suffix_b(self):
        assert parse_volume_heading("卷015b") == VolumeId(15, "b", "卷015b")

    def test_arabic_019b(self):
        assert parse_volume_heading("卷019b") == VolumeId(19, "b", "卷019b")

    def test_chinese_numerals_sixteen(self):
        assert parse_volume_heading("卷十六") == VolumeId(16, "", "卷十六")

    def test_chinese_numerals_twenty_one(self):
        # 二十一 = 21
        assert parse_volume_heading("卷二十一") == VolumeId(21, "", "卷二十一")

    def test_chinese_numerals_two(self):
        # 二 = 2
        assert parse_volume_heading("卷二") == VolumeId(2, "", "卷二")

    def test_chinese_numerals_one_hundred(self):
        # 一百 = 100
        assert parse_volume_heading("卷一百") == VolumeId(100, "", "卷一百")

    def test_section_heading_returns_none(self):
        assert parse_volume_heading("高帝紀") is None

    def test_none_returns_none(self):
        assert parse_volume_heading(None) is None

    def test_empty_returns_none(self):
        assert parse_volume_heading("") is None

    def test_non_volume_returns_none(self):
        # Heading kiểu khác
        assert parse_volume_heading("卷次") is None
        assert parse_volume_heading("卷之") is None

    def test_whitespace_tolerated(self):
        # Có space thừa giữa 卷 và số
        assert parse_volume_heading("卷 15") == VolumeId(15, "", "卷 15")


class TestCanonicalKeyAndId:
    """Canonical key là tuple, canonical_id là string."""

    def test_canonical_key_is_tuple(self):
        v = parse_volume_heading("卷099下")
        assert v is not None
        assert v.canonical_key == (99, "c")

    def test_canonical_id_zero_padding(self):
        # Zero-pad 3 chữ số tối thiểu
        assert VolumeId(5, "", "卷五").canonical_id() == "005"
        assert VolumeId(99, "c", "卷99下").canonical_id() == "099c"
        assert VolumeId(0, "a", "卷上").canonical_id() == "000a"
        assert VolumeId(1234, "", "卷1234").canonical_id() == "1234"

    def test_canonical_key_not_string(self):
        v = parse_volume_heading("卷15")
        assert v is not None
        # canonical_key là tuple, không phải string
        assert isinstance(v.canonical_key, tuple)
        assert not isinstance(v.canonical_key, str)


class TestInjectivityGuard:
    """Formatter injective → assert_output_paths_injective không raise."""

    def test_distinct_keys_produce_distinct_paths(self):
        keys = [
            VolumeId(1, "", "卷一"),
            VolumeId(2, "", "卷二"),
            VolumeId(15, "", "卷15"),
            VolumeId(15, "b", "卷015b"),
            VolumeId(99, "c", "卷099下"),
        ]
        # Không raise
        assert_output_paths_injective(keys)

    def test_equivalent_raw_produce_same_path_no_collision(self):
        # 卷15 và 卷015 cùng key → cùng path → OK
        keys = [
            VolumeId(15, "", "卷15"),
            VolumeId(15, "", "卷015"),
        ]
        assert_output_paths_injective(keys)

    def test_broken_formatter_raises(self):
        """Formatter giả lập (chỉ lấy part) sẽ gây collision."""

        def broken(v: VolumeId) -> str:
            return v.part  # Bỏ number → nhiều VolumeId cùng path

        keys = [VolumeId(1, "", "卷一"), VolumeId(2, "", "卷二")]
        seen: dict[str, tuple[int, str]] = {}
        for vid in keys:
            path_id = broken(vid)
            key = vid.canonical_key
            if path_id in seen and seen[path_id] != key:
                with pytest.raises(ValueError, match="Output path collision"):
                    raise ValueError(
                        f"Output path collision: {path_id} produced by both "
                        f"{seen[path_id]!r} and {key!r}. Formatter is not injective."
                    )
            seen[path_id] = key
        # Nếu code trên không raise, test này vẫn pass — phải kiểm tra raise
        # thủ công ở trên hoặc qua assert_output_paths_injective.

        # Dùng formatter thật nhưng giả lập collision bằng VolumeId trùng key:
        bad_keys = [VolumeId(1, "", "x"), VolumeId(2, "", "y")]
        # Patch formatter bằng monkeypatch không khả thi vì hàm free.
        # Test gián tiếp: dùng hai key khác nhau nhưng ép formatter raise.
        from hcmus_nlp import volume as vol_mod

        original = vol_mod.format_volume_output_id
        try:
            vol_mod.format_volume_output_id = lambda v: "same"  # type: ignore[assignment]
            with pytest.raises(ValueError, match="collision"):
                assert_output_paths_injective(bad_keys)
        finally:
            vol_mod.format_volume_output_id = original  # type: ignore[assignment]


class TestCanonicalDictRoundtrip:
    """canonical_key_to_dict / from_canonical_dict khớp."""

    def test_roundtrip(self):
        v = parse_volume_heading("卷099下")
        assert v is not None
        d = canonical_key_to_dict(v)
        assert d["volume_number"] == 99
        assert d["volume_part"] == "c"
        assert d["volume_id"] == "099c"
        assert d["volume_raw"] == "卷099下"

        # Roundtrip
        v2 = from_canonical_dict(d)
        assert v2 == v

    def test_json_record_has_string_volume_id(self):
        """volume_id lưu JSON phải là string, không phải list."""
        import json

        v = parse_volume_heading("卷015b")
        assert v is not None
        d = canonical_key_to_dict(v)
        serialized = json.dumps(d)
        # volume_id là string "015b", không phải list.
        assert '"015b"' in serialized
        assert "[15," not in serialized

    def test_from_canonical_dict_invalid(self):
        assert from_canonical_dict({"volume_number": "abc"}) is None
        assert from_canonical_dict({"volume_number": 1, "volume_part": "z"}) is None
        assert from_canonical_dict({}) is None


class TestDetectCollisions:
    """detect_collisions: heading lặp không liên tiếp trong cùng source_file.

    Phân biệt:
    - Cùng key lặp trong CÙNG file không liên tiếp → repeated_volume_heading.
    - Cùng key ở NHIỀU file khác nhau → KHÔNG repeated (mỗi file là 1 work).
    """

    def test_no_collisions(self):
        events = [
            ("file1.md", (1, "")),
            ("file1.md", (2, "")),
        ]
        assert detect_collisions(events) == []

    def test_consecutive_same_key_collapsed(self):
        # Cùng key xuất hiện liên tiếp → collapse → 0 collision.
        events = [
            ("file1.md", (1, "")),
            ("file1.md", (1, "")),
        ]
        assert detect_collisions(events) == []

    def test_non_consecutive_same_key_in_same_file(self):
        events = [
            ("file1.md", (1, "")),
            ("file1.md", (2, "")),
            ("file1.md", (1, "")),  # quay lại quyển 1
        ]
        result = detect_collisions(events)
        assert len(result) == 1
        assert result[0][0] == (1, "")
        assert result[0][1] == ["file1.md"]

    def test_same_key_across_different_files_not_repeated(self):
        # Cùng key ở 2 file khác nhau → KHÔNG repeated.
        events = [
            ("hanshu.txt", (1, "")),
            ("houhanshu.txt", (1, "")),
        ]
        assert detect_collisions(events) == []

    def test_real_corpus_pattern(self):
        """卷一 xuất hiện trong nhiều file, mỗi file 1 lần → 0 collision."""
        events = [
            ("hanshu.txt", (1, "")),
            ("hanshu.txt", (2, "")),
            ("houhanshu.txt", (1, "")),
            ("jiutangshu.txt", (1, "")),
        ]
        assert detect_collisions(events) == []

    def test_real_corpus_repeated(self):
        """Trong 1 file, quyển 1 xuất hiện 2 lần không liên tiếp → 1 collision."""
        events = [
            ("hanshu.txt", (1, "")),
            ("hanshu.txt", (2, "")),
            ("hanshu.txt", (3, "")),
            ("hanshu.txt", (1, "")),  # quay lại quyển 1 (vd lời tựa cuối)
        ]
        result = detect_collisions(events)
        assert len(result) == 1
        assert result[0][0] == (1, "")


class TestFormatVolumeOutputId:
    """format_volume_output_id dùng canonical_id."""

    def test_basic(self):
        v = parse_volume_heading("卷015b")
        assert v is not None
        assert format_volume_output_id(v) == "015b"

    def test_upper_only(self):
        v = parse_volume_heading("卷上")
        assert v is not None
        assert format_volume_output_id(v) == "000a"
