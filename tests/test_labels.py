"""Test mapping config — load TOML + confirmed gate.

Regression bảo vệ:
- File không tồn tại → FileNotFoundError có thông điểm.
- Internal label không trong INTERNAL_LABELS → MappingError.
- Submission label không trong SUBMISSION_LABELS → MappingError.
- Internal label thiếu → MappingError.
- `policy.unresolved_conflict` sai → MappingError.
- Mapping.confirmed=false → is_confirmed()=False; final mode sẽ fail.
- Mapping.confirmed=true (fixture) → is_confirmed()=True.
- Cache: thay đổi file → reload.
- to_submission trả None cho label không có target.
- labels_compatible cho group DYNASTY/POLITY và BOOK/OFFICIAL_TITLE.
- preferred_label chọn theo priority thấp hơn.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from hcmus_nlp.labels import (
    INTERNAL_LABELS,
    SUBMISSION_LABELS,
    Mapping,
    MappingError,
    load_mapping,
    reset_cache,
)


@pytest.fixture
def real_mapping_path(repo_root: Path) -> Path:
    return repo_root / "config" / "mapping.toml"


@pytest.fixture
def tmp_mapping(tmp_path: Path):
    """Factory: ghi file mapping tạm với content cho trước, trả path."""

    def _make(content: str, confirmed: bool = False) -> Path:
        path = tmp_path / "mapping.toml"
        full = textwrap.dedent(content).format(confirmed=str(confirmed).lower())
        path.write_text(full, encoding="utf-8")
        reset_cache()
        return path

    return _make


class TestRealMappingFile:
    """Test file mapping thật trong repo."""

    def test_load_real_file(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.version == "0.2.0"
        # Default config giữ false cho tới khi giảng viên duyệt.
        assert m.confirmed is False

    def test_all_internal_labels_have_target(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        for internal in INTERNAL_LABELS:
            assert internal in m.internal_to_submission, f"{internal} missing in real mapping.toml"

    def test_targets_in_submission_set(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        for internal, target in m.internal_to_submission.items():
            assert target in SUBMISSION_LABELS, f"{internal} -> {target} not in SUBMISSION_LABELS"

    def test_unconfirmed_blocks_final(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.is_confirmed() is False


class TestMappingErrorCases:
    """Validate fail-fast."""

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Mapping file not found"):
            load_mapping(tmp_path / "nonexistent.toml")

    def test_internal_label_unknown(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = {confirmed}

            [mapping]
            PERSON = "PER"
            UNKNOWN_LABEL = "PER"
            """
        )
        with pytest.raises(MappingError, match="UNKNOWN_LABEL"):
            load_mapping(path)

    def test_submission_label_unknown(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = {confirmed}

            [mapping]
            PERSON = "XXX"
            """
        )
        with pytest.raises(MappingError, match="'XXX'"):
            load_mapping(path)

    def test_missing_internal_label(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = {confirmed}

            [mapping]
            PERSON = "PER"
            """
        )
        with pytest.raises(MappingError, match="Missing internal labels"):
            load_mapping(path)

    def test_empty_mapping(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = {confirmed}

            [mapping]
            """
        )
        with pytest.raises(MappingError, match="empty"):
            load_mapping(path)

    def test_policy_invalid_value(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = {confirmed}

            [mapping]
            PERSON = "PER"
            LOCATION = "LOC"
            POLITY = "ORG"
            DYNASTY = "ORG"
            OFFICIAL_TITLE = "TITLE"
            BOOK = "TITLE"
            TIME = "TME"
            NUMBER = "NUM"

            [policy]
            unresolved_conflict = "bogus"
            """
        )
        with pytest.raises(MappingError, match="unresolved_conflict"):
            load_mapping(path)

    def test_version_must_be_string(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = 0.1
            confirmed = {confirmed}

            [mapping]
            PERSON = "PER"
            """
        )
        with pytest.raises(MappingError, match="version"):
            load_mapping(path)

    def test_confirmed_must_be_bool(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = "yes"

            [mapping]
            PERSON = "PER"
            """
        )
        with pytest.raises(MappingError, match="confirmed"):
            load_mapping(path)


class TestConfirmedGate:
    """`confirmed = true` (fixture) → is_confirmed() True."""

    def test_fixture_confirmed_true(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = true

            [mapping]
            PERSON = "PER"
            LOCATION = "LOC"
            POLITY = "ORG"
            DYNASTY = "ORG"
            OFFICIAL_TITLE = "TITLE"
            BOOK = "TITLE"
            TIME = "TME"
            NUMBER = "NUM"
            """,
            confirmed=True,
        )
        m = load_mapping(path)
        assert m.is_confirmed() is True


class TestToSubmission:
    """to_submission trả target hoặc None."""

    def test_known_label(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.to_submission("PERSON") == "PER"
        assert m.to_submission("LOCATION") == "LOC"
        assert m.to_submission("TIME") == "TME"
        assert m.to_submission("NUMBER") == "NUM"

    def test_unknown_label_returns_none(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.to_submission("BOGUS") is None


class TestCompatibleGroups:
    """labels_compatible và preferred_label theo priority."""

    def test_dynasty_polity_compatible(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.labels_compatible("DYNASTY", "POLITY") is True
        assert m.labels_compatible("POLITY", "DYNASTY") is True

    def test_book_official_title_compatible(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.labels_compatible("BOOK", "OFFICIAL_TITLE") is True

    def test_unrelated_labels_not_compatible(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        assert m.labels_compatible("PERSON", "LOCATION") is False
        assert m.labels_compatible("TIME", "NUMBER") is False

    def test_preferred_label_lower_priority(self, real_mapping_path: Path):
        m = load_mapping(real_mapping_path)
        # DYNASTY priority 0 < POLITY priority 1 → DYNASTY wins.
        assert m.preferred_label("DYNASTY", "POLITY") == "DYNASTY"
        assert m.preferred_label("POLITY", "DYNASTY") == "DYNASTY"


class TestCache:
    """Cache theo mtime."""

    def test_reload_after_change(self, tmp_mapping):
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = true

            [mapping]
            PERSON = "PER"
            LOCATION = "LOC"
            POLITY = "ORG"
            DYNASTY = "ORG"
            OFFICIAL_TITLE = "TITLE"
            BOOK = "TITLE"
            TIME = "TME"
            NUMBER = "NUM"
            """,
            confirmed=True,
        )
        m1 = load_mapping(path)
        assert m1.confirmed is True

        # Sửa file → confirmed=false → cache phải reload.
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("confirmed = true", "confirmed = false"), encoding="utf-8")
        # Đảm bảo mtime thay đổi (touch file).
        import os

        os.utime(path, None)
        m2 = load_mapping(path)
        assert m2.confirmed is False


class TestAcceptanceGateTwo:
    """Acceptance H2: real confirmed=false làm final fail, fixture confirmed=true
    cho phép final pilot fixture pass. Hai test này document contract."""

    def test_real_mapping_blocks_final(self, real_mapping_path: Path):
        """Real config confirmed=false → is_confirmed False → final mode fatal."""
        m = load_mapping(real_mapping_path)
        assert m.is_confirmed() is False

    def test_fixture_mapping_allows_final(self, tmp_mapping):
        """Fixture confirmed=true → is_confirmed True → final mode OK."""
        path = tmp_mapping(
            """
            version = "0.1.0"
            confirmed = true

            [mapping]
            PERSON = "PER"
            LOCATION = "LOC"
            POLITY = "ORG"
            DYNASTY = "ORG"
            OFFICIAL_TITLE = "TITLE"
            BOOK = "TITLE"
            TIME = "TME"
            NUMBER = "NUM"
            """,
            confirmed=True,
        )
        m = load_mapping(path)
        assert m.is_confirmed() is True
