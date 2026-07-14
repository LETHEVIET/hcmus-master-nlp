"""Test KB manifest + seed + CBDB/CHGIS schema detection.

Regression:
- KBManifest deterministic, không timestamp.
- sha256_of_file khớp.
- Seed cache ghi/đọc roundtrip; SHA-256 mismatch → ValueError.
- SeedSource có context gate 1-char dynasty.
- CBDB schema introspect.
- CHGIS header detector nhận 3 variants.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from hcmus_nlp.kb.manifest import KBManifest, load_manifest, sha256_of_file
from hcmus_nlp.kb.seed import (
    DEFAULT_POLITY_SEEDS,
    SeedEntry,
    SeedSource,
    all_default_seeds,
    load_seed_cache,
    write_seed_cache,
)


class TestKBManifest:
    def test_deterministic_no_timestamp(self):
        m = KBManifest(
            name="test",
            version="0.1",
            source_url=None,
            license="cc0",
            file_sha256="abc",
            file_size=100,
            row_counts={"x": 1},
        )
        d1 = m.to_dict()
        d2 = m.to_dict()
        assert d1 == d2
        # Không có field built_at.
        assert "built_at" not in d1

    def test_sha256_of_file(self, tmp_path: Path):
        p = tmp_path / "data.txt"
        p.write_text("hello world", encoding="utf-8")
        sha, size = sha256_of_file(p)
        assert size == 11
        assert len(sha) == 64

    def test_load_manifest_roundtrip(self, tmp_path: Path):
        m = KBManifest(
            name="seed",
            version="0.1",
            source_url=None,
            license="cc0",
            file_sha256="a" * 64,
            file_size=100,
            row_counts={"PERSON": 5},
        )
        p = tmp_path / "m.json"
        m.write(p)
        loaded = load_manifest(p)
        assert loaded == m


class TestSeedCache:
    def test_write_and_load(self, tmp_path: Path):
        cache = tmp_path / "seed.jsonl.gz"
        entries = (
            SeedEntry("高祖", "PERSON"),
            SeedEntry("沛", "LOCATION"),
            SeedEntry("漢", "DYNASTY"),
        )
        manifest = write_seed_cache(entries, cache, version="0.1")
        assert manifest.file_size > 0
        assert manifest.file_sha256
        assert manifest.row_counts["DYNASTY"] == 1

        loaded_entries, loaded_manifest = load_seed_cache(cache)
        assert len(loaded_entries) == 3
        assert loaded_manifest.file_sha256 == manifest.file_sha256

    def test_sha256_mismatch_raises(self, tmp_path: Path):
        cache = tmp_path / "seed.jsonl.gz"
        write_seed_cache((SeedEntry("高祖", "PERSON"),), cache, version="0.1")
        # Sửa file sau khi manifest đã ghi → SHA mismatch.
        with gzip.open(cache, "at", encoding="utf-8") as f:
            f.write('{"term":"extra","label":"PERSON","alias":"extra"}\n')
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_seed_cache(cache)

    def test_default_seeds(self):
        all_seeds = all_default_seeds()
        assert len(all_seeds) > 30
        # Có ít nhất 1 dynasty seed.
        labels = {e.label for e in all_seeds}
        assert "DYNASTY" in labels
        assert "LOCATION" in labels
        assert "OFFICIAL_TITLE" in labels


class TestSeedSource:
    def test_basic_find(self):
        source = SeedSource(list(all_default_seeds()))
        cands = list(source.candidates("沛豐邑中陽里人也", ctx=_dummy_ctx()))
        # 沛 match LOCATION
        assert any(c.text == "沛" and c.label == "LOCATION" for c in cands)

    def test_one_char_dynasty_needs_context(self):
        # 漢 đứng một mình (không có cue) → KHÔNG emit.
        source = SeedSource(list(DEFAULT_POLITY_SEEDS))
        cands = list(source.candidates("漢", ctx=_dummy_ctx()))
        assert all(c.text != "漢" or c.label != "DYNASTY" for c in cands)

    def test_one_char_dynasty_with_context(self):
        # 唐初 → 唐 match DYNASTY.
        source = SeedSource(list(DEFAULT_POLITY_SEEDS))
        cands = list(source.candidates("唐初建國", ctx=_dummy_ctx()))
        assert any(c.text == "唐" and c.label == "DYNASTY" for c in cands)

    def test_one_char_dynasty_after_sentence_end(self):
        # Sau 。 漢 → có cue "đầu câu" → emit.
        source = SeedSource(list(DEFAULT_POLITY_SEEDS))
        cands = list(source.candidates("楚滅亡。漢高祖起兵。", ctx=_dummy_ctx()))
        # 漢 sau 。 → OK; 楚 → trước 滅 (cue "亡") → OK.
        labels_found = {c.label for c in cands}
        assert "DYNASTY" in labels_found or "POLITY" in labels_found

    def test_priority_score(self):
        source = SeedSource(list(DEFAULT_POLITY_SEEDS))
        cands = list(source.candidates("唐初", ctx=_dummy_ctx()))
        assert all(c.priority_score == 0.70 for c in cands if c.label == "DYNASTY")


def _dummy_ctx():
    from hcmus_nlp.source_base import AnnotationContext

    return AnnotationContext(
        record_id="test",
        title="test",
        period=None,
        volume_id=None,
        source_file="test.txt",
        sentence_spans=(),
    )


class TestCHGISHeaderDetector:
    def test_variant_1(self):
        from hcmus_nlp.kb.chgis import detect_header

        h = detect_header(["PLACE_ID", "NAME_CN", "BEGIN", "END", "TYPE", "PARENT_ID"])
        assert h is not None
        assert h["name"] == "NAME_CN"

    def test_variant_2(self):
        from hcmus_nlp.kb.chgis import detect_header

        h = detect_header(["id", "name", "begin", "end", "type", "parent"])
        assert h is not None

    def test_variant_3(self):
        from hcmus_nlp.kb.chgis import detect_header

        h = detect_header(
            ["CHGIS_ID", "PLACE_NAME", "START_YEAR", "END_YEAR", "ADMIN_TYPE", "PARENT_CHGIS_ID"]
        )
        assert h is not None

    def test_unknown_returns_none(self):
        from hcmus_nlp.kb.chgis import detect_header

        h = detect_header(["foo", "bar"])
        assert h is None


class TestCBDBSchemaIntrospect:
    def test_introspect_real_sqlite(self, tmp_path: Path):
        """Tạo SQLite giả lập 1 schema CBDB-lite, introspect."""
        import sqlite3

        p = tmp_path / "fake_cbdb.sqlite"
        conn = sqlite3.connect(p)
        try:
            conn.execute(
                "CREATE TABLE BIOG_MAIN (c_personid INTEGER, c_name_chn TEXT, c_surname_chn TEXT)"
            )
            conn.commit()
        finally:
            conn.close()

        from hcmus_nlp.kb.cbdb import introspect_schema

        schema = introspect_schema(p)
        assert "BIOG_MAIN" in schema
        assert "c_personid" in schema["BIOG_MAIN"]
        assert "c_name_chn" in schema["BIOG_MAIN"]
