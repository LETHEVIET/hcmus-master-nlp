"""Test Doccano I/O (Phase E).

Regression — 8 case từ plan v5:
(a) Pass-through idempotent.
(b) Add entity mới → xuất hiện, status checked.
(c) Xóa entity → không còn.
(d) Relabel → label mới đúng.
(e) Duplicate text nhưng khác id → không nhầm.
(f) Sentence ID có mặt với label=[] → checked empty; ID vắng mặt →
    không tự đánh checked.
(g) Duplicate id trong Doccano → fatal.
(h) Atomic: --corpus == --output → fatal (qua argparse).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmus_nlp.doccano_io import (
    DoccanoError,
    DoccanoRecord,
    export_to_doccano,
    import_from_doccano,
    load_doccano_export,
)


@pytest.fixture
def fake_corpus_path(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": "r1",
            "title": "漢書",
            "text": "高祖沛人也。沛豐邑中陽里。",
            "sentences": [
                {
                    "sid": "r1-s1",
                    "start": 0,
                    "end": 7,
                    "text": "高祖沛人也。",
                    "review_status": "needs_review",
                },
                {
                    "sid": "r1-s2",
                    "start": 7,
                    "end": 14,
                    "text": "沛豐邑中陽里。",
                    "review_status": "needs_review",
                },
            ],
            "entities": [
                {
                    "eid": "r1-e1",
                    "sentence_id": "r1-s1",
                    "start": 0,
                    "end": 2,
                    "text": "高祖",
                    "label": "PERSON",
                    "method": "heuristic",
                    "review_status": "needs_review",
                },
            ],
        }
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


@pytest.fixture
def duplicate_text_corpus(tmp_path: Path) -> Path:
    """Hai câu cùng text '詔曰：' nhưng khác sentence_id."""
    p = tmp_path / "corpus.jsonl"
    records = [
        {
            "id": "r1",
            "title": "漢書",
            "text": "詔曰：詔曰：",
            "sentences": [
                {
                    "sid": "r1-s1",
                    "start": 0,
                    "end": 4,
                    "text": "詔曰：",
                    "review_status": "needs_review",
                },
                {
                    "sid": "r1-s2",
                    "start": 4,
                    "end": 8,
                    "text": "詔曰：",
                    "review_status": "needs_review",
                },
            ],
            "entities": [],
        }
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


@pytest.fixture
def non_zero_start_corpus(tmp_path: Path) -> Path:
    """Câu bắt đầu ở offset 150 trong record."""
    p = tmp_path / "corpus.jsonl"
    # 149 ký tự trước câu (whitespace), câu ở offset 150.
    prefix = "。" * 150
    text = prefix + "高祖沛人也。"
    records = [
        {
            "id": "r1",
            "title": "漢書",
            "text": text,
            "sentences": [
                {
                    "sid": "r1-s1",
                    "start": 150,
                    "end": 156,
                    "text": "高祖沛人也。",
                    "review_status": "needs_review",
                },
            ],
            "entities": [],
        }
    ]
    with p.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


class TestExportToDoccano:
    def test_basic_export(self, fake_corpus_path: Path, tmp_path: Path):
        out = tmp_path / "doccano.jsonl"
        with fake_corpus_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        n = export_to_doccano(records, out)
        assert n == 2  # 2 sentence

        # Verify content.
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        rec1 = json.loads(lines[0])
        assert rec1["id"] == "r1-s1"
        assert rec1["data"] == "高祖沛人也。"
        # Relative offset: 高祖 ở 0..2.
        assert rec1["label"] == [[0, 2, "PERSON"]]

    def test_export_offset_is_relative(self, fake_corpus_path: Path, tmp_path: Path):
        """Offset trong Doccano là relative tới sentence, không phải global."""
        out = tmp_path / "doccano.jsonl"
        with fake_corpus_path.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        export_to_doccano(records, out)
        rec2 = json.loads(out.read_text(encoding="utf-8").strip().split("\n")[1])
        assert rec2["id"] == "r1-s2"
        assert rec2["label"] == []  # không có entity


class TestLoadDoccanoExport:
    def test_basic_load(self, tmp_path: Path):
        p = tmp_path / "doc.jsonl"
        p.write_text(
            json.dumps(
                {"id": "s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps({"id": "s2", "data": "沛豐邑。", "label": []}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        recs = load_doccano_export(p)
        assert "s1" in recs
        assert recs["s1"].label == ((0, 2, "PERSON"),)
        assert recs["s2"].label == ()

    def test_duplicate_id_fatal(self, tmp_path: Path):
        p = tmp_path / "doc.jsonl"
        p.write_text(
            json.dumps({"id": "s1", "data": "x", "label": []}, ensure_ascii=False)
            + "\n"
            + json.dumps({"id": "s1", "data": "y", "label": []}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DoccanoError, match="duplicate id"):
            load_doccano_export(p)

    def test_invalid_span_fatal(self, tmp_path: Path):
        p = tmp_path / "doc.jsonl"
        # span (0, 10) vượt quá data length 2.
        p.write_text(
            json.dumps({"id": "s1", "data": "ab", "label": [[0, 10, "X"]]}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(DoccanoError, match="invalid span"):
            load_doccano_export(p)


class TestImportFromDoccano:
    """8 test case từ plan v5."""

    def test_a_passthrough_idempotent(self, fake_corpus_path: Path, tmp_path: Path):
        # Doccano giữ nguyên entity của r1-s1.
        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(doc, fake_corpus_path, out)
        assert stats["n_added"] == 0
        assert stats["n_removed"] == 0
        # reviewed record có r1-s1 checked.
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        sent = next(s for s in recs[0]["sentences"] if s["sid"] == "r1-s1")
        assert sent["review_status"] == "checked"

    def test_b_add_new_entity(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        # Doccano thêm 沛豐 (LOC) ở câu 2.
        doc.write_text(
            json.dumps(
                {
                    "id": "r1-s2",
                    "data": "沛豐邑中陽里。",
                    "label": [[0, 3, "LOC"]],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(doc, fake_corpus_path, out)
        assert stats["n_added"] >= 1
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        ents_s2 = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s2"]
        assert any(e["label"] == "LOC" and e["text"] == "沛豐邑" for e in ents_s2)

    def test_c_remove_entity(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        # Doccano xóa 高祖 (label=[] cho r1-s1).
        doc.write_text(
            json.dumps({"id": "r1-s1", "data": "高祖沛人也。", "label": []}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(doc, fake_corpus_path, out)
        assert stats["n_removed"] >= 1
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        ents_s1 = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s1"]
        assert ents_s1 == []

    def test_d_relabel(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        # Đổi label từ PERSON → LOCATION.
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "LOCATION"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(doc, fake_corpus_path, out)
        assert stats["n_updated"] >= 1
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        ents_s1 = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s1"]
        assert any(e["label"] == "LOCATION" for e in ents_s1)
        assert all(e["label"] != "PERSON" for e in ents_s1)

    def test_e_duplicate_text_not_confused(self, duplicate_text_corpus: Path, tmp_path: Path):
        """Hai câu cùng text nhưng id khác → đánh đúng câu theo id."""
        doc = tmp_path / "doc.jsonl"
        # Chỉ thêm entity vào r1-s2 (câu thứ hai), để trống r1-s1.
        doc.write_text(
            json.dumps(
                {
                    "id": "r1-s2",
                    "data": "詔曰：",
                    "label": [[0, 2, "BOOK"]],
                },
                ensure_ascii=False,
            )
            + "\n"
            + json.dumps(
                {"id": "r1-s1", "data": "詔曰：", "label": []},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        import_from_doccano(doc, duplicate_text_corpus, out)
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        ents_s1 = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s1"]
        ents_s2 = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s2"]
        assert ents_s1 == []
        assert len(ents_s2) == 1
        assert ents_s2[0]["label"] == "BOOK"

    def test_f_empty_marked_checked_missing_unchanged(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        # r1-s1 empty + r1-s2 có label → cả 2 checked; r1-s1 stats n_checked_empty.
        # Câu không có trong Doccano → không tự checked.
        doc.write_text(
            json.dumps({"id": "r1-s1", "data": "高祖沛人也。", "label": []}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(doc, fake_corpus_path, out)
        assert stats["n_checked_empty"] == 1
        assert stats["n_missing_in_doccano"] == 1  # r1-s2 vắng
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        sent_s1 = next(s for s in recs[0]["sentences"] if s["sid"] == "r1-s1")
        sent_s2 = next(s for s in recs[0]["sentences"] if s["sid"] == "r1-s2")
        assert sent_s1["review_status"] == "checked"
        assert sent_s2["review_status"] == "needs_review"

    def test_g_duplicate_id_in_doccano_fatal(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps({"id": "r1-s1", "data": "高祖沛人也。", "label": []}, ensure_ascii=False)
            + "\n"
            + json.dumps({"id": "r1-s1", "data": "X", "label": []}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        with pytest.raises(DoccanoError, match="duplicate id"):
            import_from_doccano(doc, fake_corpus_path, out)

    def test_h_atomic_same_path_fatal(self, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        p = tmp_path / "same.jsonl"
        # Tạo file giả.
        p.write_text('{"id": "s1", "data": "x", "label": []}\n', encoding="utf-8")
        with pytest.raises(DoccanoError, match="khác path"):
            import_from_doccano(doc, p, p)


class TestOffsetRoundtrip:
    """Offset conversion global↔relative đúng cho câu bắt đầu không phải 0."""

    def test_non_zero_start(self, non_zero_start_corpus: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        # 高祖 ở offset 0 trong câu (global 150).
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        import_from_doccano(doc, non_zero_start_corpus, out)
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        ents = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s1"]
        # Entity phải có start=150 (global), không phải 0 (relative).
        assert any(e["start"] == 150 and e["end"] == 152 for e in ents)


class TestHumanProvenance:
    """Entity do human review phải có sources=['human'] để strict validator pass."""

    def test_human_entity_has_sources(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        import_from_doccano(doc, fake_corpus_path, out, annotator="alice")
        recs = [json.loads(l) for l in out.read_text(encoding="utf-8").strip().split("\n")]
        ents = [e for e in recs[0]["entities"] if e["sentence_id"] == "r1-s1"]
        assert any(
            e.get("sources") == ["human"]
            and e.get("annotator") == "alice"
            and e.get("priority_score") == 1.0
            for e in ents
        )


class TestGoldMetadata:
    """Sidecar metadata ghi kèm gold."""

    def test_metadata_sidecar(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(
            doc,
            fake_corpus_path,
            out,
            annotator="alice",
            gold_version="v2",
            annotation_guideline_version="0.2",
        )
        assert "gold_metadata" in stats
        meta = stats["gold_metadata"]
        assert meta["gold_version"] == "v2"
        assert meta["annotator"] == "alice"
        assert meta["annotation_guideline_version"] == "0.2"
        assert "source_corpus_sha256" in meta
        assert "doccano_export_sha256" in meta
        meta_path = out.with_suffix(out.suffix + ".meta.json")
        assert meta_path.exists()


class TestOverwriteGuard:
    """Output đã tồn tại → fatal để chống ghi đè gold."""

    def test_existing_output_fatal(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        out.write_text('{"already":"exists"}\n', encoding="utf-8")
        with pytest.raises(DoccanoError, match="không ghi đè"):
            import_from_doccano(doc, fake_corpus_path, out)

    def test_existing_output_allowed_with_force(self, fake_corpus_path: Path, tmp_path: Path):
        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        out.write_text('{"already":"exists"}\n', encoding="utf-8")
        import hcmus_nlp.doccano_io as mod

        prev = getattr(mod.import_from_doccano, "_allow_overwrite", False)
        mod.import_from_doccano._allow_overwrite = True
        try:
            stats = import_from_doccano(doc, fake_corpus_path, out)
            assert stats["n_reviewed_sentences"] >= 1
        finally:
            mod.import_from_doccano._allow_overwrite = prev


class TestConflictResolution:
    """Human review resolve conflicts của sentence đó — chuyển từ
    unresolved_conflicts sang resolved_conflicts với provenance."""

    def test_conflict_moved_to_resolved_on_review(self, fake_corpus_path: Path, tmp_path: Path):
        records = []
        with fake_corpus_path.open() as f:
            for line in f:
                r = json.loads(line)
                r["unresolved_conflicts"] = [
                    {
                        "kind": "partial_overlap",
                        "offset_scope": "record",
                        "sentence_id": "r1-s1",
                        "candidates": [
                            {"start": 0, "end": 2, "label": "PERSON"},
                            {"start": 1, "end": 3, "label": "LOC"},
                        ],
                    }
                ]
                records.append(r)
        with fake_corpus_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        stats = import_from_doccano(doc, fake_corpus_path, out, annotator="alice")
        assert stats["n_resolved_conflicts"] == 1

        with out.open() as f:
            r = json.loads(next(f))
        assert r.get("unresolved_conflicts") == []
        assert len(r.get("resolved_conflicts")) == 1
        resolved = r["resolved_conflicts"][0]
        assert resolved["resolution"] == "human_decision"
        assert resolved["resolved_by"] == "alice"

    def test_only_reviewed_sentence_conflicts_resolved(
        self, fake_corpus_path: Path, tmp_path: Path
    ):
        """Conflict ở sentence không có trong Doccano → không resolve."""
        records = []
        with fake_corpus_path.open() as f:
            for line in f:
                r = json.loads(line)
                r["unresolved_conflicts"] = [
                    {
                        "kind": "partial_overlap",
                        "offset_scope": "record",
                        "sentence_id": "r1-s1",
                        "candidates": [{"start": 0, "end": 2, "label": "X"}],
                    },
                    {
                        "kind": "partial_overlap",
                        "offset_scope": "record",
                        "sentence_id": "r1-s2",
                        "candidates": [{"start": 7, "end": 10, "label": "Y"}],
                    },
                ]
                records.append(r)
        with fake_corpus_path.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        doc = tmp_path / "doc.jsonl"
        doc.write_text(
            json.dumps(
                {"id": "r1-s1", "data": "高祖沛人也。", "label": [[0, 2, "PERSON"]]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        out = tmp_path / "reviewed.jsonl"
        import_from_doccano(doc, fake_corpus_path, out, annotator="alice")
        with out.open() as f:
            r = json.loads(next(f))
        unresolved = r.get("unresolved_conflicts", [])
        assert len(unresolved) == 1
        assert unresolved[0]["sentence_id"] == "r1-s2"
        resolved = r.get("resolved_conflicts", [])
        assert len(resolved) == 1
        assert resolved[0]["sentence_id"] == "r1-s1"


class TestStreamingExport:
    """Streaming export: đọc corpus từng dòng, không load full vào RAM."""

    def test_stream_export(self, fake_corpus_path: Path, tmp_path: Path):
        from hcmus_nlp.doccano_io import export_to_doccano_stream

        out = tmp_path / "doccano.jsonl"
        n = export_to_doccano_stream(fake_corpus_path, out)
        assert n == 2
        lines = out.read_text(encoding="utf-8").strip().split("\n")
        rec1 = json.loads(lines[0])
        assert rec1["id"] == "r1-s1"
