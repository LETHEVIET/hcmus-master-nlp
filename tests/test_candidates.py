"""Test Trie + CandidateMerger (Phase C3+C4).

Regression bảo vệ:
- Trie: insert / find_all / overlap emission / empty / serialize roundtrip.
- CandidateMerger:
  - same (start, end, label): union sources, max priority, 1 entity.
  - same (start, end), label compatible (DYNASTY/POLITY): 1 entity với
    preferred label.
  - same (start, end), label incompatible: conflict, không emit.
  - strict nested same label: giữ outer (hoặc inner nếu critical).
  - partial overlap khác label: conflict, không emit.
  - disjoint: cả hai emit.
  - deterministic ordering.
- refactor regex source: byte-for-byte với snapshot baseline (c5).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmus_nlp.candidates import CandidateMerger, MergeConflict, Trie
from hcmus_nlp.labels import load_mapping, reset_cache
from hcmus_nlp.source_base import Candidate


@pytest.fixture
def merger(real_mapping_path: Path) -> CandidateMerger:
    reset_cache()
    return CandidateMerger(load_mapping(real_mapping_path))


class TestTrie:
    def test_basic_find(self):
        t = Trie()
        t.insert("高祖", "PERSON")
        results = t.find_all("高祖沛人也")
        assert results == [(0, 2, "高祖", "PERSON", "高祖")]

    def test_multiple_terms(self):
        t = Trie()
        t.insert("高祖", "PERSON")
        t.insert("漢書", "BOOK")
        # 高祖沛人也，《漢書》卷一。 — 漢書 bắt đầu ở index 7.
        results = t.find_all("高祖沛人也，《漢書》卷一。")
        assert (0, 2, "高祖", "PERSON", "高祖") in results
        assert (7, 9, "漢書", "BOOK", "漢書") in results

    def test_overlap_emission(self):
        # `漢` nằm trong `漢書` → emit cả hai (overlap, KHÔNG phải Aho-Corasick).
        t = Trie()
        t.insert("漢", "DYNASTY")
        t.insert("漢書", "BOOK")
        results = t.find_all("《漢書》卷一")
        terms = [(r[0], r[1], r[2]) for r in results]
        # `漢` xuất hiện ở index 1 (trong 《漢書》).
        # `漢書` xuất hiện ở index 1-3.
        assert any(t[2] == "漢" for t in terms)
        assert any(t[2] == "漢書" for t in terms)

    def test_no_match(self):
        t = Trie()
        t.insert("高祖", "PERSON")
        assert t.find_all("無相關內容") == []

    def test_empty_term(self):
        t = Trie()
        t.insert("", "PERSON")  # Không insert được
        assert t.find_all("anything") == []

    def test_serialize_roundtrip(self):
        t = Trie()
        t.insert("高祖", "PERSON")
        t.insert("漢書", "BOOK")
        data = t.to_dict()
        # JSON-able
        json.dumps(data)
        t2 = Trie.from_dict(data)
        results = t2.find_all("高祖沛人也，《漢書》卷一。")
        assert len(results) == 2


class TestCandidateMergerSameSpan:
    def test_same_span_same_label_union(self, merger: CandidateMerger):
        cands = [
            Candidate(text="高祖", label="PERSON", start=0, end=2, source="regex"),
            Candidate(text="高祖", label="PERSON", start=0, end=2, source="seed"),
        ]
        result = merger.merge(cands)
        assert len(result.entities) == 1
        ent = result.entities[0]
        assert ent["start"] == 0 and ent["end"] == 2
        assert ent["label"] == "PERSON"
        assert set(ent["sources"]) == {"regex", "seed"}
        assert ent["priority_score"] == max(cands[0].priority_score, cands[1].priority_score)
        assert result.conflicts == ()

    def test_same_span_compatible_label_preferred(self, merger: CandidateMerger):
        # DYNASTY/POLITY compatible → giữ preferred (DYNASTY priority 0).
        cands = [
            Candidate(text="漢", label="DYNASTY", start=0, end=1, source="seed"),
            Candidate(text="漢", label="POLITY", start=0, end=1, source="regex"),
        ]
        result = merger.merge(cands)
        assert len(result.entities) == 1
        assert result.entities[0]["label"] == "DYNASTY"
        assert "POLITY" in result.entities[0]["merged_from_labels"]
        assert result.conflicts == ()

    def test_same_span_incompatible_label_conflict(self, merger: CandidateMerger):
        # PERSON/LOCATION incompatible → conflict.
        cands = [
            Candidate(text="王", label="PERSON", start=0, end=1, source="regex"),
            Candidate(text="王", label="LOCATION", start=0, end=1, source="seed"),
        ]
        result = merger.merge(cands)
        assert result.entities == ()
        assert len(result.conflicts) == 1
        assert result.conflicts[0].kind == "same_span_label"


class TestCandidateMergerDisjoint:
    def test_disjoint_both_emit(self, merger: CandidateMerger):
        cands = [
            Candidate(text="高祖", label="PERSON", start=0, end=2, source="regex"),
            Candidate(text="沛豐邑", label="LOCATION", start=3, end=6, source="regex"),
        ]
        result = merger.merge(cands)
        assert len(result.entities) == 2
        assert result.conflicts == ()

    def test_three_disjoint(self, merger: CandidateMerger):
        cands = [
            Candidate(text="高祖", label="PERSON", start=0, end=2, source="regex"),
            Candidate(text="沛豐邑", label="LOCATION", start=3, end=6, source="regex"),
            Candidate(text="漢", label="DYNASTY", start=10, end=11, source="regex"),
        ]
        result = merger.merge(cands)
        assert len(result.entities) == 3


class TestCandidateMergerOverlap:
    def test_partial_overlap_different_label_conflict(self, merger: CandidateMerger):
        # 高祖沛 → PERSON (0,2) và 沛豐 (1,3) LOCATION overlap tại index 1-2.
        cands = [
            Candidate(text="高祖沛", label="PERSON", start=0, end=3, source="regex"),
            Candidate(text="祖沛豐", label="LOCATION", start=1, end=4, source="regex"),
        ]
        result = merger.merge(cands)
        # Cả hai vào conflict, không emit.
        assert result.entities == ()
        assert len(result.conflicts) == 1
        assert result.conflicts[0].kind == "partial_overlap"


class TestDeterminism:
    def test_same_input_same_output(self, merger: CandidateMerger):
        cands = [
            Candidate(
                text="高祖", label="PERSON", start=0, end=2, source="regex", priority_score=0.55
            ),
            Candidate(
                text="高祖", label="PERSON", start=0, end=2, source="seed", priority_score=0.70
            ),
        ]
        result1 = merger.merge(cands)
        result2 = merger.merge(cands)
        # Dict không deterministic nếu set; so sánh qua JSON serializable.
        j1 = json.dumps([dict(e) for e in result1.entities], sort_keys=True, default=str)
        j2 = json.dumps([dict(e) for e in result2.entities], sort_keys=True, default=str)
        assert j1 == j2


class TestCriticalSource:
    """Trong strict nested same label, critical_source giữ inner.

    Documented limitation: critical_source ưu tiên inner (KB/model) nhưng
    implementation hiện tại đánh dấu conflict để người duyệt xử lý thay vì
    tự động drop outer. Test xác nhận contract này.
    """

    def test_critical_inner_marks_conflict(self, real_mapping_path: Path):
        reset_cache()
        merger = CandidateMerger(load_mapping(real_mapping_path), critical_sources=("kb",))
        # LOCATION "長安" (0,2) bao bọc LOCATION "長" (0,1) cùng label.
        # Inner từ critical_source (kb) → conflict (nested_critical) để người duyệt.
        cands = [
            Candidate(text="長安", label="LOCATION", start=0, end=2, source="regex"),
            Candidate(text="長", label="LOCATION", start=0, end=1, source="kb"),
        ]
        result = merger.merge(cands)
        # Không emit cả hai — người duyệt phải chọn.
        assert result.entities == ()
        assert any(c.kind == "nested_critical" for c in result.conflicts)


class TestCandidateSourcesHaveProvenance:
    """Entity do merger sinh có `sources`, `priority_score`, `review_status`."""

    def test_emit_fields(self, merger: CandidateMerger):
        cands = [
            Candidate(text="高祖", label="PERSON", start=0, end=2, source="regex"),
        ]
        result = merger.merge(cands)
        assert len(result.entities) == 1
        ent = result.entities[0]
        assert "sources" in ent
        assert "priority_score" in ent
        assert "review_status" in ent
        assert ent["review_status"] == "needs_review"


class TestRefactorPreservesBaseline:
    """Phase C5: refactor regex source thành adapter; output phải khớp
    byte-for-byte với snapshot baseline (trừ field mới)."""

    def test_regex_source_matches_baseline(self, fixtures_dir: Path, sample_records):
        baseline_path = fixtures_dir / "regex_baseline" / "sample_paragraph.ner.json"
        if not baseline_path.exists():
            pytest.skip("Baseline snapshot chưa tạo; chạy scripts/snapshot_regex_baseline.py")

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        # Khởi tạo adapter mới và chạy trên cùng input.
        from hcmus_nlp.source_base import AnnotationContext
        from hcmus_nlp.sources.regex_source import RegexSource

        regex = RegexSource()
        for sample in baseline["samples"]:
            ctx = AnnotationContext(
                record_id=sample["id"],
                title=sample.get("title", ""),
                period=None,
                volume_id=None,
                source_file=sample.get("source_file", ""),
                sentence_spans=tuple(),
            )
            new_candidates = list(regex.candidates(sample["text"], ctx))
            # Convert to same dict shape as baseline entities.
            new_ents = []
            for cand in new_candidates:
                # Tìm sentence_id: dựa vào span có nằm trong sentence nào.
                sentence_id = None
                for sent in sample["sentences"]:
                    if sent["start"] <= cand.start and cand.end <= sent["end"]:
                        sentence_id = sent["sid"]
                        break
                if sentence_id is None:
                    continue
                new_ents.append(
                    {
                        "sentence_id": sentence_id,
                        "start": cand.start,
                        "end": cand.end,
                        "text": cand.text,
                        "label": cand.label,
                    }
                )

            # So sánh với baseline entities (chỉ các field liên quan).
            old_ents = [
                {
                    "sentence_id": e["sentence_id"],
                    "start": e["start"],
                    "end": e["end"],
                    "text": e["text"],
                    "label": e["label"],
                }
                for e in sample["entities"]
            ]

            assert sorted(new_ents, key=lambda x: (x["start"], x["end"])) == sorted(
                old_ents, key=lambda x: (x["start"], x["end"])
            ), (
                f"Refactor differs from baseline at record {sample['id']}: "
                f"new={len(new_ents)} old={len(old_ents)}"
            )
