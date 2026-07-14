"""Test compliance + strict validation (Phase H1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmus_nlp.labels import load_mapping, reset_cache
from hcmus_nlp.validation import (
    ValidationIssue,
    compliance_check,
    validate_corpus_strict,
)


@pytest.fixture
def confirmed_mapping(tmp_path: Path):
    import textwrap

    p = tmp_path / "mapping.toml"
    p.write_text(
        textwrap.dedent(
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
            """
        ),
        encoding="utf-8",
    )
    reset_cache()
    return load_mapping(p)


@pytest.fixture
def unconfirmed_mapping(tmp_path: Path):
    import textwrap

    p = tmp_path / "mapping.toml"
    p.write_text(
        textwrap.dedent(
            """
            version = "0.1.0"
            confirmed = false

            [mapping]
            PERSON = "PER"
            LOCATION = "LOC"
            POLITY = "ORG"
            DYNASTY = "ORG"
            OFFICIAL_TITLE = "TITLE"
            BOOK = "TITLE"
            TIME = "TME"
            NUMBER = "NUM"
            """
        ),
        encoding="utf-8",
    )
    reset_cache()
    return load_mapping(p)


@pytest.fixture
def fake_checked_corpus(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.jsonl"
    # Text "高祖沛人也。沛豐邑。"
    #   idx: 0=高 1=祖 2=沛 3=人 4=也 5=。 6=沛 7=豐 8=邑 9=。
    record = {
        "id": "r1",
        "title": "漢書",
        "text": "高祖沛人也。沛豐邑。",
        "volume_id": "001",
        "cleaning_status": "kept",
        "cleaning_reasons": [],
        "sentences": [
            {
                "sid": "r1-s1",
                "start": 0,
                "end": 6,
                "text": "高祖沛人也。",
                "review_status": "checked",
            },
            {"sid": "r1-s2", "start": 6, "end": 10, "text": "沛豐邑。", "review_status": "checked"},
        ],
        "entities": [
            {
                "eid": "r1-e1",
                "sentence_id": "r1-s1",
                "start": 0,
                "end": 2,
                "text": "高祖",
                "label": "PERSON",
                "sources": ["regex"],
                "review_status": "checked",
            },
            {
                "eid": "r1-e2",
                "sentence_id": "r1-s2",
                "start": 6,
                "end": 9,
                "text": "沛豐邑",
                "label": "LOCATION",
                "sources": ["regex"],
                "review_status": "checked",
            },
        ],
    }
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return p


class TestExtendedStrictChecks:
    """Check entity.text, entity_in_sentence, internal_label."""

    def _make_corpus(
        self,
        tmp_path: Path,
        *,
        entity_text: str = "高祖",
        start: int = 0,
        end: int = 2,
        label: str = "PERSON",
    ) -> Path:
        p = tmp_path / "corpus.jsonl"
        record = {
            "id": "r1",
            "title": "漢書",
            "text": "高祖沛人也。沛豐邑。",
            "volume_id": "001",
            "cleaning_status": "kept",
            "cleaning_reasons": [],
            "sentences": [
                {
                    "sid": "r1-s1",
                    "start": 0,
                    "end": 7,
                    "text": "高祖沛人也。",
                    "review_status": "checked",
                },
                {
                    "sid": "r1-s2",
                    "start": 7,
                    "end": 11,
                    "text": "沛豐邑。",
                    "review_status": "checked",
                },
            ],
            "entities": [
                {
                    "eid": "r1-e1",
                    "sentence_id": "r1-s1",
                    "start": start,
                    "end": end,
                    "text": entity_text,
                    "label": label,
                    "sources": ["regex"],
                    "review_status": "checked",
                },
            ],
        }
        p.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        return p

    def test_entity_text_mismatch_fatal(self, tmp_path: Path, confirmed_mapping):
        # text claim "高祖" nhưng span thực tế là "高祖沛" → mismatch.
        p = self._make_corpus(tmp_path, entity_text="高祖", start=0, end=3)
        r = validate_corpus_strict(p, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "entity_text_mismatch" in codes

    def test_entity_outside_sentence_fatal(self, tmp_path: Path, confirmed_mapping):
        # Entity nằm ngoài sentence gán cho nó (r1-s1 chỉ tới offset 7).
        p = self._make_corpus(tmp_path, start=8, end=10)
        r = validate_corpus_strict(p, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "entity_outside_sentence" in codes

    def test_entity_orphan_sentence_fatal(self, tmp_path: Path, confirmed_mapping):
        # Entity tham chiếu sentence không tồn tại.
        p = tmp_path / "corpus.jsonl"
        record = {
            "id": "r1",
            "title": "漢書",
            "text": "高祖沛人也。",
            "cleaning_status": "kept",
            "cleaning_reasons": [],
            "sentences": [
                {
                    "sid": "r1-s1",
                    "start": 0,
                    "end": 7,
                    "text": "高祖沛人也。",
                    "review_status": "checked",
                }
            ],
            "entities": [
                {
                    "eid": "r1-e1",
                    "sentence_id": "r1-sX",  # không tồn tại
                    "start": 0,
                    "end": 2,
                    "text": "高祖",
                    "label": "PERSON",
                    "sources": ["regex"],
                    "review_status": "checked",
                },
            ],
        }
        p.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        r = validate_corpus_strict(p, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "entity_orphan_sentence" in codes

    def test_invalid_internal_label_fatal(self, tmp_path: Path, confirmed_mapping):
        p = self._make_corpus(tmp_path, label="BOGUS_LABEL")
        r = validate_corpus_strict(p, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "invalid_internal_label" in codes

    def test_pilot_scope_skips_out_of_scope_records(self, tmp_path: Path, confirmed_mapping):
        # 2 record: 1 trong pilot (clean), 1 ngoài pilot (cleaning_needs_review).
        # Pilot scope chỉ check record trong scope.
        pilot_ids = {"r1-s1"}
        p = tmp_path / "corpus.jsonl"
        records = [
            {
                "id": "r1",
                "title": "漢書",
                "text": "高祖沛人也。",
                "cleaning_status": "kept",
                "cleaning_reasons": [],
                "sentences": [
                    {
                        "sid": "r1-s1",
                        "start": 0,
                        "end": 7,
                        "text": "高祖沛人也。",
                        "review_status": "checked",
                    }
                ],
                "entities": [],
            },
            {
                "id": "r2",
                "title": "漢書",
                "text": "沛豐邑。",
                "cleaning_status": "needs_review",  # sẽ trigger cleaning_needs_review nếu không scope filter
                "cleaning_reasons": ["inline_ocr_placeholder"],
                "sentences": [
                    {
                        "sid": "r2-s1",
                        "start": 0,
                        "end": 4,
                        "text": "沛豐邑。",
                        "review_status": "checked",
                    }
                ],
                "entities": [],
            },
        ]
        with p.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        r = validate_corpus_strict(
            p, scope="pilot", mapping=confirmed_mapping, pilot_sentence_ids=pilot_ids
        )
        codes = [i.code for i in r.issues]
        # r2 nằm ngoài pilot → KHÔNG bị tính cleaning_needs_review.
        assert "cleaning_needs_review" not in codes


class TestValidateCorpusStrict:
    def test_clean_corpus_passes(self, fake_checked_corpus: Path, confirmed_mapping):
        r = validate_corpus_strict(fake_checked_corpus, scope="full", mapping=confirmed_mapping)
        assert r.fatal_count == 0

    def test_unchecked_sentence_fatal(self, fake_checked_corpus: Path, confirmed_mapping):
        # Sửa 1 sentence thành needs_review.
        records = []
        with fake_checked_corpus.open() as f:
            for line in f:
                r = json.loads(line)
                r["sentences"][0]["review_status"] = "needs_review"
                records.append(r)
        with fake_checked_corpus.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        r = validate_corpus_strict(fake_checked_corpus, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "unchecked_sentence" in codes
        assert r.fatal_count >= 1

    def test_cleaning_needs_review_fatal(self, fake_checked_corpus: Path, confirmed_mapping):
        records = []
        with fake_checked_corpus.open() as f:
            for line in f:
                r = json.loads(line)
                r["cleaning_status"] = "needs_review"
                records.append(r)
        with fake_checked_corpus.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        r = validate_corpus_strict(fake_checked_corpus, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "cleaning_needs_review" in codes

    def test_missing_provenance_fatal(self, fake_checked_corpus: Path, confirmed_mapping):
        records = []
        with fake_checked_corpus.open() as f:
            for line in f:
                r = json.loads(line)
                # Xóa sources ở entity đầu.
                if r["entities"]:
                    r["entities"][0].pop("sources", None)
                records.append(r)
        with fake_checked_corpus.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        r = validate_corpus_strict(fake_checked_corpus, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "missing_provenance" in codes

    def test_unresolved_conflicts_fatal(self, fake_checked_corpus: Path, confirmed_mapping):
        records = []
        with fake_checked_corpus.open() as f:
            for line in f:
                r = json.loads(line)
                r["unresolved_conflicts"] = [{"kind": "partial_overlap", "candidates": []}]
                records.append(r)
        with fake_checked_corpus.open("w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        r = validate_corpus_strict(fake_checked_corpus, scope="full", mapping=confirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "unresolved_conflicts" in codes

    def test_unconfirmed_mapping_fatal(self, fake_checked_corpus: Path, unconfirmed_mapping):
        r = validate_corpus_strict(fake_checked_corpus, scope="final", mapping=unconfirmed_mapping)
        codes = [i.code for i in r.issues]
        assert "mapping_unconfirmed" in codes


class TestComplianceCheck:
    """Test compliance_check() trên submission artifact fixture."""

    def test_extra_provenance_fatal_at_final(self, fake_submission: Path):
        """Provenance fields ở mode=final là fatal, không warning."""
        ner_path = fake_submission / "HCH_006" / "HCH_006_001" / "HCH_006_001_ner.json"
        data = json.loads(ner_path.read_text(encoding="utf-8"))
        data[0]["entities"][0]["sources"] = ["human"]  # field thừa ở final
        ner_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        r = compliance_check(fake_submission, mode="final")
        codes = [i.code for i in r.issues]
        assert "extra_provenance_fields" in codes
        assert r.fatal_count >= 1

    def test_minimal_shape_draft_no_fatal(self, fake_submission: Path):
        """Ở draft mode, field extra không fatal."""
        ner_path = fake_submission / "HCH_006" / "HCH_006_001" / "HCH_006_001_ner.json"
        data = json.loads(ner_path.read_text(encoding="utf-8"))
        data[0]["entities"][0]["sources"] = ["human"]
        ner_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        r = compliance_check(fake_submission)  # default draft
        codes = [i.code for i in r.issues]
        assert "extra_provenance_fields" not in codes

    @pytest.fixture
    def fake_submission(self, tmp_path: Path) -> Path:
        """Tạo submission fixture: 1 work, 1 volume, 2 sentences."""
        sub = tmp_path / "submission"
        work = sub / "HCH_006"
        vol = work / "HCH_006_001"
        vol.mkdir(parents=True)

        base = "HCH_006_001"
        seg_path = vol / f"{base}_seg.tsv"
        ner_path = vol / f"{base}_ner.json"
        seg_path.write_text(
            "HCH_006_001_000001\t高祖沛人也。\nHCH_006_001_000002\t沛豐邑中陽里。\n",
            encoding="utf-8",
        )
        ner_path.write_text(
            json.dumps(
                [
                    {
                        "sentence_id": "HCH_006_001_000001",
                        "sentence": "高祖沛人也。",
                        "entities": [
                            {"text": "高祖", "label": "PER", "start": 0, "end": 2},
                        ],
                    },
                    {
                        "sentence_id": "HCH_006_001_000002",
                        "sentence": "沛豐邑中陽里。",
                        "entities": [
                            {"text": "沛豐邑", "label": "LOC", "start": 0, "end": 3},
                        ],
                    },
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        manifest = {
            "mode": "draft",
            "sentences": 2,
            "input_sha256": "abc",
            "source_corpus_sha256": "abc",
        }
        (sub / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return sub

    def test_clean_submission_passes(self, fake_submission: Path):
        r = compliance_check(fake_submission)
        assert r.fatal_count == 0
        assert r.n_entities == 2

    def test_seg_ner_mismatch_fatal(self, fake_submission: Path):
        # Sửa _seg.tsv để sentence_id khác.
        ner_path = fake_submission / "HCH_006" / "HCH_006_001" / "HCH_006_001_ner.json"
        data = json.loads(ner_path.read_text(encoding="utf-8"))
        data[0]["sentence_id"] = "DIFFERENT"
        ner_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        r = compliance_check(fake_submission)
        codes = [i.code for i in r.issues]
        assert "seg_ner_mismatch" in codes

    def test_duplicate_sentence_id_fatal(self, fake_submission: Path):
        # Trùng sentence_id trong 2 folder khác nhau.
        work = fake_submission / "HCH_006"
        vol2 = work / "HCH_006_002"
        vol2.mkdir()
        (vol2 / "HCH_006_002_seg.tsv").write_text("HCH_006_001_000001\tX\n", encoding="utf-8")
        (vol2 / "HCH_006_002_ner.json").write_text(
            json.dumps(
                [{"sentence_id": "HCH_006_001_000001", "sentence": "X", "entities": []}],
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        r = compliance_check(fake_submission)
        codes = [i.code for i in r.issues]
        assert "duplicate_sentence_id" in codes

    def test_invalid_label_fatal(self, fake_submission: Path):
        ner_path = fake_submission / "HCH_006" / "HCH_006_001" / "HCH_006_001_ner.json"
        data = json.loads(ner_path.read_text(encoding="utf-8"))
        data[0]["entities"][0]["label"] = "BOGUS"
        ner_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        r = compliance_check(fake_submission)
        codes = [i.code for i in r.issues]
        assert "invalid_label" in codes

    def test_entity_overlap_fatal(self, fake_submission: Path):
        ner_path = fake_submission / "HCH_006" / "HCH_006_001" / "HCH_006_001_ner.json"
        data = json.loads(ner_path.read_text(encoding="utf-8"))
        data[0]["entities"].append({"text": "高祖沛", "label": "PER", "start": 0, "end": 3})
        ner_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        r = compliance_check(fake_submission)
        codes = [i.code for i in r.issues]
        assert "entity_overlap" in codes

    def test_source_corpus_sha_mismatch_fatal(self, fake_submission: Path):
        r = compliance_check(fake_submission, source_corpus_sha256="DIFFERENT")
        codes = [i.code for i in r.issues]
        assert "source_corpus_hash_mismatch" in codes

    def test_sentence_count_mismatch_fatal(self, fake_submission: Path):
        r = compliance_check(fake_submission, expected_sentences=999)
        codes = [i.code for i in r.issues]
        assert "sentence_count_mismatch" in codes
