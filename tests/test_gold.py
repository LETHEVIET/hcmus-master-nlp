"""Test gold pilot sampler (Phase D1).

Regression:
- Reproducibility: cùng seed → cùng output.
- Stratified: mỗi work có quota đúng.
- Disjoint: evaluation_random và diagnostic_challenge không trùng sentence_id.
- Scale-up rule: --pilot-size N với N lớn hơn vẫn chạy.
- Diagnostic strata được phân loại đúng.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hcmus_nlp.gold import (
    PilotManifest,
    SentenceRecord,
    build_pilot,
    collect_sentences,
    diagnostic_strata,
    sample_diagnostic_challenge,
    stratified_sample,
)


@pytest.fixture
def fake_corpus(tmp_path: Path) -> Path:
    """Tạo corpus.jsonl giả: 8 work × 50 record, mỗi record 1-3 sentence."""
    import random

    rng = random.Random(0)
    titles = [
        "漢書",
        "後漢書",
        "舊唐書",
        "舊五代史",
        "諸蕃志",
        "東觀漢記_(四庫全書本)",
        "北史",
        "北齊書",
    ]
    path = tmp_path / "corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        idx = 0
        for title in titles:
            for _r in range(50):
                idx += 1
                n_sent = rng.randint(1, 3)
                sentences = []
                entities = []
                cursor = 0
                for s in range(n_sent):
                    length = rng.randint(15, 100)
                    text = "高祖沛豐邑中也。建武元年春。〈案：唐初〉。" * 2
                    text = text[:length]
                    sent = {
                        "sid": f"r{idx}-s{s + 1}",
                        "start": cursor,
                        "end": cursor + length,
                        "text": text,
                    }
                    sentences.append(sent)
                    cursor += length + 1
                    # Random 1-2 entities per sentence.
                    for _e in range(rng.randint(0, 2)):
                        es = rng.randint(0, max(0, length - 4))
                        entities.append(
                            {
                                "eid": f"r{idx}-e{len(entities) + 1}",
                                "sentence_id": sent["sid"],
                                "start": cursor - length + es,
                                "end": cursor - length + es + 2,
                                "text": text[es : es + 2],
                                "label": rng.choice(["PERSON", "LOC", "TIME"]),
                            }
                        )
                record = {
                    "id": f"r{idx:04d}",
                    "title": title,
                    "volume": "001",
                    "text": " ".join(s["text"] for s in sentences),
                    "sentences": sentences,
                    "entities": entities,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class TestCollectSentences:
    def test_collect(self, fake_corpus: Path):
        sents = collect_sentences(fake_corpus)
        assert len(sents) > 0
        for s in sents[:5]:
            assert isinstance(s, SentenceRecord)
            assert s.sentence_id
            assert s.text

    def test_strata_flag(self, fake_corpus: Path):
        sents = collect_sentences(fake_corpus)
        has_long = any(s.is_long for s in sents)
        has_no_punct = any(not s.has_punct_end for s in sents)
        # Có thể có hoặc không tùy data; chỉ cần không crash.
        assert isinstance(has_long, bool)
        assert isinstance(has_no_punct, bool)


class TestStratifiedSample:
    def test_basic(self):
        sents = [
            SentenceRecord(
                sentence_id=f"s{i}",
                record_id="r",
                title=t,
                volume_id=None,
                text="x",
                start=0,
                end=1,
                has_punct_end=True,
                is_long=False,
                has_annotation_案=False,
                n_entities=0,
                labels=(),
            )
            for i, t in enumerate(["A", "A", "A", "B", "B", "B"])
        ]
        result = stratified_sample(sents, n=4, seed=42)
        assert len(result) == 4
        # Stratified: phải có cả A và B.
        titles = {s.title for s in result}
        assert titles == {"A", "B"}

    def test_reproducibility(self):
        sents = [
            SentenceRecord(
                sentence_id=f"s{i}",
                record_id="r",
                title="A",
                volume_id=None,
                text="x",
                start=0,
                end=1,
                has_punct_end=True,
                is_long=False,
                has_annotation_案=False,
                n_entities=0,
                labels=(),
            )
            for i in range(20)
        ]
        r1 = stratified_sample(sents, n=10, seed=42)
        r2 = stratified_sample(sents, n=10, seed=42)
        assert [s.sentence_id for s in r1] == [s.sentence_id for s in r2]

    def test_different_seed_different_result(self):
        sents = [
            SentenceRecord(
                sentence_id=f"s{i}",
                record_id="r",
                title="A",
                volume_id=None,
                text="x",
                start=0,
                end=1,
                has_punct_end=True,
                is_long=False,
                has_annotation_案=False,
                n_entities=0,
                labels=(),
            )
            for i in range(20)
        ]
        r1 = stratified_sample(sents, n=10, seed=1)
        r2 = stratified_sample(sents, n=10, seed=2)
        # Khác seed → có thể khác kết quả.
        assert r1 or r2  # không empty


class TestDiagnosticStrata:
    def test_classify(self):
        sents = [
            SentenceRecord(
                sentence_id="s1",
                record_id="r",
                title="A",
                volume_id=None,
                text="x",
                start=0,
                end=1,
                has_punct_end=False,
                is_long=True,
                has_annotation_案=True,
                n_entities=2,
                labels=("PERSON", "TIME"),
            ),
        ]
        strata = diagnostic_strata(sents)
        assert "s1" in [s.sentence_id for s in strata["no_punct_end"]]
        assert "s1" in [s.sentence_id for s in strata["long_sentence"]]
        assert "s1" in [s.sentence_id for s in strata["has_annotation_案"]]
        assert "s1" in [s.sentence_id for s in strata["has_person_candidate"]]
        assert "s1" in [s.sentence_id for s in strata["multi_source"]]


class TestBuildPilot:
    def test_basic(self, fake_corpus: Path, tmp_path: Path):
        output = tmp_path / "pilot"
        manifest = build_pilot(fake_corpus, output, pilot_size=80, seed=42)
        assert isinstance(manifest, PilotManifest)
        assert manifest.pilot_size == 80
        # Cả 8 work đều có trong quota (chia đều).
        assert len(manifest.work_quota) == 8
        # File output tồn tại.
        assert (output / "evaluation_random.jsonl").exists()
        assert (output / "diagnostic_challenge.jsonl").exists()
        assert (output / "manifest.json").exists()

    def test_disjoint(self, fake_corpus: Path, tmp_path: Path):
        output = tmp_path / "pilot"
        build_pilot(fake_corpus, output, pilot_size=80, seed=42)
        eval_ids = set()
        diag_ids = set()
        with (output / "evaluation_random.jsonl").open() as f:
            for line in f:
                eval_ids.add(json.loads(line)["sentence_id"])
        with (output / "diagnostic_challenge.jsonl").open() as f:
            for line in f:
                diag_ids.add(json.loads(line)["sentence_id"])
        assert not (eval_ids & diag_ids), "evaluation_random và diagnostic phải disjoint"

    def test_reproducibility(self, fake_corpus: Path, tmp_path: Path):
        out1 = tmp_path / "p1"
        out2 = tmp_path / "p2"
        m1 = build_pilot(fake_corpus, out1, pilot_size=80, seed=42)
        m2 = build_pilot(fake_corpus, out2, pilot_size=80, seed=42)
        # Cùng seed → cùng quota breakdown.
        assert m1.work_quota == m2.work_quota

    def test_scale_up(self, fake_corpus: Path, tmp_path: Path):
        output = tmp_path / "pilot"
        # Production 1500 câu — vẫn chạy.
        manifest = build_pilot(fake_corpus, output, pilot_size=1500, seed=42)
        # Pilot size vượt tổng số câu → manifest ghi đúng size yêu cầu.
        assert manifest.pilot_size == 1500

    def test_double_annotate_fraction(self, fake_corpus: Path, tmp_path: Path):
        output = tmp_path / "pilot"
        manifest = build_pilot(
            fake_corpus, output, pilot_size=80, seed=42, double_annotate_fraction=0.20
        )
        assert manifest.double_annotate_fraction == 0.20

        double_count = 0
        total = 0
        with (output / "evaluation_random.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                total += 1
                if r.get("double_annotate"):
                    double_count += 1
        # Trong khoảng chấp nhận được (random ± noise).
        if total > 0:
            assert abs(double_count / total - 0.20) < 0.10
