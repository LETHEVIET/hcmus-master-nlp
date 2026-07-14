"""End-to-end final export test (Plan v5 acceptance).

Mục tiêu: chứng minh final mode chạy thành công khi:
1. Mapping confirmed = true.
2. Source corpus mọi sentence checked.
3. Mọi entity có provenance + checked.
4. Cleaning status ∈ {kept, checked}.
5. Không có unresolved_conflicts.
6. Entity text khớp offset, label ∈ INTERNAL_LABELS.

Test này KHÔNG phụ thuộc vào corpus 57k records đã build sẵn — nó tự tạo
fixture corpus nhỏ (1 record, 2 sentences, 2 entities) và chạy toàn bộ
pipeline:
    prepare_corpus → annotate_corpus → Doccano review → export --mode final

Đây là acceptance gate thật cho --mode final (Plan v5 H2).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Tạo workspace tạm với dataset/, config/, scripts/."""
    ws = tmp_path / "ws"
    ws.mkdir()

    # Copy dataset — workspace chứa thư mục dataset/ chuẩn để prepare_corpus
    # quét được và match SOURCE_INFO.
    dataset_dir = ws / "dataset"
    dataset_dir.mkdir()
    dataset_src = Path(__file__).resolve().parent.parent / "dataset"
    if not dataset_src.exists():
        pytest.skip("dataset/ không tồn tại")
    # Copy file nhỏ (漢書 là tác phẩm chính, dùng để test).
    shutil.copy(dataset_src / "漢書_full.txt", dataset_dir / "漢書_full.txt")

    # Tạo mapping confirmed.
    config_dir = ws / "config"
    config_dir.mkdir()
    (config_dir / "mapping.toml").write_text(
        textwrap.dedent(
            """
            version = "0.1.0"
            confirmed = true

            [mapping]
            PERSON         = "PER"
            LOCATION       = "LOC"
            POLITY         = "ORG"
            DYNASTY        = "ORG"
            OFFICIAL_TITLE = "TITLE"
            BOOK           = "TITLE"
            TIME           = "TME"
            NUMBER         = "NUM"
            """
        ).strip(),
        encoding="utf-8",
    )
    return ws


@pytest.fixture
def scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=60)


def test_draft_export_does_not_require_validation_report(tmp_path: Path):
    """Draft mode must not reference final-only validation report state."""
    from hcmus_nlp.labels import load_mapping
    from scripts.export_submission import export

    corpus_path = tmp_path / "preannotated.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "id": "r1",
                "title": "漢書",
                "volume": "001",
                "volume_id": "001",
                "text": "高祖沛人也。",
                "cleaning_status": "kept",
                "sentences": [
                    {
                        "sid": "r1-s1",
                        "start": 0,
                        "end": 6,
                        "text": "高祖沛人也。",
                        "review_status": "needs_review",
                    }
                ],
                "entities": [],
                "unresolved_conflicts": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "submission"
    mapping_path = Path(__file__).resolve().parent.parent / "config" / "mapping.toml"
    manifest = export(
        corpus_path,
        output_dir,
        mode="draft",
        mapping=load_mapping(mapping_path),
        pilot_sentence_ids=None,
    )

    assert manifest["mode"] == "draft"
    assert manifest["source_validation_report_sha256"] is None
    assert (output_dir / "manifest.json").exists()
    assert not (output_dir / "validation_report.json").exists()


def test_final_export_e2e_minimal(workspace: Path, scripts_dir: Path):
    """Chạy full pipeline từ dataset → final export, expect success."""
    # Step 1: prepare_corpus — quét dataset/ trong workspace.
    r = _run(
        [
            sys.executable,
            str(scripts_dir / "prepare_corpus.py"),
            "--input",
            str(workspace / "dataset"),
            "--output",
            str(workspace / "build"),
        ],
        cwd=workspace,
    )
    if r.returncode != 0:
        pytest.fail(f"prepare_corpus failed:\nstdout: {r.stdout}\nstderr: {r.stderr}")
    corpus_path = workspace / "build" / "corpus.jsonl"
    assert corpus_path.exists()

    # Step 2: annotate_corpus
    r = _run(
        [
            sys.executable,
            str(scripts_dir / "annotate_corpus.py"),
            "--input",
            str(corpus_path),
            "--output",
            str(workspace / "build" / "preannotated.jsonl"),
            "--mapping",
            str(workspace / "config" / "mapping.toml"),
        ],
        cwd=workspace,
    )
    if r.returncode != 0:
        pytest.fail(f"annotate_corpus failed:\nstdout: {r.stdout}\nstderr: {r.stderr}")
    preannot_path = workspace / "build" / "preannotated.jsonl"
    assert preannot_path.exists()

    # Step 3: simulate human review bằng cách dùng Doccano I/O round-trip
    # với chính Doccano export từ preannotation → checked corpus.
    doccano_path = workspace / "build" / "doccano.jsonl"
    gold_path = workspace / "build" / "gold.jsonl"

    # Export to Doccano.
    r = _run(
        [
            sys.executable,
            str(scripts_dir / "doccano_io.py"),
            "to-doccano",
            "--input",
            str(preannot_path),
            "--output",
            str(doccano_path),
        ],
        cwd=workspace,
    )
    if r.returncode != 0:
        pytest.fail(f"doccano to-doccano failed:\n{r.stdout}\n{r.stderr}")

    # Import from Doccano (giả lập human review không sửa gì).
    r = _run(
        [
            sys.executable,
            str(scripts_dir / "doccano_io.py"),
            "from-doccano",
            "--doccano",
            str(doccano_path),
            "--input",
            str(preannot_path),
            "--output",
            str(gold_path),
            "--annotator",
            "alice",
        ],
        cwd=workspace,
    )
    if r.returncode != 0:
        pytest.fail(f"doccano from-doccano failed:\n{r.stdout}\n{r.stderr}")
    assert gold_path.exists()

    # Verify gold có provenance + checked.
    with gold_path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            for e in r.get("entities", []):
                assert "sources" in e
                assert "human" in e["sources"]
                assert e["review_status"] == "checked"

    # Step 4: export --mode final — expect SUCCESS (không crash).
    submission_dir = workspace / "build" / "submission"
    r = _run(
        [
            sys.executable,
            str(scripts_dir / "export_submission.py"),
            "--input",
            str(gold_path),
            "--output",
            str(submission_dir),
            "--mode",
            "final",
            "--mapping",
            str(workspace / "config" / "mapping.toml"),
        ],
        cwd=workspace,
    )
    if r.returncode != 0:
        pytest.fail(
            f"final export FAILED (expected success):\nstdout: {r.stdout}\nstderr: {r.stderr}"
        )

    manifest = json.loads((submission_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "final"
    assert manifest["mapping_confirmed"] is True
    assert manifest["sentences"] > 0
    assert manifest["entities"] > 0
    # source_corpus_sha256 phải có.
    assert manifest["source_corpus_sha256"]
    # source_validation_report_sha256 phải có (vì final chạy pre-validation).
    assert manifest["source_validation_report_sha256"]

    # Verify submission structure.
    works = [d for d in submission_dir.iterdir() if d.is_dir()]
    assert len(works) >= 1, "submission phải có ít nhất 1 work folder"
    work = works[0]
    volumes = [d for d in work.iterdir() if d.is_dir()]
    assert len(volumes) >= 1
    vol = volumes[0]
    assert (vol / f"{vol.name}_seg.tsv").exists()
    assert (vol / f"{vol.name}_ner.json").exists()

    # Step 5: compliance --mode final trên submission, với --source-corpus và
    # --validation-report để verify SHA chain.
    validation_report_path = submission_dir / "validation_report.json"
    assert validation_report_path.exists(), "validation_report.json phải tồn tại trong submission"

    r = _run(
        [
            sys.executable,
            str(scripts_dir / "compliance_check.py"),
            "--submission",
            str(submission_dir),
            "--source-corpus",
            str(gold_path),
            "--validation-report",
            str(validation_report_path),
            "--mode",
            "final",
        ],
        cwd=workspace,
    )
    payload = json.loads(r.stdout)
    assert payload["fatal_count"] == 0, (
        f"compliance final có fatal: {payload['fatal_count']}\nfirst 5: {payload['issues'][:5]}"
    )
    # Verify SHA chain: source_corpus + validation_report hashes supplied match.
    assert "source_corpus_sha256_supplied" in payload
    assert "source_validation_report_sha256_supplied" in payload
    # Manifest hashes (từ exporter) phải khớp supplied.
    assert payload["source_corpus_sha256_supplied"] == manifest["source_corpus_sha256"]
    assert (
        payload["source_validation_report_sha256_supplied"]
        == manifest["source_validation_report_sha256"]
    )

    # Step 6: compliance với --validation-report sai → expect fatal mismatch.
    wrong_report = workspace / "wrong_report.json"
    wrong_report.write_text('{"bogus": true}\n', encoding="utf-8")
    r = _run(
        [
            sys.executable,
            str(scripts_dir / "compliance_check.py"),
            "--submission",
            str(submission_dir),
            "--source-corpus",
            str(gold_path),
            "--validation-report",
            str(wrong_report),
            "--mode",
            "final",
        ],
        cwd=workspace,
    )
    payload = json.loads(r.stdout)
    codes = [i["code"] for i in payload["issues"]]
    assert "source_report_hash_mismatch" in codes
    assert payload["fatal_count"] >= 1
