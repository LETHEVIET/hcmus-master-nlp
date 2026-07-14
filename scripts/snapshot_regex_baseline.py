"""Snapshot regex baseline trước khi refactor.

Chạy `annotate_corpus.py` cũ trên sample N record đầu từ corpus.jsonl và lưu
output (sentences + entities) vào `tests/fixtures/regex_baseline/`. Test
`test_c5_refactor_preserves_baseline()` sẽ assert byte-for-byte khớp sau khi
refactor regex source thành adapter ở C5.

Chạy một lần:
    uv run python scripts/snapshot_regex_baseline.py

Sau khi đã snapshot, file được commit. Re-run chỉ khi cố ý thay đổi baseline
(ví dụ sửa regex cũ vì lý do khác).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Bootstrap để chạy được `python3 scripts/snapshot_regex_baseline.py` mà
# không cần editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

# Import module cũ qua importlib để tránh xung đột namespace.
import importlib.util  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OLD_SCRIPT = REPO_ROOT / "scripts" / "annotate_corpus.py"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "regex_baseline"


def _load_old_annotator():
    spec = importlib.util.spec_from_file_location("_old_annotator", OLD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {OLD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(input_path: Path, output_path: Path, n_records: int) -> dict:
    """Đọc n_records đầu từ input_path, chạy annotator cũ, ghi baseline."""
    annotator = _load_old_annotator()
    samples: list[dict] = []
    with input_path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= n_records:
                break
            record = json.loads(line)
            annotated, _ = annotator.annotate_record(record)
            # Chỉ giữ field cần so sánh để diff dễ đọc.
            slim = {
                "id": annotated["id"],
                "title": annotated["title"],
                "volume": annotated.get("volume"),
                "section": annotated.get("section"),
                "source_file": annotated.get("source_file"),
                "source_line": annotated.get("source_line"),
                "text": annotated["text"],
                "sentences": [
                    {
                        "sid": s["sid"],
                        "start": s["start"],
                        "end": s["end"],
                        "text": s["text"],
                        "method": s.get("method"),
                        "review_status": s.get("review_status"),
                    }
                    for s in annotated["sentences"]
                ],
                "entities": [
                    {
                        "eid": e["eid"],
                        "sentence_id": e["sentence_id"],
                        "start": e["start"],
                        "end": e["end"],
                        "text": e["text"],
                        "label": e["label"],
                        "method": e.get("method"),
                        "review_status": e.get("review_status"),
                        "normalized": e.get("normalized"),
                    }
                    for e in annotated["entities"]
                ],
            }
            samples.append(slim)

    payload = {
        "input": str(input_path),
        "n_records": len(samples),
        "samples": samples,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(samples, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "build" / "corpus.jsonl")
    parser.add_argument(
        "--output",
        type=Path,
        default=FIXTURE_DIR / "sample_paragraph.ner.json",
    )
    parser.add_argument("--n-records", type=int, default=50)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input corpus not found: {args.input}. Chạy `python3 scripts/prepare_corpus.py` trước."
        )

    payload = snapshot(args.input, args.output, args.n_records)
    print(
        json.dumps(
            {
                "wrote": str(args.output),
                "n_records": payload["n_records"],
                "sha256": payload["sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
