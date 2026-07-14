#!/usr/bin/env python3
"""Compliance check cho submission artifact (Phase H1.2).

Validate manifest, folder/file pairing, sentence_id unique, không overlap,
label membership, source corpus SHA-256.

Đọc --mode từ CLI (KHÔNG sửa manifest). Check tất cả entity (không chỉ
entity đầu). Provenance fields ở final mode là fatal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.validation import compliance_check  # noqa: E402


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", type=Path, default=Path("build/submission"))
    parser.add_argument("--source-corpus", type=Path, help="Path source corpus JSONL")
    parser.add_argument(
        "--validation-report",
        type=Path,
        help="Path validation_report.json. Nếu không truyền, đọc từ "
        "submission/validation_report.json và hash.",
    )
    parser.add_argument(
        "--mode",
        choices=["draft", "pilot", "final"],
        default="draft",
        help="Mode từ CLI; dùng để quyết định minimal vs extended shape có fatal không.",
    )
    parser.add_argument("--expected-sentences", type=int, help="Số câu kỳ vọng từ pipeline")
    parser.add_argument("--output", type=Path, help="Ghi report JSON ra file")
    args = parser.parse_args()

    source_sha = _sha256(args.source_corpus) if args.source_corpus else None

    # Validation report: ưu tiên --validation-report (file external). Nếu
    # không truyền, đọc từ submission/validation_report.json và hash file đó.
    report_sha = None
    report_path = None
    if args.validation_report:
        report_sha = _sha256(args.validation_report)
        report_path = args.validation_report
    else:
        default_report = args.submission / "validation_report.json"
        if default_report.exists():
            report_sha = _sha256(default_report)
            report_path = default_report

    report = compliance_check(
        args.submission,
        expected_sentences=args.expected_sentences,
        source_corpus_sha256=source_sha,
        source_validation_report_sha256=report_sha,
        mode=args.mode,
    )

    payload = report.to_dict()
    payload["mode_checked"] = args.mode
    payload["submission"] = str(args.submission)
    if source_sha:
        payload["source_corpus_sha256_supplied"] = source_sha
    if report_sha:
        payload["source_validation_report_sha256_supplied"] = report_sha
        if report_path:
            payload["source_validation_report_path"] = str(report_path)

    # Cap issues for large artifacts.
    if len(payload["issues"]) > 100:
        payload["issues_truncated"] = True
        payload["issues"] = payload["issues"][:100]

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output:
        report.write(args.output)

    sys.exit(1 if report.fatal_count > 0 else 0)


if __name__ == "__main__":
    main()
