#!/usr/bin/env python3
"""Validate corpus internal (Phase H1.1).

Hai chế độ:
- default (offset-only): chỉ kiểm span câu/entity hợp lệ + text khớp offset.
  Tương thích với hành vi cũ, dùng để smoke test nhanh.
- --strict: dùng `hcmus_nlp.validation.validate_corpus_strict`. Đây là
  validation hai tầng thật của plan v5 — kiểm review_status, cleaning,
  unresolved_conflicts, provenance, mapping confirmation.

Ví dụ:
    uv run python3 scripts/validate_corpus.py
        # offset-only, exit 0 nếu OK
    uv run python3 scripts/validate_corpus.py --strict
        # strict, exit 0 nếu không fatal
    uv run python3 scripts/validate_corpus.py --strict --scope final
        # chặn mapping unconfirmed
    uv run python3 scripts/validate_corpus.py --strict --scope pilot \\
        --pilot build/pilot/evaluation_random.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Bootstrap cho direct-script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.labels import MappingError, load_mapping  # noqa: E402
from hcmus_nlp.validation import validate_corpus_strict  # noqa: E402


def validate_offset_only(path: Path) -> dict:
    """Validator cũ — chỉ kiểm span/text offset. Smoke test nhanh."""
    issues = Counter()
    records = sentences = entities = 0

    with path.open(encoding="utf-8") as handle:
        for _line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues["invalid_json"] += 1
                continue

            records += 1
            text = record.get("text", "")
            sentence_by_id = {}
            previous_end = 0
            for sentence in record.get("sentences", []):
                sentences += 1
                start, end = sentence.get("start"), sentence.get("end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or 0 > start
                    or start >= end
                    or end > len(text)
                ):
                    issues["invalid_sentence_span"] += 1
                    continue
                if text[start:end] != sentence.get("text"):
                    issues["sentence_text_mismatch"] += 1
                if text[previous_end:start].strip():
                    issues["non_whitespace_sentence_gap"] += 1
                previous_end = end
                sentence_by_id[sentence.get("sid")] = sentence

            for entity in record.get("entities", []):
                entities += 1
                start, end = entity.get("start"), entity.get("end")
                if (
                    not isinstance(start, int)
                    or not isinstance(end, int)
                    or 0 > start
                    or start >= end
                    or end > len(text)
                ):
                    issues["invalid_entity_span"] += 1
                    continue
                if text[start:end] != entity.get("text"):
                    issues["entity_text_mismatch"] += 1
                sentence = sentence_by_id.get(entity.get("sentence_id"))
                if sentence is None or not (sentence["start"] <= start and end <= sentence["end"]):
                    issues["entity_outside_sentence"] += 1

    return {
        "file": str(path),
        "records": records,
        "sentences": sentences,
        "entities": entities,
        "issues": dict(issues),
        "valid": not issues,
        "mode": "offset_only",
    }


def load_pilot_ids(pilot_path: Path) -> set[str]:
    ids: set[str] = set()
    with pilot_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("sentence_id") or d.get("sid")
            if sid:
                ids.add(sid)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("build/corpus_preannotated.jsonl"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Dùng validate_corpus_strict (review/cleaning/provenance/mapping).",
    )
    parser.add_argument(
        "--scope",
        choices=["full", "pilot", "final"],
        default="full",
        help="full: toàn bộ; pilot: chỉ sentence trong pilot; final: yêu cầu mapping.confirmed.",
    )
    parser.add_argument("--pilot", type=Path, help="Pilot JSONL (khi --scope pilot)")
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/mapping.toml"),
        help="Path mapping TOML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Ghi report JSON ra file (mặc định: stdout).",
    )
    args = parser.parse_args()

    if not args.strict:
        # Offset-only mode (backward compatible).
        report = validate_offset_only(args.input)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.output:
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        sys.exit(0 if report["valid"] else 1)

    # Strict mode.
    try:
        mapping = load_mapping(args.mapping)
    except MappingError as e:
        raise SystemExit(f"Mapping error: {e}")
    except FileNotFoundError as e:
        raise SystemExit(str(e))

    pilot_ids: set[str] | None = None
    if args.scope == "pilot":
        if not args.pilot:
            raise SystemExit("--scope pilot yêu cầu --pilot PATH")
        pilot_ids = load_pilot_ids(args.pilot)

    report = validate_corpus_strict(
        args.input,
        scope=args.scope,
        mapping=mapping,
        pilot_sentence_ids=pilot_ids,
    )
    payload = report.to_dict()
    payload["input"] = str(args.input)
    payload["mode"] = f"strict:{args.scope}"
    payload["mapping_version"] = mapping.version
    payload["mapping_confirmed"] = mapping.is_confirmed()

    # Cap issue list để tránh spam khi corpus lớn.
    if len(payload["issues"]) > 50:
        payload["issues_truncated"] = True
        payload["issues"] = payload["issues"][:50]

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.output:
        report.write(args.output)

    sys.exit(1 if report.fatal_count > 0 else 0)


if __name__ == "__main__":
    main()
