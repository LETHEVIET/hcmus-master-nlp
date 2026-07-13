#!/usr/bin/env python3
"""Validate sentence and entity offsets in the annotated corpus."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def validate(path: Path) -> dict:
    issues = Counter()
    records = sentences = entities = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
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
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
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
                if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("build/corpus_annotated.jsonl"))
    args = parser.parse_args()
    report = validate(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


if __name__ == "__main__":
    main()
