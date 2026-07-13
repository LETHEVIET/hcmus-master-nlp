#!/usr/bin/env python3
"""Add sentence boundaries and reviewable NER candidates to corpus JSONL.

This is intentionally a corpus-building tool, not a model-training pipeline.
Sentence boundaries are created conservatively from existing punctuation. NER
output is pre-annotation only: every heuristic entity is marked
``needs_review`` and must be checked before being called gold data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


OPEN_TO_CLOSE = {"〈": "〉", "《": "》", "（": "）", "(": ")", "「": "」", "『": "』", "【": "】", "[": "]"}
CLOSE_CHARS = set(OPEN_TO_CLOSE.values())
SENTENCE_ENDS = set("。！？!?｡")

ERA_NAMES = (
    "建武|永平|建安|元嘉|太和|孝昌|武德|貞觀|開元|天寶|乾隆|寶慶|更始|"
    "地皇|天鳳|永興|正始|太康|咸和|大業|顯慶|神龍|天成|同光|廣順|乾祐|顯德"
)
TIME_RE = re.compile(rf"(?:{ERA_NAMES})?(?:元年|[一二三四五六七八九十百千万〇零０-９0-9]{{1,4}}年)")
BOOK_RE = re.compile(r"《[^》\n]{1,40}》")
OFFICIAL_RE = re.compile(
    r"(?:太守|刺史|將軍|大將軍|司馬|尚書|侍郎|丞相|御史|博士|校尉|中郎將|令史|大夫|太子|公主|皇帝|皇后|侯國|縣令)"
)
LOCATION_RE = re.compile(r"[\u3400-\u9fff]{1,6}(?:郡|縣|州|邑|城|關|鄉|鎮)")
POLITY_RE = re.compile(r"(?:漢|魏|吳|蜀|秦|楚|齊|梁|陳|周|晉|隋|唐|宋|遼|金|元|明|清)(?:朝|國|氏)")


def split_sentences(text: str) -> list[dict]:
    """Split at strong sentence punctuation outside paired editorial spans."""
    sentences: list[dict] = []
    stack: list[str] = []
    start = 0

    def emit(end: int) -> None:
        nonlocal start
        raw = text[start:end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trim = len(raw.rstrip())
        s = start + left_trim
        e = start + right_trim
        if s < e:
            sentences.append({"start": s, "end": e, "text": text[s:e]})
        start = end

    for index, char in enumerate(text):
        if char in OPEN_TO_CLOSE:
            stack.append(OPEN_TO_CLOSE[char])
        elif stack and char == stack[-1]:
            stack.pop()
        elif char in CLOSE_CHARS and stack:
            # Keep malformed/unbalanced markup in the same sentence rather
            # than losing text or guessing a new boundary.
            if char in stack:
                stack = stack[: len(stack) - 1 - stack[::-1].index(char)]

        if char in SENTENCE_ENDS and not stack:
            emit(index + 1)

    emit(len(text))
    return sentences


def candidate_matches(text: str) -> list[dict]:
    """Generate conservative, non-overlapping candidates for human review."""
    matches: list[dict] = []
    occupied: list[tuple[int, int]] = []

    def overlaps(start: int, end: int) -> bool:
        return any(start < other_end and end > other_start for other_start, other_end in occupied)

    # Longer/more explicit patterns have priority over broad location rules.
    patterns = [
        ("BOOK", BOOK_RE),
        ("TIME", TIME_RE),
        ("OFFICIAL_TITLE", OFFICIAL_RE),
        ("POLITY", POLITY_RE),
        ("LOCATION", LOCATION_RE),
    ]
    for label, pattern in patterns:
        for match in pattern.finditer(text):
            start, end = match.span()
            if overlaps(start, end):
                continue
            occupied.append((start, end))
            matches.append({
                "start": start,
                "end": end,
                "text": match.group(0),
                "label": label,
                "method": "heuristic",
                "review_status": "needs_review",
                "normalized": None,
            })
    return sorted(matches, key=lambda item: (item["start"], item["end"]))


def annotate_record(record: dict) -> tuple[dict, Counter]:
    text = record["text"]
    sentences = split_sentences(text)
    entities: list[dict] = []
    counts = Counter()

    for number, sentence in enumerate(sentences, start=1):
        sentence["sid"] = f"{record['id']}-s{number}"
        sentence["method"] = "rule"
        sentence["review_status"] = "needs_review"
        for entity_number, entity in enumerate(candidate_matches(sentence["text"]), start=1):
            entity["start"] += sentence["start"]
            entity["end"] += sentence["start"]
            entity["eid"] = f"{record['id']}-e{len(entities) + 1}"
            entity["sentence_id"] = sentence["sid"]
            entities.append(entity)
            counts[entity["label"]] += 1

    result = dict(record)
    result["sentences"] = sentences
    result["entities"] = entities
    result["annotation"] = {
        "sentence_guideline_version": "0.1",
        "ner_guideline_version": "0.1",
        "status": "preannotation_needs_review",
        "offset_convention": "Unicode code points; start inclusive, end exclusive",
    }
    return result, counts


def build(input_path: Path, output_path: Path, stats_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    records = 0
    sentences = 0
    entities = 0
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            record = json.loads(line)
            annotated, counts = annotate_record(record)
            target.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            records += 1
            sentences += len(annotated["sentences"])
            entities += len(annotated["entities"])
            totals.update(counts)

    stats_path.write_text(json.dumps({
        "records": records,
        "sentences": sentences,
        "entities": entities,
        "entities_by_label": dict(totals),
        "annotation_status": "preannotation_needs_review",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/corpus.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("build/corpus_annotated.jsonl"))
    parser.add_argument("--stats", type=Path, default=Path("build/annotation_statistics.json"))
    args = parser.parse_args()
    build(args.input, args.output, args.stats)
    print(f"Wrote annotated corpus to {args.output}")


if __name__ == "__main__":
    main()
