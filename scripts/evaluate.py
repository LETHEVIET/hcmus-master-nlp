#!/usr/bin/env python3
"""Evaluate pipeline output vs gold (Phase D2).

Usage:
    uv run python scripts/evaluate.py \\
        --gold build/gold/pilot.checked.jsonl \\
        --pred build/corpus_preannotated.jsonl \\
        --filter build/pilot/evaluation_random.jsonl \\
        --baseline build/baseline_regex.jsonl \\
        --output build/pilot/eval_report.json

Metric output: entity_strict P/R/F1, boundary_only F1, per_label F1, label
accuracy on matched boundaries, confusion. So sánh với regex baseline (nếu
truyền --baseline) — acceptance: pipeline mới không được regression strict F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.eval import evaluate, write_report  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def collect_entities(
    records: list[dict],
    *,
    sentence_id_key: str = "sentence_id",
    text_key: str = "text",
) -> list[dict]:
    """Trả list entity flat từ JSONL có thể là gold (per-sentence) hoặc
    corpus (per-record chứa sentences[]/entities[])."""
    entities: list[dict] = []
    for r in records:
        if "sentences" in r and "entities" in r:
            # corpus preannotated format
            for e in r.get("entities", []):
                ent = {
                    "start": e["start"],
                    "end": e["end"],
                    "label": e["label"],
                    "sentence_id": e.get("sentence_id"),
                }
                entities.append(ent)
        elif "entities" in r:
            # gold pilot per-sentence
            for e in r["entities"]:
                ent = {
                    "start": e["start"],
                    "end": e["end"],
                    "label": e["label"],
                    "sentence_id": r.get("sentence_id") or r.get("sid"),
                }
                entities.append(ent)
        else:
            # single entity record
            entities.append(r)
    return entities


def filter_entities_by_sentence(entities: list[dict], keep_sentence_ids: set[str]) -> list[dict]:
    if not keep_sentence_ids:
        return entities
    return [e for e in entities if e.get("sentence_id") in keep_sentence_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Gold JSONL (reviewed)")
    parser.add_argument("--pred", type=Path, default=Path("build/corpus_preannotated.jsonl"))
    parser.add_argument(
        "--filter",
        type=Path,
        help="Pilot JSONL — chỉ đánh giá trên sentence_id trong file này",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Regex-only corpus JSONL để so sánh",
    )
    parser.add_argument("--output", type=Path, default=Path("build/pilot/eval_report.json"))
    args = parser.parse_args()

    gold_records = load_jsonl(args.gold)
    gold_entities = collect_entities(gold_records)

    pred_records = load_jsonl(args.pred)
    pred_entities = collect_entities(pred_records)

    if args.filter:
        pilot_ids = {r.get("sentence_id") or r.get("sid") for r in load_jsonl(args.filter)}
        pilot_ids.discard(None)
        gold_entities = filter_entities_by_sentence(gold_entities, pilot_ids)
        pred_entities = filter_entities_by_sentence(pred_entities, pilot_ids)

    if not gold_entities:
        raise SystemExit("Gold trống sau filter. Kiểm tra gold file + filter sentence_id.")

    report = evaluate(gold_entities, pred_entities)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(report, args.output)

    payload = report.to_dict()
    print(f"strict F1:   {payload['strict']['f1']:.4f}")
    print(f"boundary F1: {payload['boundary_only']['f1']:.4f}")
    print(f"label acc on matched boundaries: {payload['label_accuracy_on_matched_boundaries']:.4f}")
    print(f"n_gold={payload['n_gold']}  n_pred={payload['n_pred']}")

    # So sánh baseline.
    if args.baseline:
        baseline_records = load_jsonl(args.baseline)
        baseline_entities = collect_entities(baseline_records)
        if args.filter:
            baseline_entities = filter_entities_by_sentence(baseline_entities, pilot_ids)
        baseline_report = evaluate(gold_entities, baseline_entities)
        b_payload = baseline_report.to_dict()
        print()
        print(f"baseline strict F1: {b_payload['strict']['f1']:.4f}")
        delta = payload["strict"]["f1"] - b_payload["strict"]["f1"]
        print(f"delta: {delta:+.4f}")
        if delta < 0:
            print("REGRESSION: pipeline strict F1 thấp hơn baseline.")
            sys.exit(2)
        else:
            print("OK: không regression strict F1.")


if __name__ == "__main__":
    main()
