"""Test eval framework (Phase D2).

Regression:
- entity_strict: (start, end, label) exact.
- boundary_only: (start, end) match, bỏ label.
- per_label_f1 đúng cho mỗi label.
- label_accuracy_on_matched_boundaries.
- Confusion matrix.
- Edge case: gold rỗng, pred rỗng, mismatch sentence_id.
"""

from __future__ import annotations

import pytest

from hcmus_nlp.eval import EvalReport, Metrics, evaluate


def test_perfect_match():
    gold = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
    ]
    pred = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
    ]
    r = evaluate(gold, pred)
    assert r.strict.f1 == 1.0
    assert r.boundary_only.f1 == 1.0
    assert r.label_accuracy_on_matched_boundaries == 1.0


def test_completely_wrong():
    gold = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
    ]
    pred = [
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
    ]
    r = evaluate(gold, pred)
    assert r.strict.f1 == 0.0
    assert r.strict.tp == 0
    assert r.strict.fp == 1
    assert r.strict.fn == 1


def test_boundary_match_label_wrong():
    """Boundary match + label sai: TP cho boundary, FN cho strict."""
    gold = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
    ]
    pred = [
        {"start": 0, "end": 2, "label": "LOC", "sentence_id": "s1"},
    ]
    r = evaluate(gold, pred)
    assert r.strict.tp == 0  # label khác → strict fail
    assert r.strict.fp == 1
    assert r.strict.fn == 1
    assert r.boundary_only.tp == 1  # boundary khớp → boundary TP
    assert r.boundary_only.f1 == 1.0
    # Label accuracy trên matched boundary: 0/1 = 0
    assert r.label_accuracy_on_matched_boundaries == 0.0


def test_partial_overlap_strict_fail():
    gold = [{"start": 0, "end": 5, "label": "PERSON", "sentence_id": "s1"}]
    pred = [{"start": 0, "end": 3, "label": "PERSON", "sentence_id": "s1"}]
    r = evaluate(gold, pred)
    assert r.strict.tp == 0  # khác end → strict fail
    assert r.boundary_only.tp == 0  # khác end → boundary fail


def test_sentence_id_isolation():
    """Hai entity ở hai sentence khác nhau nhưng cùng span/label là khác nhau."""
    gold = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}]
    pred = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s2"}]
    r = evaluate(gold, pred)
    assert r.strict.tp == 0


def test_per_label_breakdown():
    gold = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
        {"start": 10, "end": 12, "label": "LOC", "sentence_id": "s1"},
    ]
    pred = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
        {"start": 10, "end": 12, "label": "PERSON", "sentence_id": "s1"},  # sai label
    ]
    r = evaluate(gold, pred)
    # LOC: 2 gold, 1 pred (matched 1) → P=1.0, R=0.5, F1=0.667
    assert r.per_label["LOC"].precision == 1.0
    assert r.per_label["LOC"].recall == 0.5
    assert abs(r.per_label["LOC"].f1 - 2 / 3) < 1e-6
    # PERSON: 1 gold, 2 pred (matched 1) → P=0.5, R=1.0, F1=0.667
    assert r.per_label["PERSON"].precision == 0.5
    assert r.per_label["PERSON"].recall == 1.0
    assert abs(r.per_label["PERSON"].f1 - 2 / 3) < 1e-6


def test_confusion_matrix():
    gold = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}]
    pred = [{"start": 0, "end": 2, "label": "LOC", "sentence_id": "s1"}]
    r = evaluate(gold, pred)
    assert r.confusion[("LOC", "PERSON")] == 1


def test_empty_gold_or_pred():
    r1 = evaluate([], [])
    assert r1.strict.tp == 0 and r1.n_gold == 0 and r1.n_pred == 0

    r2 = evaluate([{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}], [])
    assert r2.strict.tp == 0 and r2.strict.fn == 1

    r3 = evaluate([], [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}])
    assert r3.strict.tp == 0 and r3.strict.fp == 1


def test_prf_zero_division():
    r = evaluate([], [{"start": 0, "end": 2, "label": "X", "sentence_id": "s1"}])
    assert r.strict.precision == 0.0
    assert r.strict.recall == 0.0
    assert r.strict.f1 == 0.0


def test_report_to_dict():
    gold = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}]
    pred = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}]
    r = evaluate(gold, pred)
    d = r.to_dict()
    assert "strict" in d
    assert "boundary_only" in d
    assert "per_label" in d
    assert "label_accuracy_on_matched_boundaries" in d
    assert "confusion" in d


def test_acceptance_baseline_diff_logic():
    """Verify pattern mà scripts/evaluate.py dùng: pipeline F1 < baseline → exit 2."""
    gold = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
    ]
    # Pipeline tốt hơn baseline.
    pipeline_pred = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
        {"start": 5, "end": 7, "label": "LOC", "sentence_id": "s1"},
    ]
    # Baseline chỉ match 1.
    baseline_pred = [
        {"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"},
    ]
    p = evaluate(gold, pipeline_pred)
    b = evaluate(gold, baseline_pred)
    assert p.strict.f1 >= b.strict.f1  # không regression


def test_acceptance_no_regression():
    """Pipeline strict F1 >= baseline strict F1 là acceptance gate."""
    gold = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}]
    pipeline_pred = [{"start": 0, "end": 2, "label": "PERSON", "sentence_id": "s1"}]
    baseline_pred = []  # empty baseline
    p = evaluate(gold, pipeline_pred)
    b = evaluate(gold, baseline_pred)
    # Pipeline có F1=1, baseline có F1=0 → không regression.
    assert p.strict.f1 >= b.strict.f1
