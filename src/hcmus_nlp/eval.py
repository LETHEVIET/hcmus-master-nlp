"""Evaluation framework (Phase D2).

Metric:
- entity_strict_precision/recall/f1: (start, end, label) exact match.
- boundary_only_f1: (start, end) match, bỏ qua label.
  - Boundary khớp + label sai → TP cho boundary-only + lỗi strict entity.
- per_label_f1: per label.
- label_accuracy_on_matched_boundaries: trong prediction có boundary khớp gold,
  tỷ lệ label đúng.

Đầu vào: list[dict] entities, mỗi dict có `start`, `end`, `label`, optional
`sentence_id`. So sánh theo sentence_id để tránh trùng entity từ sentence
khác nhau.

Token-level mode optional (`--token-level`): chuyển span sang BIO và dùng
seqeval nếu có.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Entity:
    start: int
    end: int
    label: str
    sentence_id: str | None = None


@dataclass(frozen=True)
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class EvalReport:
    """Báo cáo đầy đủ cho một tập."""

    strict: Metrics
    boundary_only: Metrics
    per_label: dict[str, Metrics] = field(default_factory=dict)
    label_accuracy_on_matched_boundaries: float = 0.0
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    n_gold: int = 0
    n_pred: int = 0

    def to_dict(self) -> dict:
        return {
            "strict": _metrics_to_dict(self.strict),
            "boundary_only": _metrics_to_dict(self.boundary_only),
            "per_label": {k: _metrics_to_dict(v) for k, v in self.per_label.items()},
            "label_accuracy_on_matched_boundaries": self.label_accuracy_on_matched_boundaries,
            "confusion": {f"{k[0]}|{k[1]}": v for k, v in self.confusion.items()},
            "n_gold": self.n_gold,
            "n_pred": self.n_pred,
        }


def _metrics_to_dict(m: Metrics) -> dict:
    return {
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
    }


def _prf(tp: int, fp: int, fn: int) -> Metrics:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return Metrics(precision=p, recall=r, f1=f1, tp=tp, fp=fp, fn=fn)


def _entities_to_tuples(entities: Iterable[dict | Entity]) -> set[tuple]:
    """Convert entities thành set tuple cho so sánh exact.

    Mỗi tuple: (sentence_id, start, end, label). sentence_id="global" nếu
    không có, để so sánh cross-sentence an toàn.
    """
    out: set[tuple] = set()
    for e in entities:
        if isinstance(e, Entity):
            sid = e.sentence_id or "global"
            out.add((sid, e.start, e.end, e.label))
        else:
            sid = e.get("sentence_id") or "global"
            out.add((sid, int(e["start"]), int(e["end"]), str(e["label"])))
    return out


def _boundary_tuples(entities: Iterable[dict | Entity]) -> set[tuple]:
    out: set[tuple] = set()
    for e in entities:
        if isinstance(e, Entity):
            sid = e.sentence_id or "global"
            out.add((sid, e.start, e.end))
        else:
            sid = e.get("sentence_id") or "global"
            out.add((sid, int(e["start"]), int(e["end"])))
    return out


def evaluate(
    gold: list[dict | Entity],
    pred: list[dict | Entity],
) -> EvalReport:
    """Tính đầy đủ metric từ gold + pred.

    Args:
        gold: list entity vàng (đã review). Mỗi item có start/end/label/
            sentence_id.
        pred: list entity dự đoán. Cùng shape.
    """
    gold_tuples = _entities_to_tuples(gold)
    pred_tuples = _entities_to_tuples(pred)
    gold_b = _boundary_tuples(gold)
    pred_b = _boundary_tuples(pred)

    # Strict match.
    tp = len(gold_tuples & pred_tuples)
    fp = len(pred_tuples - gold_tuples)
    fn = len(gold_tuples - pred_tuples)
    strict = _prf(tp, fp, fn)

    # Boundary-only.
    tp_b = len(gold_b & pred_b)
    fp_b = len(pred_b - gold_b)
    fn_b = len(gold_b - pred_b)
    boundary = _prf(tp_b, fp_b, fn_b)

    # Per-label.
    per_label: dict[str, Metrics] = {}
    labels = {e[3] for e in gold_tuples} | {e[3] for e in pred_tuples}
    for label in labels:
        g_l = {t for t in gold_tuples if t[3] == label}
        p_l = {t for t in pred_tuples if t[3] == label}
        tp_l = len(g_l & p_l)
        fp_l = len(p_l - g_l)
        fn_l = len(g_l - p_l)
        per_label[label] = _prf(tp_l, fp_l, fn_l)

    # Label accuracy on matched boundaries.
    matched = gold_b & pred_b
    if matched:
        # Build gold label lookup by (sid, start, end).
        gold_label_by_b = {(t[0], t[1], t[2]): t[3] for t in gold_tuples}
        pred_label_by_b = {(t[0], t[1], t[2]): t[3] for t in pred_tuples}
        correct = sum(1 for b in matched if gold_label_by_b.get(b) == pred_label_by_b.get(b))
        label_acc = correct / len(matched)
    else:
        label_acc = 0.0

    # Confusion matrix: predicted × gold (chỉ cho matched boundary, để dễ đọc).
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for b in matched:
        g_label = {(t[0], t[1], t[2]): t[3] for t in gold_tuples}.get(b)
        p_label = {(t[0], t[1], t[2]): t[3] for t in pred_tuples}.get(b)
        if g_label and p_label:
            confusion[(p_label, g_label)] += 1

    return EvalReport(
        strict=strict,
        boundary_only=boundary,
        per_label=per_label,
        label_accuracy_on_matched_boundaries=label_acc,
        confusion=dict(confusion),
        n_gold=len(gold_tuples),
        n_pred=len(pred_tuples),
    )


def write_report(report: EvalReport, output_path) -> None:
    """Ghi report ra JSON."""
    payload = report.to_dict()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "Entity",
    "EvalReport",
    "Metrics",
    "evaluate",
    "write_report",
]
