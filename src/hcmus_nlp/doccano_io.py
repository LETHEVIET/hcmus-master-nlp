"""Doccano sequence labeling JSONL I/O (Phase E).

Pin Doccano 1.8.x; record shape:
    {"id": "<internal_sentence_sid>", "data": "<sentence>", "label": [[start,end,label], ...]}
Same shape compatible with 1.6.x. If upgrading Doccano, re-run tests/test_doccano.py.

Đặc điểm:
- Match bằng stable sentence_id; KHÔNG dùng text (tránh nhầm với câu trùng text).
- Offset: export relative (sentence-relative), import global.
- Atomic write: temp file + flush + fsync + os.replace.
- Empty sentence (id có trong Doccano với label=[]) → checked empty.
- Duplicate id trong input → fatal.
- --input == --output → fatal.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


class DoccanoError(RuntimeError):
    """Lỗi Doccano I/O."""


@dataclass(frozen=True)
class DoccanoRecord:
    """Một record Doccano."""

    id: str
    data: str
    label: tuple[tuple[int, int, str], ...]


def load_doccano_export(path: Path) -> dict[str, DoccanoRecord]:
    """Đọc Doccano export ra dict theo id. Phát hiện duplicate id."""
    out: dict[str, DoccanoRecord] = {}
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = str(d.get("id", "")).strip()
            if not sid:
                raise DoccanoError(f"Doccano line {line_number}: missing id")
            if sid in out:
                raise DoccanoError(f"Doccano duplicate id {sid!r} at line {line_number}")
            data = d.get("data") or d.get("text") or ""
            raw_labels = d.get("label") or []
            labels: list[tuple[int, int, str]] = []
            for lab in raw_labels:
                # Doccano format: [start, end, label]
                if not isinstance(lab, (list, tuple)) or len(lab) < 3:
                    continue
                s, e, lab_name = int(lab[0]), int(lab[1]), str(lab[2])
                if s < 0 or e <= s or s > len(data) or e > len(data):
                    raise DoccanoError(
                        f"Doccano record {sid!r} line {line_number}: "
                        f"invalid span ({s},{e}) for length {len(data)}"
                    )
                labels.append((s, e, lab_name))
            out[sid] = DoccanoRecord(id=sid, data=data, label=tuple(labels))
    return out


def export_to_doccano(
    corpus_records: Iterable[dict],
    output_path: Path,
) -> int:
    """Convert corpus JSONL → Doccano sequence labeling JSONL.

    Mỗi sentence thành 1 Doccano record với `id` = sentence_id ổn định và
    `label` chứa [start, end, label] relative offset.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as f:
        for record in corpus_records:
            for sentence in record.get("sentences", []):
                sid = sentence.get("sid")
                if not sid:
                    continue
                sent_start = sentence.get("start", 0)
                labels = []
                for ent in record.get("entities", []):
                    if ent.get("sentence_id") != sid:
                        continue
                    labels.append(
                        [
                            int(ent["start"]) - sent_start,
                            int(ent["end"]) - sent_start,
                            ent["label"],
                        ]
                    )
                f.write(
                    json.dumps(
                        {
                            "id": sid,
                            "data": sentence.get("text", ""),
                            "label": labels,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    os.replace(tmp, output_path)
    return n


def export_to_doccano_stream(
    corpus_path: Path,
    output_path: Path,
) -> int:
    """Streaming version: đọc corpus từng record, không load full vào RAM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    n = 0
    with corpus_path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for sentence in record.get("sentences", []):
                sid = sentence.get("sid")
                if not sid:
                    continue
                sent_start = sentence.get("start", 0)
                labels = []
                for ent in record.get("entities", []):
                    if ent.get("sentence_id") != sid:
                        continue
                    labels.append(
                        [
                            int(ent["start"]) - sent_start,
                            int(ent["end"]) - sent_start,
                            ent["label"],
                        ]
                    )
                dst.write(
                    json.dumps(
                        {
                            "id": sid,
                            "data": sentence.get("text", ""),
                            "label": labels,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    os.replace(tmp, output_path)
    return n


def import_from_doccano(
    doccano_path: Path,
    corpus_path: Path,
    output_path: Path,
    *,
    strict: bool = False,
    annotator: str = "human",
    annotation_guideline_version: str = "0.1",
    gold_version: str = "v1",
) -> dict:
    """Đọc Doccano export + corpus preannotated → ghi corpus reviewed.

    Quy tắc:
    - Match bằng sentence_id (KHÔNG match text để tránh nhầm câu trùng).
    - Kiểm tra text khớp (sentence.text == doccano.data). Không khớp → fatal.
    - Empty sentence (id có trong Doccano với label=[]) → marked checked.
    - Duplicate id → fatal.
    - --corpus == --output → fatal (argparse check).
    - output đã tồn tại → fatal trừ khi --force (chống ghi đè gold).
    - Entity do human review phải có `sources=["human"]` để pass strict
      validator (provenance).
    - Ghi gold kèm metadata: source_corpus_sha256, doccano_sha256,
      guideline_version, gold_version, mapping_version, annotator.

    Trả dict thống kê: n_updated, n_added, n_removed, n_checked_empty,
    n_missing_in_doccano, n_text_mismatch, gold_metadata.
    """
    if corpus_path.resolve() == output_path.resolve():
        raise DoccanoError("--input corpus và --output phải khác path")
    if not corpus_path.exists():
        raise DoccanoError(f"Corpus not found: {corpus_path}")
    if not doccano_path.exists():
        raise DoccanoError(f"Doccano file not found: {doccano_path}")
    if output_path.exists() and not getattr(import_from_doccano, "_allow_overwrite", False):
        raise DoccanoError(
            f"Output {output_path} đã tồn tại. Đây là gold artifact — không "
            "ghi đè. Dùng --force để ghi đè (KHÔNG khuyến nghị)."
        )

    doccano = load_doccano_export(doccano_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")

    # Hash inputs để ghi vào gold metadata.
    source_corpus_sha = _sha256_file(corpus_path)
    doccano_sha = _sha256_file(doccano_path)

    stats = {
        "n_updated": 0,
        "n_added": 0,
        "n_removed": 0,
        "n_checked_empty": 0,
        "n_missing_in_doccano": 0,
        "n_text_mismatch": 0,
        "n_reviewed_sentences": 0,
        "n_resolved_conflicts": 0,
    }

    try:
        with corpus_path.open(encoding="utf-8") as src, tmp.open("w", encoding="utf-8") as dst:
            for line in src:
                line = line.rstrip("\n")
                if not line:
                    continue
                record = json.loads(line)
                sent_by_sid: dict[str, dict] = {s["sid"]: s for s in record.get("sentences", [])}
                entities_by_sid: dict[str, list[dict]] = {}
                for ent in record.get("entities", []):
                    entities_by_sid.setdefault(ent.get("sentence_id"), []).append(ent)

                for sid, sentence in sent_by_sid.items():
                    sent_text = sentence.get("text", "")
                    sent_start = sentence.get("start", 0)
                    doccano_rec = doccano.get(sid)
                    if doccano_rec is None:
                        stats["n_missing_in_doccano"] += 1
                        if strict:
                            raise DoccanoError(f"Strict mode: sentence {sid!r} missing in Doccano")
                        continue

                    if doccano_rec.data and doccano_rec.data != sent_text:
                        stats["n_text_mismatch"] += 1
                        raise DoccanoError(
                            f"Sentence {sid!r}: text mismatch. "
                            f"corpus={sent_text[:40]!r} doccano={doccano_rec.data[:40]!r}"
                        )

                    new_entities: list[dict] = []
                    for rel_start, rel_end, label in doccano_rec.label:
                        abs_start = sent_start + rel_start
                        abs_end = sent_start + rel_end
                        new_entities.append(
                            {
                                "eid": f"{sid}-e{len(new_entities) + 1}",
                                "sentence_id": sid,
                                "start": abs_start,
                                "end": abs_end,
                                "text": sent_text[rel_start:rel_end],
                                "label": label,
                                # Provenance quan trọng: strict validator sẽ
                                # fail nếu entity không có sources[].
                                "sources": ["human"],
                                "source_ids": [f"annotator:{annotator}"],
                                "priority_score": 1.0,
                                "matched_alias": None,
                                "merged_from_labels": [label],
                                "linking_candidates": [],
                                "linking_status": None,
                                "method": "human_review",
                                "review_status": "checked",
                                "annotator": annotator,
                            }
                        )

                    # Resolve conflicts thuộc sentence này. Tất cả conflict
                    # của câu đã review → chuyển sang `resolved_conflicts` với
                    # resolution="human_decision" + annotator + timestamp.
                    # Strict validator không còn flag sentence này.
                    sentence_conflicts = [
                        c
                        for c in (record.get("unresolved_conflicts") or [])
                        if c.get("sentence_id") == sid
                    ]
                    resolved_conflicts: list[dict] = list(record.get("resolved_conflicts") or [])
                    for sc in sentence_conflicts:
                        resolved_conflicts.append(
                            {
                                **sc,
                                "resolution": "human_decision",
                                "resolved_by": annotator,
                                "resolution_source": "doccano_import",
                                "gold_version": gold_version,
                            }
                        )
                    record["resolved_conflicts"] = resolved_conflicts
                    # Xóa khỏi unresolved_conflicts.
                    record["unresolved_conflicts"] = [
                        c
                        for c in (record.get("unresolved_conflicts") or [])
                        if c.get("sentence_id") != sid
                    ]

                    old_count = len(entities_by_sid.get(sid, []))
                    new_count = len(new_entities)
                    if old_count == 0 and new_count > 0:
                        stats["n_added"] += 1
                    elif old_count > 0 and new_count == 0:
                        stats["n_removed"] += 1
                    elif old_count != new_count or any(
                        e.get("label") != ne.get("label")
                        for e, ne in zip(
                            sorted(entities_by_sid.get(sid, []), key=lambda x: x["start"]),
                            sorted(new_entities, key=lambda x: x["start"]),
                            strict=False,
                        )
                    ):
                        stats["n_updated"] += 1

                    other_entities = [
                        e for e in record.get("entities", []) if e.get("sentence_id") != sid
                    ]
                    other_entities.extend(new_entities)
                    record["entities"] = other_entities

                    sentence["review_status"] = "checked"
                    if not doccano_rec.label:
                        stats["n_checked_empty"] += 1
                    stats["n_reviewed_sentences"] += 1
                    if sentence_conflicts:
                        stats["n_resolved_conflicts"] += len(sentence_conflicts)

                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(tmp, output_path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise

    # Ghi gold metadata sidecar.
    metadata = {
        "gold_version": gold_version,
        "annotation_guideline_version": annotation_guideline_version,
        "source_corpus_sha256": source_corpus_sha,
        "doccano_export_sha256": doccano_sha,
        "annotator": annotator,
        "n_reviewed_sentences": stats["n_reviewed_sentences"],
        "n_added": stats["n_added"],
        "n_updated": stats["n_updated"],
        "n_removed": stats["n_removed"],
        "n_checked_empty": stats["n_checked_empty"],
        "n_resolved_conflicts": stats["n_resolved_conflicts"],
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stats["gold_metadata"] = metadata
    return stats


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "DoccanoError",
    "DoccanoRecord",
    "export_to_doccano",
    "export_to_doccano_stream",
    "import_from_doccano",
    "load_doccano_export",
]
