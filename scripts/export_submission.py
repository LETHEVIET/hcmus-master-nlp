#!/usr/bin/env python3
"""Export annotated corpus theo format đề tài yêu cầu (per-volume).

Phase B3+B3.4+H1 (plan v5): 3 mode rõ ràng:
- draft: cho `needs_review` qua; format check; không phải submission thật.
- pilot: chỉ lấy sentence trong pilot file; cho `needs_review` qua.
- final: submission thật — fatal nếu còn unchecked, unresolved, mapping
  chưa confirmed, hoặc cleaning_status=needs_review.

Pre-validation (H1.1): ở --mode final, gọi `validate_corpus_strict` lên source
corpus TRƯỚC khi tạo bất kỳ folder output nào. Nếu có fatal, raise mà không
để lại submission một phần.

Atomic: mỗi volume ghi vào temp dir + flush + fsync + rename. Nếu lỗi giữa
chừng, output cũ vẫn còn (không mất) và không có folder lửa lở.

Submission shape mặc định (B3.4): chỉ ghi `{text, label, start, end}`. Provenance
(`sources`, `priority_score`, `review_status`, `merged_from_labels`,
`linking_candidates`) chỉ xuất khi `--include-provenance` (off mặc định).

Manifest ghi `source_corpus_sha256`, `source_validation_report_sha256`,
`mapping_version`, `mapping_confirmed`, `built_at` (từ SOURCE_DATE_EPOCH nếu
có, để deterministic).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

# Bootstrap cho direct-script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.labels import (  # noqa: E402
    SUBMISSION_LABELS,
    MappingError,
    load_mapping,
)
from hcmus_nlp.validation import validate_corpus_strict  # noqa: E402

WORK_IDS = {
    "諸蕃志": "HCH_001",
    "東觀漢記_(四庫全書本)": "HCH_002",
    "北史": "HCH_003",
    "北齊書": "HCH_004",
    "後漢書": "HCH_005",
    "漢書": "HCH_006",
    "舊五代史": "HCH_007",
    "舊唐書": "HCH_008",
}

REQUIRED_LABELS = ["PER", "LOC", "ORG", "TITLE", "TME", "NUM"]


class ExportError(RuntimeError):
    """Lỗi nghiêm trọng khi xuất submission (đặc biệt ở --mode final)."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _deterministic_built_at() -> str:
    """Trả timestamp từ SOURCE_DATE_EPOCH (chuẩn reproducible-build)."""
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch is None:
        return ""
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
    except (ValueError, OSError):
        return ""


def _load_pilot_sentence_ids(pilot_path: Path) -> set[str]:
    if not pilot_path.exists():
        raise ExportError(f"Pilot file not found: {pilot_path}")
    ids: set[str] = set()
    with pilot_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            sid = d.get("sentence_id") or d.get("sid")
            if not sid:
                raise ExportError(f"Pilot record missing sentence_id: {d}")
            ids.add(sid)
    return ids


def _to_minimal_entity(entity: dict, sentence_start: int, mapping_to_sub: dict) -> dict | None:
    """Convert entity nội bộ sang minimal submission shape.

    Trả None nếu internal label không có target trong mapping (lỗi cấu hình).
    """
    label_internal = entity["label"]
    label = mapping_to_sub.get(label_internal)
    if label is None:
        # Internal label không có mapping → fatal (không silently skip).
        raise ExportError(
            f"Entity {entity.get('eid', '?')!r}: internal label {label_internal!r} "
            f"không có mapping → submission. Cập nhật config/mapping.toml."
        )
    if label not in SUBMISSION_LABELS:
        raise ExportError(f"Mapping target {label!r} không thuộc SUBMISSION_LABELS")
    return {
        "text": entity["text"],
        "label": label,
        "start": entity["start"] - sentence_start,
        "end": entity["end"] - sentence_start,
    }


def _to_extended_entity(entity: dict, sentence_start: int, mapping_to_sub: dict) -> dict | None:
    minimal = _to_minimal_entity(entity, sentence_start, mapping_to_sub)
    if minimal is None:
        return None
    extra = {}
    for key in (
        "sources",
        "priority_score",
        "review_status",
        "merged_from_labels",
        "linking_candidates",
        "linking_status",
    ):
        if key in entity:
            extra[key] = entity[key]
    return {**minimal, **extra}


def volume_number(value: str | None) -> str:
    """Output id cho folder volume. Single source of truth: `record["volume_id"]`."""
    if not value:
        return "00"
    if not value.startswith("卷"):
        if re.fullmatch(r"\d+[abc]?", value):
            return value
    digits = re.search(r"[0-9０-９]+", value)
    if digits:
        return str(int(digits.group(0))).zfill(2)
    return value.replace("卷", "").replace("上", "a").replace("中", "b").replace("下", "c")


def _check_entity_invariants(entities: list[dict], sentence_id: str) -> list[str]:
    errors: list[str] = []
    seen: list[tuple[int, int, str]] = []
    for ent in entities:
        label = ent.get("label")
        if label not in SUBMISSION_LABELS:
            errors.append(f"{sentence_id}: entity label {label!r} not in SUBMISSION_LABELS")
        s, e = ent.get("start"), ent.get("end")
        if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e <= s:
            errors.append(f"{sentence_id}: entity span invalid ({s}, {e})")
            continue
        for ps, pe, plabel in seen:
            if s < pe and e > ps:
                errors.append(
                    f"{sentence_id}: entity overlap ({s},{e},{label}) vs ({ps},{pe},{plabel})"
                )
        seen.append((s, e, label))
    return errors


def export(
    input_path: Path,
    output_dir: Path,
    *,
    mode: str,
    mapping,
    pilot_sentence_ids: set[str] | None,
    include_provenance: bool = False,
    scope: str = "full",
) -> dict:
    if mode not in {"draft", "pilot", "final"}:
        raise ExportError(f"Unknown mode: {mode!r}")
    if mode == "final" and not mapping.is_confirmed():
        raise ExportError(
            "Mapping.confirmed=false; cannot export --mode final. "
            "Đổi `confirmed = true` trong config/mapping.toml sau khi giảng "
            "viên duyệt mapping."
        )
    if mode == "pilot" and not pilot_sentence_ids:
        raise ExportError("--mode pilot yêu cầu --pilot PATH")

    # Pre-validation cho --mode final.
    validation_report = None
    validation_report_sha: str | None = None
    if mode == "final":
        # Pilot scope dùng pilot_sentence_ids; full scope dùng toàn bộ.
        vscope = "final" if scope == "full" else "pilot"
        validation_report = validate_corpus_strict(
            input_path,
            scope=vscope,
            mapping=mapping,
            pilot_sentence_ids=pilot_sentence_ids,
        )
        if validation_report.fatal_count > 0:
            codes = Counter(i.code for i in validation_report.issues)
            raise ExportError(
                f"--mode final pre-validation failed: {validation_report.fatal_count} "
                f"fatal issues. Top codes: {dict(codes.most_common(5))}. "
                "Sửa source corpus trước khi export."
            )
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            work_id = WORK_IDS.get(record["title"])
            if work_id is None:
                raise ExportError(f"No work id configured for {record['title']!r}")
            groups[(work_id, volume_number(record.get("volume")))].append(record)

    total_sentences = 0
    total_entities = 0
    excluded_unchecked = 0
    excluded_unresolved = 0
    excluded_unknown_label = 0
    fatal_errors: list[str] = []

    # Atomic write: ghi vào temp dir trong cùng parent (cùng filesystem),
    # rename khi xong. `os.rename` trên cùng filesystem là atomic — không có
    # khoảnh khắc nào submission_dir tồn tại "nửa nạc nửa mỡ".
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_dir.parent
    with tempfile.TemporaryDirectory(
        dir=staging_root, prefix=f".{output_dir.name}.staging."
    ) as staging:
        staging_path = Path(staging) / output_dir.name
        # Tạo thư mục gốc của staging; mkdir parents=True cho work/vol bên dưới.
        staging_path.mkdir(parents=True, exist_ok=True)
        for (work_id, volume), records in sorted(groups.items()):
            base = f"{work_id}_{volume}"
            volume_dir = staging_path / work_id / base
            volume_dir.mkdir(parents=True, exist_ok=True)
            seg_path = volume_dir / f"{base}_seg.tsv"
            ner_path = volume_dir / f"{base}_ner.json"
            ner_records: list[dict] = []
            sentence_number = 0

            with seg_path.open("w", encoding="utf-8") as seg:
                for record in records:
                    for sentence in record.get("sentences", []):
                        sentence_number += 1
                        sentence_id = f"{base}_{sentence_number:06d}"

                        if mode == "pilot" and sentence["sid"] not in pilot_sentence_ids:
                            continue
                        if scope == "pilot" and mode == "final":
                            if sentence["sid"] not in (pilot_sentence_ids or set()):
                                continue

                        if mode == "final":
                            if sentence.get("review_status") != "checked":
                                excluded_unchecked += 1
                                continue

                        entities: list[dict] = []
                        for entity in record.get("entities", []):
                            if entity.get("sentence_id") != sentence["sid"]:
                                continue
                            if mode == "final" and entity.get("review_status") != "checked":
                                excluded_unchecked += 1
                                continue
                            if mode == "final" and entity.get("unresolved"):
                                excluded_unresolved += 1
                                continue
                            try:
                                if include_provenance:
                                    ent = _to_extended_entity(
                                        entity,
                                        sentence["start"],
                                        mapping.internal_to_submission,
                                    )
                                else:
                                    ent = _to_minimal_entity(
                                        entity,
                                        sentence["start"],
                                        mapping.internal_to_submission,
                                    )
                            except ExportError as e:
                                if mode == "final":
                                    fatal_errors.append(str(e))
                                    excluded_unknown_label += 1
                                    continue
                                raise
                            if ent is not None:
                                entities.append(ent)

                        if mode == "final":
                            errs = _check_entity_invariants(entities, sentence_id)
                            if errs:
                                fatal_errors.extend([f"{base}: {e}" for e in errs[:3]])
                                continue

                        seg.write(f"{sentence_id}\t{sentence['text']}\n")
                        ner_records.append(
                            {
                                "sentence_id": sentence_id,
                                "sentence": sentence["text"],
                                "entities": entities,
                            }
                        )

            ner_path.write_text(
                json.dumps(ner_records, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            total_sentences += len(ner_records)
            total_entities += sum(len(item["entities"]) for item in ner_records)

        if mode == "final" and fatal_errors:
            # KHÔNG rename staging → output_dir. Temp sẽ tự cleanup khi exit context.
            raise ExportError(
                f"--mode final failed with {len(fatal_errors)} errors. First 5:\n"
                + "\n".join(fatal_errors[:5])
            )

        # Final validation report thuộc cùng staging lifecycle với submission.
        # Ghi trực tiếp vào staging để không tạo tempfile/file descriptor bên
        # ngoài atomic artifact.
        if validation_report is not None:
            validation_report_path = staging_path / "validation_report.json"
            validation_report.write(validation_report_path)
            validation_report_sha = _sha256(validation_report_path)

        # Manifest deterministic.
        built_at = _deterministic_built_at()
        manifest = {
            "scope": scope,
            "mode": mode,
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "output": str(output_dir),
            "works": len({key[0] for key in groups}),
            "volumes": len(groups),
            "sentences": total_sentences,
            "entities": total_entities,
            "excluded_sentences": {
                "unchecked": excluded_unchecked,
                "unresolved": excluded_unresolved,
                "unknown_label": excluded_unknown_label,
            },
            "input_type": "text",
            "raw_files": "not generated; raw text is preserved under dataset/",
            "label_schema": REQUIRED_LABELS,
            "annotation_status": (
                "preannotation_needs_review"
                if mode == "draft"
                else "pilot_review"
                if mode == "pilot"
                else "gold_checked"
            ),
            "mapping_version": mapping.version,
            "mapping_confirmed": mapping.is_confirmed(),
            "include_provenance": include_provenance,
            "submission_shape": "minimal" if not include_provenance else "extended",
            "source_corpus_sha256": _sha256(input_path),
            "source_validation_report_sha256": validation_report_sha,
        }
        if built_at:
            manifest["built_at"] = built_at

        (staging_path / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        # Atomic swap (rename).
        # Nếu output_dir đã tồn tại: rename thành .bak.<ts> rồi rename
        # staging sang output_dir. Cả 2 rename đều atomic trong cùng FS;
        # cuối cùng rename lại .bak thành output_dir nếu lỗi.
        if output_dir.exists():
            # Dùng os.replace cho atomic exchange khi cùng filesystem.
            backup = output_dir.with_suffix(
                output_dir.suffix + f".bak.{int.from_bytes(os.urandom(4), 'big'):08x}"
            )
            os.replace(output_dir, backup)
            try:
                os.replace(staging_path, output_dir)
            except Exception:
                # Restore backup nếu rename thất bại.
                if backup.exists() and not output_dir.exists():
                    os.replace(backup, output_dir)
                raise
            # Cleanup backup.
            if backup.exists():
                shutil.rmtree(backup)
        else:
            os.rename(staging_path, output_dir)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/corpus_preannotated.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("build/submission"))
    parser.add_argument(
        "--mode",
        choices=["draft", "pilot", "final"],
        default="draft",
        help="draft: format check; pilot: pilot subset; final: submission thật.",
    )
    parser.add_argument("--scope", choices=["full", "pilot"], default="full")
    parser.add_argument(
        "--pilot", type=Path, help="Pilot JSONL (cho --mode pilot hoặc --scope pilot)"
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/mapping.toml"),
        help="Path mapping TOML.",
    )
    parser.add_argument(
        "--include-provenance",
        action="store_true",
        help="Ghi thêm sources/priority_score/review_status vào entity.",
    )
    args = parser.parse_args()

    try:
        mapping = load_mapping(args.mapping)
    except MappingError as e:
        raise SystemExit(f"Mapping error: {e}")

    pilot_ids: set[str] | None = None
    if args.mode == "pilot" or args.scope == "pilot":
        if not args.pilot:
            raise SystemExit("--mode pilot hoặc --scope pilot yêu cầu --pilot PATH")
        pilot_ids = _load_pilot_sentence_ids(args.pilot)

    try:
        manifest = export(
            args.input,
            args.output,
            mode=args.mode,
            mapping=mapping,
            pilot_sentence_ids=pilot_ids,
            include_provenance=args.include_provenance,
            scope=args.scope,
        )
    except ExportError as e:
        raise SystemExit(f"Export error: {e}")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
