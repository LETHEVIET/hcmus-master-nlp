"""Strict internal validator (Phase H1.1) + Compliance checker (Phase H1.2).

Hai tầng validation, tách trách nhiệm:
- validate_corpus: kiểm tra source corpus nội bộ (review_status, cleaning,
  unresolved, provenance, mapping). Output JSON report deterministic.
- compliance_check: kiểm tra submission artifact xuất ra (shape, label
  membership, ID unique, no overlap, folder/file pairing, manifest hash).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValidationIssue:
    severity: str  # "fatal" | "warning"
    code: str
    message: str
    sentence_id: str | None = None
    record_id: str | None = None


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]
    n_records: int = 0
    n_sentences: int = 0
    n_entities: int = 0

    @property
    def fatal_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "fatal")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def to_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "n_sentences": self.n_sentences,
            "n_entities": self.n_entities,
            "fatal_count": self.fatal_count,
            "warning_count": self.warning_count,
            "issues": [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "sentence_id": i.sentence_id,
                    "record_id": i.record_id,
                }
                for i in self.issues
            ],
        }

    def to_json(self) -> str:
        """Serialize thành JSON string (sort_keys=True để deterministic)."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def write(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def validate_corpus_strict(
    corpus_path: Path,
    *,
    scope: str = "full",
    mapping,
    pilot_sentence_ids: set[str] | None = None,
) -> ValidationReport:
    """Strict validator cho source corpus nội bộ.

    Check (fatal ở final):
    - mapping confirmed khi scope dùng cho final.
    - sentence trong scope: review_status == checked.
    - cleaning_status ∈ {kept, checked} cho record có ít nhất 1 sentence trong scope.
    - không có unresolved_conflicts cho record có ít nhất 1 sentence trong scope.
    - entity span hợp lệ, flat, không overlap.
    - entity.text khớp record.text[start:end].
    - entity nằm trong sentence tương ứng.
    - internal label ∈ INTERNAL_LABELS (mapping đầy đủ).
    - 100% entity có provenance (sources[]).
    """
    issues: list[ValidationIssue] = []
    n_records = n_sentences = n_entities = 0

    if scope == "final" and not mapping.is_confirmed():
        issues.append(
            ValidationIssue(
                severity="fatal",
                code="mapping_unconfirmed",
                message="Mapping.confirmed=false; final mode cần confirmed=true.",
            )
        )

    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_records += 1
            record_id = r.get("id")
            text = r.get("text", "")

            sentences = r.get("sentences", [])
            sentence_by_id = {s.get("sid"): s for s in sentences if s.get("sid")}

            # Scope filter cho record-level checks.
            in_scope = False
            if scope == "full":
                in_scope = True
            elif scope == "pilot" and pilot_sentence_ids:
                in_scope = any(s.get("sid") in pilot_sentence_ids for s in sentences)

            for sent in sentences:
                n_sentences += 1
                sid = sent.get("sid")
                if scope == "pilot" and pilot_sentence_ids and sid not in pilot_sentence_ids:
                    continue
                if sent.get("review_status") != "checked":
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="unchecked_sentence",
                            message=f"sentence review_status={sent.get('review_status')!r}, expected 'checked'",
                            sentence_id=sid,
                            record_id=record_id,
                        )
                    )

            # Record-level checks chỉ áp dụng nếu record có câu trong scope.
            if in_scope:
                cleaning = r.get("cleaning_status")
                if cleaning not in {"kept", "checked"}:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="cleaning_needs_review",
                            message=f"record cleaning_status={cleaning!r}",
                            record_id=record_id,
                        )
                    )

                conflicts = r.get("unresolved_conflicts") or []
                if conflicts:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="unresolved_conflicts",
                            message=f"record có {len(conflicts)} unresolved_conflict(s)",
                            record_id=record_id,
                        )
                    )

            # Entity invariants.
            ents = r.get("entities", [])
            spans: list[tuple[int, int, str, str]] = []
            text_len = len(text)
            for ent in ents:
                if (
                    scope == "pilot"
                    and pilot_sentence_ids
                    and ent.get("sentence_id") not in pilot_sentence_ids
                ):
                    continue
                n_entities += 1
                s = ent.get("start")
                e = ent.get("end")
                if (
                    not isinstance(s, int)
                    or not isinstance(e, int)
                    or s < 0
                    or e <= s
                    or e > text_len
                ):
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="invalid_entity_span",
                            message=f"entity span invalid ({s},{e}) for text length {text_len}",
                            record_id=record_id,
                            sentence_id=ent.get("sentence_id"),
                        )
                    )
                    continue
                # entity.text khớp text[start:end]
                actual_text = text[s:e]
                if ent.get("text") != actual_text:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="entity_text_mismatch",
                            message=f"entity.text={ent.get('text')!r} không khớp text[{s}:{e}]={actual_text!r}",
                            record_id=record_id,
                            sentence_id=ent.get("sentence_id"),
                        )
                    )
                # entity nằm trong sentence tương ứng
                sid = ent.get("sentence_id")
                sent = sentence_by_id.get(sid)
                if sent is None:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="entity_orphan_sentence",
                            message=f"entity sentence_id={sid!r} không có trong record",
                            record_id=record_id,
                            sentence_id=sid,
                        )
                    )
                elif not (sent["start"] <= s and e <= sent["end"]):
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="entity_outside_sentence",
                            message=f"entity ({s},{e}) nằm ngoài sentence {sid!r} ({sent['start']},{sent['end']})",
                            record_id=record_id,
                            sentence_id=sid,
                        )
                    )

                # internal label ∈ mapping
                from hcmus_nlp.labels import INTERNAL_LABELS

                label = ent.get("label")
                if label not in INTERNAL_LABELS:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="invalid_internal_label",
                            message=f"entity label={label!r} không thuộc INTERNAL_LABELS",
                            record_id=record_id,
                            sentence_id=ent.get("sentence_id"),
                        )
                    )

                if ent.get("review_status") != "checked":
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="unchecked_entity",
                            message=f"entity review_status={ent.get('review_status')!r}",
                            record_id=record_id,
                            sentence_id=ent.get("sentence_id"),
                        )
                    )
                sources = ent.get("sources")
                if not sources:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="missing_provenance",
                            message="entity không có sources[]",
                            record_id=record_id,
                            sentence_id=ent.get("sentence_id"),
                        )
                    )
                spans.append((s, e, ent.get("label", ""), ent.get("sentence_id", "")))

            # Overlap check.
            spans.sort()
            for i in range(len(spans) - 1):
                s1, e1, l1, sid1 = spans[i]
                s2, e2, l2, sid2 = spans[i + 1]
                if sid1 == sid2 and s2 < e1:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="entity_overlap",
                            message=f"entity overlap ({s1},{e1},{l1}) vs ({s2},{e2},{l2})",
                            record_id=record_id,
                            sentence_id=sid1,
                        )
                    )

    return ValidationReport(
        issues=tuple(issues),
        n_records=n_records,
        n_sentences=n_sentences,
        n_entities=n_entities,
    )


def compliance_check(
    submission_dir: Path,
    *,
    expected_sentences: int | None = None,
    source_corpus_sha256: str | None = None,
    source_validation_report_sha256: str | None = None,
    mode: str = "draft",
) -> ValidationReport:
    """Validate submission artifact. Strict — fatal trên mọi vi phạm.

    Check:
    - manifest.json tồn tại, parse được.
    - Mỗi folder HCH_NNN có HCH_NNN_seg.tsv + HCH_NNN_ner.json đúng cặp.
    - sentence_id unique trong submission.
    - Không overlap entity trong cùng sentence.
    - Label ∈ SUBMISSION_LABELS.
    - expected_sentences khớp manifest (nếu truyền).
    - source_corpus_sha256 khớp manifest (nếu truyền).
    - Entity object đúng minimal shape khi mode=final: chỉ {text, label,
      start, end}. Nếu có field thừa → fatal (không phải warning như trước).
    - Tất cả entity đều được kiểm tra (không chỉ entity đầu).
    """
    issues: list[ValidationIssue] = []

    manifest_path = submission_dir / "manifest.json"
    if not manifest_path.exists():
        issues.append(
            ValidationIssue(
                severity="fatal",
                code="missing_manifest",
                message="manifest.json không tồn tại",
            )
        )
        return ValidationReport(issues=tuple(issues))

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        issues.append(
            ValidationIssue(
                severity="fatal",
                code="invalid_manifest_json",
                message=str(e),
            )
        )
        return ValidationReport(issues=tuple(issues))

    # expected_sentences check.
    if expected_sentences is not None:
        actual = manifest.get("sentences")
        if actual != expected_sentences:
            issues.append(
                ValidationIssue(
                    severity="fatal",
                    code="sentence_count_mismatch",
                    message=f"manifest.sentences={actual}, expected {expected_sentences}",
                )
            )

    if source_corpus_sha256 and manifest.get("source_corpus_sha256") != source_corpus_sha256:
        issues.append(
            ValidationIssue(
                severity="fatal",
                code="source_corpus_hash_mismatch",
                message="source_corpus_sha256 không khớp",
            )
        )

    if (
        source_validation_report_sha256
        and manifest.get("source_validation_report_sha256") != source_validation_report_sha256
    ):
        issues.append(
            ValidationIssue(
                severity="fatal",
                code="source_report_hash_mismatch",
                message="source_validation_report_sha256 không khớp",
            )
        )

    # Folder pairing + ID unique.
    seen_sentence_ids: set[str] = set()
    n_entities = 0
    from hcmus_nlp.labels import SUBMISSION_LABELS

    for work_dir in sorted(submission_dir.iterdir()):
        if not work_dir.is_dir() or not work_dir.name.startswith("HCH_"):
            continue
        for vol_dir in sorted(work_dir.iterdir()):
            if not vol_dir.is_dir():
                continue
            base = vol_dir.name
            seg_path = vol_dir / f"{base}_seg.tsv"
            ner_path = vol_dir / f"{base}_ner.json"
            if not seg_path.exists() or not ner_path.exists():
                issues.append(
                    ValidationIssue(
                        severity="fatal",
                        code="folder_file_pairing",
                        message=f"{vol_dir}: thiếu seg/ner file",
                    )
                )
                continue
            # Parse _seg.tsv.
            seg_ids: list[str] = []
            with seg_path.open(encoding="utf-8") as f:
                for line in f:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2:
                        seg_ids.append(parts[0])
            # Parse _ner.json.
            try:
                ner_data = json.loads(ner_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                issues.append(
                    ValidationIssue(
                        severity="fatal",
                        code="invalid_ner_json",
                        message=f"{ner_path}: {e}",
                    )
                )
                continue
            ner_ids = [r.get("sentence_id") for r in ner_data]
            if seg_ids != ner_ids:
                issues.append(
                    ValidationIssue(
                        severity="fatal",
                        code="seg_ner_mismatch",
                        message=f"{base}: sentence_id không khớp seg/ner",
                    )
                )
            for sid in ner_ids:
                if sid in seen_sentence_ids:
                    issues.append(
                        ValidationIssue(
                            severity="fatal",
                            code="duplicate_sentence_id",
                            message=f"sentence_id trùng: {sid}",
                            sentence_id=sid,
                        )
                    )
                seen_sentence_ids.add(sid)
            # Per-sentence check.
            for rec in ner_data:
                sid = rec.get("sentence_id")
                entities = rec.get("entities", [])
                seen_spans: list[tuple[int, int, str]] = []
                for ent in entities:
                    n_entities += 1
                    label = ent.get("label")
                    if label not in SUBMISSION_LABELS:
                        issues.append(
                            ValidationIssue(
                                severity="fatal",
                                code="invalid_label",
                                message=f"{sid}: label {label!r} not in SUBMISSION_LABELS",
                                sentence_id=sid,
                            )
                        )
                    s = ent.get("start")
                    e = ent.get("end")
                    if not isinstance(s, int) or not isinstance(e, int) or s < 0 or e <= s:
                        issues.append(
                            ValidationIssue(
                                severity="fatal",
                                code="invalid_entity_span",
                                message=f"{sid}: span invalid ({s},{e})",
                                sentence_id=sid,
                            )
                        )
                        continue
                    for ps, pe, plabel in seen_spans:
                        if s < pe and e > ps:
                            issues.append(
                                ValidationIssue(
                                    severity="fatal",
                                    code="entity_overlap",
                                    message=f"{sid}: entity overlap ({s},{e},{label}) vs ({ps},{pe},{plabel})",
                                    sentence_id=sid,
                                )
                            )
                    seen_spans.append((s, e, label))
                    # Minimal shape ở final: kiểm MỌI entity (không chỉ đầu).
                    if mode == "final":
                        allowed = {"text", "label", "start", "end"}
                        extra_keys = set(ent.keys()) - allowed
                        if extra_keys:
                            issues.append(
                                ValidationIssue(
                                    severity="fatal",
                                    code="extra_provenance_fields",
                                    message=(
                                        f"{sid}: entity có field thừa ở mode=final: "
                                        f"{sorted(extra_keys)}"
                                    ),
                                    sentence_id=sid,
                                )
                            )
                        # Required keys.
                        missing = allowed - set(ent.keys())
                        if missing:
                            issues.append(
                                ValidationIssue(
                                    severity="fatal",
                                    code="missing_required_field",
                                    message=f"{sid}: entity thiếu field {sorted(missing)}",
                                    sentence_id=sid,
                                )
                            )

    n_records = len([d for d in submission_dir.iterdir() if d.is_dir()])
    return ValidationReport(
        issues=tuple(issues),
        n_records=n_records,
        n_sentences=len(seen_sentence_ids),
        n_entities=n_entities,
    )


__all__ = [
    "ValidationIssue",
    "ValidationReport",
    "compliance_check",
    "validate_corpus_strict",
]
