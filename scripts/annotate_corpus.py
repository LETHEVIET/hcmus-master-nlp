#!/usr/bin/env python3
"""Add sentence boundaries and reviewable NER candidates to corpus JSONL.

Phase C5 (plan v5): pipeline dùng `RegexSource` adapter + `CandidateMerger`
để sinh entity. Mỗi entity có đầy đủ provenance:
    sources: [adapter name, ...]
    priority_score: float
    merged_from_labels: [...]
    review_status: needs_review

Output:
- build/corpus_preannotated.jsonl (mặc định) — output production theo plan.
- Tên `corpus_annotated.jsonl` đã được thay bằng `corpus_preannotated.jsonl`
  vì entity giờ kèm provenance (sources/priority_score) từ CandidateMerger,
  không chỉ là heuristic đơn thuần.

Sentence segmentation giữ rule-based cũ (src/hcmus_nlp/sentence.segment).
Nguồn mặc định: RegexSource + SeedSource + CBDBPersonSource khi các cache tương
ứng tồn tại. CBDB không được tải lúc annotate; người chạy ingest một bản SQLite
đã pin trước bằng `scripts/build_kb.py ingest-cbdb`.

KHÔNG ghi đè build/gold/. Mọi write đều atomic (temp + os.replace).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# Bootstrap cho direct-script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.candidates import CandidateMerger  # noqa: E402
from hcmus_nlp.labels import MappingError, load_mapping  # noqa: E402
from hcmus_nlp.sentence import segment  # noqa: E402
from hcmus_nlp.source_base import (  # noqa: E402
    AnnotationContext,
    Candidate,
    Span,
)
from hcmus_nlp.sources.regex_source import RegexSource  # noqa: E402
from hcmus_nlp.volume import from_canonical_dict  # noqa: E402


def _build_sources(
    *,
    use_seed: bool,
    use_cbdb: bool,
    kb_dir: Path,
    cbdb_short_name_policy: str = "exclude",
    cbdb_max_ambiguity: int = 10,
    cbdb_period_policy: str = "strict",
) -> list:
    """Build source adapters from verified local caches.

    Missing optional caches are skipped. A cache that exists but fails its
    manifest/hash check is reported and skipped rather than silently trusted.
    """
    sources: list = [RegexSource()]
    if use_seed:
        seed_cache = kb_dir / "seed.jsonl.gz"
        if seed_cache.exists():
            try:
                from hcmus_nlp.kb.seed import SeedSource, load_seed_cache

                entries, _manifest = load_seed_cache(seed_cache)
                if entries:
                    sources.append(SeedSource(entries))
            except (FileNotFoundError, ValueError) as e:
                print(f"[warn] seed cache unreadable: {e}", file=sys.stderr)
    if use_cbdb:
        cbdb_cache = kb_dir / "cbdb.sqlite"
        if cbdb_cache.exists():
            try:
                from hcmus_nlp.kb.cbdb import CBDBError, CBDBPersonSource

                source = CBDBPersonSource.from_cache(
                    cbdb_cache,
                    max_ambiguity=cbdb_max_ambiguity,
                    short_name_policy=cbdb_short_name_policy,
                    period_policy=cbdb_period_policy,
                )
                if source.available():
                    sources.append(source)
            except (CBDBError, FileNotFoundError, ValueError) as e:
                print(f"[warn] CBDB cache unreadable: {e}", file=sys.stderr)
    return sources


def _annotate_text(
    text: str,
    ctx: AnnotationContext,
    sources: list,
    merger: CandidateMerger,
) -> tuple[list[dict], list]:
    """Chạy tất cả source trên text, merge, trả (entities, conflicts)."""
    candidates: list[Candidate] = []
    for src in sources:
        if not src.available():
            continue
        candidates.extend(src.candidates(text, ctx))
    result = merger.merge(candidates)
    return list(result.entities), list(result.conflicts)


def _span_to_sentence_dict(s: Span, record_id: str, idx: int, text: str) -> dict:
    return {
        "sid": f"{record_id}-s{idx + 1}",
        "start": s.start,
        "end": s.end,
        "text": text,
        "method": "rule",
        "review_status": "needs_review",
    }


def _serialize_conflict(conflict) -> dict:
    """MergeConflict → JSON-serializable dict.

    Conflict spans bên trong `_annotate_text` là sentence-relative (vì
    `_annotate_text` được gọi với `sent_text`). Khi serialize vào record,
    ta cộng `sentence.start` để convert sang record-global (giống entity)
    và đánh dấu `offset_scope: "record"` để downstream biết cách xử lý.
    """
    return {
        "kind": conflict.kind,
        "offset_scope": "sentence",  # default; annotate_record sẽ đổi thành "record"
        "candidates": list(conflict.candidates),
    }


def annotate_record(
    record: dict,
    *,
    sources: list,
    merger: CandidateMerger,
) -> tuple[dict, Counter]:
    """Annotate một record: sentence + entities + unresolved_conflicts.

    Conflict từ CandidateMerger được gom vào `record["unresolved_conflicts"]`
    để strict validator và downstream tools có thể phát hiện. Mỗi conflict
    được tag `sentence_id` để truy ngược.
    """
    text = record["text"]
    record_id = record["id"]

    # Build AnnotationContext.
    vol_obj = from_canonical_dict(record) if record.get("volume_id") else None
    ctx = AnnotationContext(
        record_id=record_id,
        title=record.get("title", ""),
        period=record.get("period"),
        volume_id=vol_obj.canonical_key if vol_obj else None,
        source_file=record.get("source_file", ""),
        sentence_spans=tuple(),
    )

    sentence_spans = segment(text)
    sentences: list[dict] = []
    entities: list[dict] = []
    counts: Counter = Counter()
    all_conflicts: list[dict] = []

    for idx, s in enumerate(sentence_spans):
        sent = _span_to_sentence_dict(s, record_id, idx, text[s.start : s.end])
        sentences.append(sent)

        sent_text = text[s.start : s.end]
        sent_ctx = AnnotationContext(
            record_id=record_id,
            title=ctx.title,
            period=ctx.period,
            volume_id=ctx.volume_id,
            source_file=ctx.source_file,
            sentence_spans=tuple(sentence_spans),
        )
        sent_entities, sent_conflicts = _annotate_text(sent_text, sent_ctx, sources, merger)
        for ent in sent_entities:
            ent["eid"] = f"{record_id}-e{len(entities) + 1}"
            ent["sentence_id"] = sent["sid"]
            ent["start"] = ent["start"] + s.start
            ent["end"] = ent["end"] + s.start
            entities.append(ent)
            counts[ent["label"]] += 1

        # Tag conflict với sentence_id và convert span sang record-global.
        for conflict in sent_conflicts:
            cd = _serialize_conflict(conflict)
            cd["sentence_id"] = sent["sid"]
            # Convert sentence-relative spans sang record-global bằng cách
            # cộng sent.start vào mỗi candidate span; đổi offset_scope.
            cd["offset_scope"] = "record"
            new_candidates: list[dict] = []
            for cand in cd["candidates"]:
                cand_copy = dict(cand)
                if isinstance(cand_copy.get("start"), int):
                    cand_copy["start"] = cand_copy["start"] + s.start
                if isinstance(cand_copy.get("end"), int):
                    cand_copy["end"] = cand_copy["end"] + s.start
                new_candidates.append(cand_copy)
            cd["candidates"] = new_candidates
            all_conflicts.append(cd)

    result = dict(record)
    result["sentences"] = sentences
    result["entities"] = entities
    result["unresolved_conflicts"] = all_conflicts
    result["annotation"] = {
        "sentence_guideline_version": "0.1",
        "ner_guideline_version": "0.1",
        "status": "preannotation_needs_review",
        "offset_convention": "Unicode code points; start inclusive, end exclusive",
    }
    return result, counts


# Gold artifact root — resolve relative to repo root (script location), không
# phụ thuộc CWD. annotate_corpus.py không được ghi đè.
REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_ROOT = (REPO_ROOT / "build" / "gold").resolve()


def build(
    input_path: Path,
    output_path: Path,
    stats_path: Path,
    *,
    sources: list,
    merger: CandidateMerger,
) -> None:
    """Annotate corpus. KHÔNG ghi vào build/gold/ (Doccano CLI mới được ghi gold)."""
    output_resolved = output_path.resolve()
    try:
        is_in_gold = output_resolved.is_relative_to(GOLD_ROOT)
    except ValueError:
        is_in_gold = False
    if is_in_gold:
        raise ValueError(
            f"annotate_corpus.py không ghi vào {output_resolved} "
            "(nằm trong build/gold/). Dùng scripts/doccano_io.py from-doccano."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    totals: Counter = Counter()
    records = 0
    sentences = 0
    entities = 0
    with input_path.open(encoding="utf-8") as source, tmp.open("w", encoding="utf-8") as target:
        for line in source:
            record = json.loads(line)
            annotated, counts = annotate_record(record, sources=sources, merger=merger)
            target.write(json.dumps(annotated, ensure_ascii=False) + "\n")
            records += 1
            sentences += len(annotated["sentences"])
            entities += len(annotated["entities"])
            totals.update(counts)

    os.replace(tmp, output_path)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(
            {
                "records": records,
                "sentences": sentences,
                "entities": entities,
                "entities_by_label": dict(totals),
                "annotation_status": "preannotation_needs_review",
                "sources": [s.name for s in sources],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/corpus.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/corpus_preannotated.jsonl"),
        help="Output production. KHÔNG dùng build/gold/*.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("build/annotation_statistics.json"),
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Không load SeedSource (chỉ dùng RegexSource).",
    )
    parser.add_argument(
        "--no-cbdb",
        action="store_true",
        help="Không load CBDBPersonSource dù build/kb/cbdb.sqlite tồn tại.",
    )
    parser.add_argument(
        "--cbdb-short-names",
        choices=("context", "all", "exclude"),
        default="exclude",
        help=(
            "Policy cho tên CBDB dài 2 ký tự: exclude (mặc định, precision), "
            "context (cần cue), hoặc all."
        ),
    )
    parser.add_argument(
        "--cbdb-max-ambiguity",
        type=int,
        default=10,
        help="Bỏ surface CBDB liên kết quá số person ID này (mặc định: 10).",
    )
    parser.add_argument(
        "--cbdb-period-policy",
        choices=("strict", "prefer", "off"),
        default="strict",
        help=(
            "Dùng thời kỳ tác phẩm để lọc person ID: strict (mặc định, precision), "
            "prefer, hoặc off."
        ),
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=Path("build/kb"),
        help="Thư mục chứa seed.jsonl.gz và cbdb.sqlite (nếu có).",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("config/mapping.toml"),
        help="Mapping TOML cho CandidateMerger.",
    )
    args = parser.parse_args()

    try:
        mapping = load_mapping(args.mapping)
    except MappingError as e:
        raise SystemExit(f"Mapping error: {e}")

    if args.cbdb_max_ambiguity < 1:
        raise SystemExit("--cbdb-max-ambiguity phải >= 1")
    sources = _build_sources(
        use_seed=not args.no_seed,
        use_cbdb=not args.no_cbdb,
        kb_dir=args.kb_dir,
        cbdb_short_name_policy=args.cbdb_short_names,
        cbdb_max_ambiguity=args.cbdb_max_ambiguity,
        cbdb_period_policy=args.cbdb_period_policy,
    )
    merger = CandidateMerger(mapping, critical_sources=("cbdb", "guwen_basic"))

    build(
        args.input,
        args.output,
        args.stats,
        sources=sources,
        merger=merger,
    )
    print(f"Wrote annotated corpus to {args.output}")
    print(f"Sources: {[s.name for s in sources]}")


if __name__ == "__main__":
    main()
