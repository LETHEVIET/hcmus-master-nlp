#!/usr/bin/env python3
"""Build a structured monolingual Classical Chinese history corpus.

The files in dataset/ are treated as immutable raw sources.  This script
creates cleaned JSONL records plus metadata and statistics under build/.
Only obvious source-page boilerplate is removed; scholarly notes remain in
the text so that the transformation is reversible from the raw files.

Phase A1 (plan v5): volume heading được parse bằng `hcmus_nlp.volume.parse_volume_heading`
thay vì regex cũ. Record lưu thêm `volume_raw`, `volume_id` (string canonical),
`volume_number`, `volume_part` để downstream dùng trực tiếp. Field `volume`
giữ lại ở dạng canonical id để backward-compat với `export_submission.py`
hiện tại.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Bootstrap cho direct-script mode.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.cleaning import (  # noqa: E402
    Decision,
    audit,
    normalize_text,
)
from hcmus_nlp.volume import (  # noqa: E402
    VolumeId,
    assert_output_paths_injective,
    canonical_key_to_dict,
    detect_collisions,
    parse_volume_heading,
)

SOURCE_INFO = {
    "北史_full.txt": {
        "title_cn": "北史",
        "title_vi": "Bắc Sử",
        "period": "南北朝",
        "genre": "正史",
    },
    "北齊書_full.txt": {
        "title_cn": "北齊書",
        "title_vi": "Bắc Tề Thư",
        "period": "南北朝",
        "genre": "正史",
    },
    "Chư Phiên Chí - 諸蕃志_full.txt": {
        "title_cn": "諸蕃志",
        "title_vi": "Chư Phiên Chí",
        "period": "宋",
        "genre": "地理志",
    },
    "舊唐書_full.txt": {
        "title_cn": "舊唐書",
        "title_vi": "Cựu Đường Thư",
        "period": "唐",
        "genre": "正史",
    },
    "舊五代史_full.txt": {
        "title_cn": "舊五代史",
        "title_vi": "Cựu Ngũ Đại Sử",
        "period": "五代",
        "genre": "正史",
    },
    "Đông Quan Hán Ký - 東觀漢記_(四庫全書本)_full.txt": {
        "title_cn": "東觀漢記_(四庫全書本)",
        "title_vi": "Đông Quán Hán Ký",
        "period": "東漢",
        "genre": "史書",
    },
    "漢書_full.txt": {"title_cn": "漢書", "title_vi": "Hán Thư", "period": "西漢", "genre": "正史"},
    "後漢書_full.txt": {
        "title_cn": "後漢書",
        "title_vi": "Hậu Hán Thư",
        "period": "東漢",
        "genre": "正史",
    },
}

HASH_HEADING_RE = re.compile(r"^\s*#+\s*(.*?)\s*$")


def is_boilerplate(line: str) -> bool:
    """Backward-compat shim — dùng cleaning.is_boilerplate_line."""
    return line.strip() == "" or any(
        line.strip().startswith(p)
        for p in ("姊妹计划", "Public domain", "本作品在全世界都属于", "本作品 原文没有標點")
    )


# Use hcmus_nlp.cleaning.normalize_text thay vì định nghĩa cục bộ.


def parse_file(
    path: Path,
) -> tuple[list[dict], Counter, dict, list[dict], list[tuple[str, tuple[int, str]]]]:
    """Parse một file raw.

    Trả về (records, removed, stats, audit_decisions, heading_events).
    `heading_events` là list `(source_file, canonical_key)` cho MỖI heading
    event (không phải mỗi record). Mục đích: `detect_collisions` phân tích
    per-file để bắt repeated heading không liên tiếp, không bị ảnh hưởng bởi
    số record trong mỗi volume.

    Heading event collapse semantics: 2 heading markdown liên tiếp về
    mặt vật lý (không có record content chen giữa) cùng key → collapse.
    Nếu giữa 2 heading có record content thì đó là 2 event riêng biệt,
    kể cùng key. Việc collapse dựa vào `last_event_was_heading` flag,
    không dựa vào volume state.
    """
    nfc_name = unicodedata.normalize("NFC", path.name)
    info = SOURCE_INFO.get(nfc_name, {})
    title = info.get("title_cn", nfc_name.removesuffix("_full.txt"))
    records: list[dict] = []
    removed = Counter()
    audit_decisions: list[dict] = []
    heading_events: list[tuple[str, tuple[int, str]]] = []
    volume_canonical_id: str | None = None
    volume_id_obj: VolumeId | None = None
    section = None
    seen_first_record = False
    # True nếu dòng trước đó là heading event (không phải record content).
    # Khi gặp heading tiếp theo, nếu last_event_was_heading=True và
    # canonical_key giống → collapse (không push heading_event mới).
    last_event_was_heading = False

    def emit_drop(reason: str, line_no: int, raw: str) -> None:
        audit_decisions.append(
            {
                "source_file": path.name,
                "source_line": line_no,
                "decision": "drop",
                "reason": reason,
                "raw_preview": raw[:80],
            }
        )

    def flush_record_event() -> None:
        nonlocal last_event_was_heading
        last_event_was_heading = False

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            heading_match = HASH_HEADING_RE.match(line)
            if heading_match:
                heading = normalize_text(heading_match.group(1))
                parsed = parse_volume_heading(heading)
                if parsed is not None:
                    # Collapse: chỉ push heading_event khi KHÁC với event
                    # liền trước về mặt vật lý (last_event_was_heading=True
                    # và cùng key).
                    is_consecutive_same = (
                        last_event_was_heading
                        and volume_id_obj is not None
                        and parsed.canonical_key == volume_id_obj.canonical_key
                    )
                    if not is_consecutive_same:
                        heading_events.append((path.name, parsed.canonical_key))
                    volume_id_obj = parsed
                    volume_canonical_id = parsed.canonical_id()
                    section = None
                    last_event_was_heading = True
                elif heading:
                    section = heading
                    # Section heading (không phải volume) cũng là "heading event"
                    # trong luồng, nhưng không push vào heading_events vì chỉ
                    # theo dõi volume heading.
                    last_event_was_heading = True
                else:
                    last_event_was_heading = False
                removed["heading"] += 1
                continue

            flush_record_event()

            # Phase A2: dùng cleaning.audit() để phát hiện inline OCR / URL /
            # ellipsis dài cùng với boilerplate cũ.
            audit_result = audit(line, work_title=title if not seen_first_record else None)

            if audit_result.decision is Decision.DROP:
                primary_reason = audit_result.reasons[0].value if audit_result.reasons else "other"
                emit_drop(primary_reason, line_number, line)
                removed[primary_reason] += 1
                continue

            text = audit_result and normalize_text(line) or ""
            if not text:
                emit_drop("blank", line_number, line)
                removed["blank"] += 1
                continue

            seen_first_record = True

            # Quyết định cleaning_status cho record:
            # - DROP đã xử lý ở trên (không tới đây).
            # - NEEDS_REVIEW: giữ record, gắn reasons.
            # - KEEP: giữ record, cleaning_status="kept".
            cleaning_status = (
                "needs_review" if audit_result.decision is Decision.NEEDS_REVIEW else "kept"
            )
            cleaning_reasons = [r.value for r in audit_result.reasons]

            record: dict = {
                "id": f"{path.stem}-{len(records) + 1:06d}",
                "title": title,
                "title_vi": info.get("title_vi"),
                "period": info.get("period"),
                "genre": info.get("genre"),
                "language": "文言文",
                "volume": volume_canonical_id,
                "section": section,
                "source_file": path.name,
                "source_line": line_number,
                "text": text,
                "cleaning_status": cleaning_status,
                "cleaning_reasons": cleaning_reasons,
            }
            if volume_id_obj is not None:
                record.update(canonical_key_to_dict(volume_id_obj))
            else:
                record.update(
                    {
                        "volume_raw": None,
                        "volume_id": None,
                        "volume_number": None,
                        "volume_part": None,
                    }
                )
            records.append(record)

    stats = {
        "title": title,
        "title_vi": info.get("title_vi"),
        "period": info.get("period"),
        "genre": info.get("genre"),
        "records": len(records),
        "characters": sum(len(record["text"]) for record in records),
        "removed": dict(removed),
        "audit_decisions": len(audit_decisions),
        "heading_events": len(heading_events),
    }
    return records, removed, stats, audit_decisions, heading_events


def build(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    audit_path = output_dir / "audit_decisions.jsonl"
    all_stats = []
    total_removed = Counter()
    total_records = 0
    total_characters = 0
    seen_volumes: list[VolumeId] = []

    files = sorted(input_dir.glob("*.txt"))
    # heading_events từ parse_file (per heading event, không per record).
    all_heading_events: list[tuple[str, tuple[int, str]]] = []
    with (
        corpus_path.open("w", encoding="utf-8") as corpus,
        audit_path.open("w", encoding="utf-8") as audit_handle,
    ):
        for path in files:
            records, removed, stats, audit_decisions, heading_events = parse_file(path)
            all_heading_events.extend(heading_events)
            for record in records:
                vid_num = record.get("volume_number")
                vid_part = record.get("volume_part")
                vid_raw = record.get("volume_raw")
                if vid_num is not None and vid_raw is not None:
                    part = vid_part if vid_part in ("", "a", "b", "c") else ""
                    seen_volumes.append(VolumeId(number=vid_num, part=part, raw=vid_raw))
                corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
            for decision in audit_decisions:
                audit_handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
            all_stats.append(stats)
            total_removed.update(removed)
            total_records += len(records)
            total_characters += stats["characters"]

    # Injectivity guard: hai canonical key khác nhau phải tạo hai path khác nhau.
    assert_output_paths_injective(seen_volumes)

    # Repeated volume heading không liên tiếp trong cùng file → audit warning.
    collisions = detect_collisions(all_heading_events)
    if collisions:
        with audit_path.open("a", encoding="utf-8") as audit_append:
            for key, sources in collisions:
                audit_append.write(
                    json.dumps(
                        {
                            "source_file": sources[0] if sources else "<multiple>",
                            "decision": "warning",
                            "reason": "repeated_volume_heading",
                            "canonical_key": list(key),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        total_repeated = len(collisions)
    else:
        total_repeated = 0

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"sources": all_stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "statistics.json").write_text(
        json.dumps(
            {
                "files": len(files),
                "records": total_records,
                "characters": total_characters,
                "removed": dict(total_removed),
                "audit_decisions": sum(s.get("audit_decisions", 0) for s in all_stats),
                "repeated_volume_headings": total_repeated,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("build"))
    args = parser.parse_args()
    build(args.input, args.output)
    print(f"Wrote structured corpus to {args.output}")


if __name__ == "__main__":
    main()
