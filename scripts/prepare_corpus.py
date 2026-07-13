#!/usr/bin/env python3
"""Build a structured monolingual Classical Chinese history corpus.

The files in dataset/ are treated as immutable raw sources.  This script
creates cleaned JSONL records plus metadata and statistics under build/.
Only obvious source-page boilerplate is removed; scholarly notes remain in
the text so that the transformation is reversible from the raw files.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


SOURCE_INFO = {
    "北史_full.txt": {"title_cn": "北史", "title_vi": "Bắc Sử", "period": "南北朝", "genre": "正史"},
    "北齊書_full.txt": {"title_cn": "北齊書", "title_vi": "Bắc Tề Thư", "period": "南北朝", "genre": "正史"},
    "Chư Phiên Chí - 諸蕃志_full.txt": {
        "title_cn": "諸蕃志", "title_vi": "Chư Phiên Chí", "period": "宋", "genre": "地理志"
    },
    "舊唐書_full.txt": {"title_cn": "舊唐書", "title_vi": "Cựu Đường Thư", "period": "唐", "genre": "正史"},
    "舊五代史_full.txt": {"title_cn": "舊五代史", "title_vi": "Cựu Ngũ Đại Sử", "period": "五代", "genre": "正史"},
    "Đông Quan Hán Ký - 東觀漢記_(四庫全書本)_full.txt": {
        "title_cn": "東觀漢記_(四庫全書本)", "title_vi": "Đông Quán Hán Ký", "period": "東漢", "genre": "史書"
    },
    "漢書_full.txt": {"title_cn": "漢書", "title_vi": "Hán Thư", "period": "西漢", "genre": "正史"},
    "後漢書_full.txt": {"title_cn": "後漢書", "title_vi": "Hậu Hán Thư", "period": "東漢", "genre": "正史"},
}

VOLUME_RE = re.compile(r"^卷\s*[0-9０-９一二三四五六七八九十百千零〇]+(?:上|中|下)?$")
HASH_HEADING_RE = re.compile(r"^\s*#+\s*(.*?)\s*$")


def is_boilerplate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped.startswith("姊妹计划"):
        return True
    if stripped.startswith("Public domain"):
        return True
    if stripped.startswith("本作品在全世界都属于"):
        return True
    if stripped.startswith("本作品 原文没有標點"):
        return True
    return False


def normalize_text(line: str) -> str:
    # Preserve Chinese punctuation and characters; normalize only layout.
    return re.sub(r"[ \t\u00a0]+", " ", line.strip())


def parse_file(path: Path) -> tuple[list[dict], Counter, dict]:
    info = SOURCE_INFO.get(path.name, {})
    title = info.get("title_cn", path.name.removesuffix("_full.txt"))
    records: list[dict] = []
    removed = Counter()
    volume = None
    section = None

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            heading_match = HASH_HEADING_RE.match(line)
            if heading_match:
                heading = normalize_text(heading_match.group(1))
                if VOLUME_RE.match(heading):
                    volume = heading
                    section = None
                elif heading:
                    section = heading
                removed["heading"] += 1
                continue

            if is_boilerplate(line):
                removed["boilerplate"] += 1
                continue

            text = normalize_text(line)
            if not text:
                removed["blank"] += 1
                continue

            # Drop a repeated standalone work title, but never alter a title
            # occurring inside a historical paragraph.
            if not records and text in {title, path.name.removesuffix("_full.txt")}:
                removed["work_title"] += 1
                continue

            records.append({
                "id": f"{path.stem}-{len(records) + 1:06d}",
                "title": title,
                "title_vi": info.get("title_vi"),
                "period": info.get("period"),
                "genre": info.get("genre"),
                "language": "文言文",
                "volume": volume,
                "section": section,
                "source_file": path.name,
                "source_line": line_number,
                "text": text,
            })

    stats = {
        "title": title,
        "title_vi": info.get("title_vi"),
        "period": info.get("period"),
        "genre": info.get("genre"),
        "records": len(records),
        "characters": sum(len(record["text"]) for record in records),
        "removed": dict(removed),
    }
    return records, removed, stats


def build(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    all_stats = []
    total_removed = Counter()
    total_records = 0
    total_characters = 0

    files = sorted(input_dir.glob("*.txt"))
    with corpus_path.open("w", encoding="utf-8") as corpus:
        for path in files:
            records, removed, stats = parse_file(path)
            for record in records:
                corpus.write(json.dumps(record, ensure_ascii=False) + "\n")
            all_stats.append(stats)
            total_removed.update(removed)
            total_records += len(records)
            total_characters += stats["characters"]

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps({"sources": all_stats}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "statistics.json").write_text(
        json.dumps({
            "files": len(files),
            "records": total_records,
            "characters": total_characters,
            "removed": dict(total_removed),
        }, ensure_ascii=False, indent=2) + "\n",
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
