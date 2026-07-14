#!/usr/bin/env python3
"""KB ingest CLI — seed (default) / CBDB / CHGIS.

KHÔNG auto-download trong session này. User cần:
1. Tải file external (CBDB SQLite, CHGIS CSV) thủ công.
2. Chạy `ingest-cbdb --input PATH` hoặc `ingest-chgis --input PATH`.
3. Cache + manifest tự động ghi vào build/kb/.

Usage:
    uv run python scripts/build_kb.py ingest-seed
    uv run python scripts/build_kb.py ingest-cbdb --input path/to/cbdb.sqlite \\
        --version 2024.06 --source-url https://... --license cc-by-nc-sa-4.0
    uv run python scripts/build_kb.py ingest-chgis --input path/to/places.csv \\
        --version 2024.06 --source-url https://... --license chgis-terms
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()


KB_CACHE_DIR = Path("build/kb")


def cmd_ingest_seed(args: argparse.Namespace) -> None:
    from hcmus_nlp.kb.seed import all_default_seeds, write_seed_cache

    cache = KB_CACHE_DIR / "seed.jsonl.gz"
    manifest = write_seed_cache(
        all_default_seeds(),
        cache,
        version=args.version,
        source_url=args.source_url or "internal://seed-default",
        license=args.license or "internal-cc0",
    )
    print(f"Wrote {cache}")
    print(f"Manifest: {cache}.manifest.json")
    print(f"SHA-256: {manifest.file_sha256}")
    print(f"Rows: {manifest.row_counts}")


def cmd_ingest_cbdb(args: argparse.Namespace) -> None:
    from hcmus_nlp.kb.cbdb import build_cbdb_cache

    cache = KB_CACHE_DIR / "cbdb.sqlite"
    manifest = build_cbdb_cache(
        args.input,
        cache,
        version=args.version,
        source_url=args.source_url,
        license=args.license or "cc-by-nc-sa-4.0",
    )
    print(f"Wrote {cache}")
    print(f"SHA-256: {manifest.file_sha256}")
    print(f"Rows: {manifest.row_counts}")


def cmd_ingest_chgis(args: argparse.Namespace) -> None:
    from hcmus_nlp.kb.chgis import build_chgis_cache

    if not args.source_url or not args.license:
        raise SystemExit(
            "CHGIS ingest yêu cầu --source-url và --license (license thay đổi theo phiên bản)."
        )
    cache = KB_CACHE_DIR / "chgis.sqlite"
    manifest = build_chgis_cache(
        args.input,
        cache,
        version=args.version,
        source_url=args.source_url,
        license=args.license,
    )
    print(f"Wrote {cache}")
    print(f"SHA-256: {manifest.file_sha256}")
    print(f"Rows: {manifest.row_counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("ingest-seed", help="Ingest default seed lists")
    p_seed.add_argument("--version", default="0.1.0")
    p_seed.add_argument("--source-url", default=None)
    p_seed.add_argument("--license", default=None)

    p_cbdb = sub.add_parser("ingest-cbdb", help="Ingest CBDB SQLite")
    p_cbdb.add_argument("--input", type=Path, required=True)
    p_cbdb.add_argument("--version", required=True)
    p_cbdb.add_argument("--source-url", default=None)
    p_cbdb.add_argument("--license", default="cc-by-nc-sa-4.0")

    p_chgis = sub.add_parser("ingest-chgis", help="Ingest CHGIS CSV")
    p_chgis.add_argument("--input", type=Path, required=True)
    p_chgis.add_argument("--version", required=True)
    p_chgis.add_argument("--source-url", required=True)
    p_chgis.add_argument("--license", required=True)

    args = parser.parse_args()
    if args.command == "ingest-seed":
        cmd_ingest_seed(args)
    elif args.command == "ingest-cbdb":
        cmd_ingest_cbdb(args)
    elif args.command == "ingest-chgis":
        cmd_ingest_chgis(args)


if __name__ == "__main__":
    main()
