#!/usr/bin/env python3
"""Doccano CLI (Phase E).

Subcommands:
- to-doccano: convert corpus preannotated → Doccano format.
- from-doccano: apply Doccano review → corpus reviewed (atomic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.doccano_io import (  # noqa: E402
    DoccanoError,
    import_from_doccano,
)


def cmd_to_doccano(args: argparse.Namespace) -> None:
    """Stream corpus → Doccano JSONL. Không load full corpus vào RAM."""
    from hcmus_nlp.doccano_io import export_to_doccano_stream

    if args.output.exists():
        raise SystemExit(
            f"Output {args.output} đã tồn tại. Doccano scratch có thể xóa; "
            "gold KHÔNG bao giờ nằm trong build/doccano/."
        )
    n = export_to_doccano_stream(args.input, args.output)
    print(f"Wrote {n} Doccano records to {args.output}")


def cmd_from_doccano(args: argparse.Namespace) -> None:
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("--input và --output không được trùng path")
    try:
        # Bypass overwrite guard nếu --force.
        from hcmus_nlp import doccano_io as _mod

        _mod.import_from_doccano._allow_overwrite = args.force
        stats = import_from_doccano(
            args.doccano,
            args.input,
            args.output,
            strict=args.strict,
            annotator=args.annotator,
            annotation_guideline_version=args.guideline_version,
            gold_version=args.gold_version,
        )
    except DoccanoError as e:
        raise SystemExit(f"Doccano error: {e}")
    # In tóm tắt + metadata path.
    summary = {k: v for k, v in stats.items() if k != "gold_metadata"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if "gold_metadata" in stats:
        meta_path = args.output.with_suffix(args.output.suffix + ".meta.json")
        print(f"Gold metadata: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_to = sub.add_parser("to-doccano", help="Convert corpus → Doccano format")
    p_to.add_argument("--input", type=Path, default=Path("build/corpus_preannotated.jsonl"))
    p_to.add_argument("--output", type=Path, default=Path("build/doccano/import.jsonl"))
    p_to.set_defaults(func=cmd_to_doccano)

    p_from = sub.add_parser("from-doccano", help="Apply Doccano review → corpus reviewed")
    p_from.add_argument("--doccano", type=Path, required=True)
    p_from.add_argument("--input", type=Path, default=Path("build/corpus_preannotated.jsonl"))
    p_from.add_argument(
        "--output",
        type=Path,
        default=Path("build/gold/pilot.checked.jsonl"),
    )
    p_from.add_argument("--strict", action="store_true")
    p_from.add_argument("--force", action="store_true", help="Ghi đè gold nếu tồn tại")
    p_from.add_argument("--annotator", default="human", help="Tên annotator (provenance)")
    p_from.add_argument(
        "--guideline-version",
        default="0.1",
        help="Annotation guideline version",
    )
    p_from.add_argument("--gold-version", default="v1", help="Gold artifact version")
    p_from.set_defaults(func=cmd_from_doccano)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
