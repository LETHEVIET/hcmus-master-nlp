#!/usr/bin/env python3
"""Create gold pilot (Phase D1).

Hai split tách biệt:
- evaluation_random: 200 câu default, stratified theo work, reproducible seed.
- diagnostic_challenge: stratified theo nhiều tiêu chí để error analysis.

Usage:
    uv run python scripts/create_gold_pilot.py \\
        --input build/corpus_preannotated.jsonl \\
        --output build/pilot \\
        --pilot-size 200 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()

from hcmus_nlp.gold import build_pilot  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("build/corpus_preannotated.jsonl"),
        help="Corpus preannotated để sample sentence.",
    )
    parser.add_argument("--output", type=Path, default=Path("build/pilot"))
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=200,
        help="Số câu cho evaluation_random. Production nên 800-1500.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--double-annotate",
        type=float,
        default=0.15,
        help="Tỷ lệ câu đánh double (15-20%%).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Input not found: {args.input}. Chạy prepare_corpus + annotate_corpus trước."
        )

    manifest = build_pilot(
        args.input,
        args.output,
        pilot_size=args.pilot_size,
        seed=args.seed,
        double_annotate_fraction=args.double_annotate,
    )

    print(
        json.dumps(
            {
                "wrote_evaluation_random": manifest.evaluation_random_path,
                "wrote_diagnostic_challenge": manifest.diagnostic_challenge_path,
                "wrote_manifest": str(args.output / "manifest.json"),
                "n_random": manifest.pilot_size,
                "work_quota": manifest.work_quota,
                "double_annotate_fraction": manifest.double_annotate_fraction,
                "seed": manifest.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
