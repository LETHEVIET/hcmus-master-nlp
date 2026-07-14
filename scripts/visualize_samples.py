#!/usr/bin/env python3
"""Render representative annotated corpus samples as a self-contained HTML file."""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import ensure_src_on_path  # noqa: E402

ensure_src_on_path()


LABEL_COLORS = {
    "PERSON": "#dc2626",
    "LOCATION": "#2563eb",
    "POLITY": "#7c3aed",
    "DYNASTY": "#9333ea",
    "OFFICIAL_TITLE": "#059669",
    "BOOK": "#d97706",
    "TIME": "#0891b2",
    "NUMBER": "#4f46e5",
}


@dataclass(frozen=True)
class Sample:
    title: str
    volume_id: str | None
    record_id: str
    sentence_id: str
    sentence_start: int
    text: str
    review_status: str
    entities: tuple[dict, ...]
    conflicts: tuple[dict, ...]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted({str(entity.get("label", "")) for entity in self.entities}))

    @property
    def score(self) -> tuple[int, int, int, int, str]:
        readable_length = 1 if 12 <= len(self.text) <= 120 else 0
        return (
            bool(self.conflicts),
            len(self.labels),
            len(self.entities),
            readable_length,
            self.sentence_id,
        )


def collect_samples(corpus_path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with corpus_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            entities_by_sentence: dict[str, list[dict]] = {}
            for entity in record.get("entities", []):
                entities_by_sentence.setdefault(entity.get("sentence_id", ""), []).append(entity)
            conflicts_by_sentence: dict[str, list[dict]] = {}
            for conflict in record.get("unresolved_conflicts", []):
                conflicts_by_sentence.setdefault(conflict.get("sentence_id", ""), []).append(
                    conflict
                )

            for sentence in record.get("sentences", []):
                sid = sentence.get("sid", "")
                entities = tuple(
                    sorted(
                        entities_by_sentence.get(sid, []),
                        key=lambda item: (item.get("start", 0), item.get("end", 0)),
                    )
                )
                conflicts = tuple(conflicts_by_sentence.get(sid, []))
                if not entities and not conflicts:
                    continue
                samples.append(
                    Sample(
                        title=record.get("title", ""),
                        volume_id=record.get("volume_id"),
                        record_id=record.get("id", ""),
                        sentence_id=sid,
                        sentence_start=sentence.get("start", 0),
                        text=sentence.get("text", ""),
                        review_status=sentence.get("review_status", "unknown"),
                        entities=entities,
                        conflicts=conflicts,
                    )
                )
    return samples


def choose_samples(samples: list[Sample], limit: int) -> list[Sample]:
    """Choose a deterministic mix of conflict and ordinary annotated samples."""
    readable = [
        sample for sample in samples if 12 <= len(sample.text) <= 120 and len(sample.entities) <= 20
    ]
    pool = readable if len(readable) >= limit else samples
    selected: list[Sample] = []
    selected_ids: set[str] = set()
    covered_labels: set[str] = set()
    covered_works: set[str] = set()

    def pick(pool: list[Sample], count: int) -> None:
        def gain(sample: Sample) -> tuple[int, int, int, tuple[int, int, int, int, str]]:
            return (
                len(set(sample.labels) - covered_labels),
                int(sample.title not in covered_works),
                int(bool(sample.conflicts)),
                sample.score,
            )

        for _ in range(count):
            remaining = [sample for sample in pool if sample.sentence_id not in selected_ids]
            if not remaining:
                break
            chosen = max(remaining, key=gain)
            selected.append(chosen)
            selected_ids.add(chosen.sentence_id)
            covered_labels.update(chosen.labels)
            covered_works.add(chosen.title)

    conflict_samples = [sample for sample in pool if sample.conflicts]
    ordinary_samples = [sample for sample in pool if not sample.conflicts]
    conflict_quota = min(len(conflict_samples), max(1, limit // 3))
    pick(conflict_samples, conflict_quota)
    pick(ordinary_samples, limit - len(selected))
    if len(selected) < limit:
        pick(pool, limit - len(selected))

    return selected


def render_annotated_text(sample: Sample) -> str:
    cursor = 0
    chunks: list[str] = []
    for entity in sample.entities:
        start = entity.get("start", 0) - sample.sentence_start
        end = entity.get("end", 0) - sample.sentence_start
        if not (0 <= start < end <= len(sample.text)) or start < cursor:
            continue
        chunks.append(html.escape(sample.text[cursor:start]))
        label = str(entity.get("label", "UNKNOWN"))
        color = LABEL_COLORS.get(label, "#475569")
        surface = html.escape(sample.text[start:end])
        sources = ", ".join(map(str, entity.get("sources", []))) or "unknown"
        title = html.escape(
            f"{label} · offsets [{entity.get('start')}, {entity.get('end')}) · sources: {sources}"
        )
        chunks.append(
            f'<mark class="entity" style="--label-color:{color}" title="{title}">'
            f'{surface}<span class="entity-label">{html.escape(label)}</span></mark>'
        )
        cursor = end
    chunks.append(html.escape(sample.text[cursor:]))
    return "".join(chunks)


def render_conflicts(sample: Sample) -> str:
    if not sample.conflicts:
        return ""
    rows: list[str] = []
    for conflict in sample.conflicts:
        candidates = []
        for candidate in conflict.get("candidates", []):
            sources = candidate.get("sources") or [candidate.get("source", "unknown")]
            candidates.append(
                "<li>"
                f"<code>{html.escape(str(candidate.get('text', '')))}</code> "
                f"→ <strong>{html.escape(str(candidate.get('label', '')))}</strong> "
                f"[{candidate.get('start')}, {candidate.get('end')}) · "
                f"{html.escape(', '.join(map(str, sources)))}"
                "</li>"
            )
        rows.append(
            '<div class="conflict">'
            f'<div class="conflict-title">⚠ {html.escape(str(conflict.get("kind", "conflict")))}</div>'
            f"<ul>{''.join(candidates)}</ul>"
            "</div>"
        )
    return (
        '<section class="conflicts"><h3>Unresolved candidates</h3>' + "".join(rows) + "</section>"
    )


def render_html(samples: list[Sample], corpus_path: Path) -> str:
    label_counts = Counter(label for sample in samples for label in sample.labels)
    legend = "".join(
        f'<span class="legend-item"><i style="background:{color}"></i>{html.escape(label)}</span>'
        for label, color in LABEL_COLORS.items()
    )
    cards = []
    for index, sample in enumerate(samples, start=1):
        chips = "".join(
            f'<span class="chip">{html.escape(label)}</span>' for label in sample.labels
        )
        cards.append(
            '<article class="card">'
            "<header>"
            f'<div><span class="sample-number">#{index:02d}</span> '
            f"<strong>{html.escape(sample.title)}</strong> · 卷 {html.escape(sample.volume_id or '—')}</div>"
            f'<div class="status">{html.escape(sample.review_status)}</div>'
            "</header>"
            f'<div class="sid">{html.escape(sample.sentence_id)}</div>'
            f'<div class="annotated-text" lang="zh-Hant">{render_annotated_text(sample)}</div>'
            f"<footer>{chips}<span>{len(sample.entities)} entities · {len(sample.conflicts)} conflicts</span></footer>"
            f"{render_conflicts(sample)}"
            "</article>"
        )

    summary = ", ".join(f"{label}: {count}" for label, count in sorted(label_counts.items()))
    return f"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HCMUS Hán cổ · Annotation samples</title>
<style>
:root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; background:#f6f3ed; color:#1f2937; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; }}
.page {{ max-width:1100px; margin:auto; padding:42px 24px 64px; }}
h1 {{ margin:0 0 8px; font-family:Georgia, serif; font-size:clamp(2rem,5vw,3.5rem); color:#172554; }}
.subtitle {{ color:#64748b; margin-bottom:24px; }}
.summary {{ background:#172554; color:#e2e8f0; border-radius:14px; padding:16px 18px; margin-bottom:18px; }}
.legend {{ display:flex; flex-wrap:wrap; gap:9px 15px; margin:20px 0 28px; }}
.legend-item {{ display:inline-flex; align-items:center; gap:6px; font-size:.82rem; }}
.legend-item i {{ width:10px; height:10px; border-radius:50%; }}
.grid {{ display:grid; gap:18px; }}
.card {{ background:#fff; border:1px solid #e2ddd4; border-radius:18px; padding:20px; box-shadow:0 8px 30px rgba(30,41,59,.06); }}
.card header, .card footer {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }}
.sample-number {{ color:#94a3b8; font-variant-numeric:tabular-nums; }}
.status {{ color:#b45309; background:#fffbeb; border:1px solid #fde68a; padding:3px 9px; border-radius:999px; font-size:.75rem; }}
.sid {{ margin-top:7px; color:#94a3b8; font:11px ui-monospace, monospace; overflow-wrap:anywhere; }}
.annotated-text {{ font-family:"Noto Serif CJK TC","Songti TC","SimSun",serif; font-size:1.45rem; line-height:2.35; margin:18px 0; }}
.entity {{ --label-color:#475569; position:relative; background:color-mix(in srgb,var(--label-color) 15%,white); border-bottom:3px solid var(--label-color); border-radius:4px; padding:2px 3px; margin:0 1px; }}
.entity-label {{ font:700 9px ui-sans-serif,system-ui; color:white; background:var(--label-color); border-radius:4px; padding:2px 4px; margin-left:4px; vertical-align:super; letter-spacing:.03em; }}
.chip {{ background:#f1f5f9; color:#475569; border-radius:999px; padding:3px 8px; font-size:.72rem; margin-right:5px; }}
.card footer {{ color:#64748b; font-size:.76rem; border-top:1px solid #f1f5f9; padding-top:12px; }}
.conflicts {{ margin-top:15px; border-top:1px dashed #f59e0b; padding-top:12px; }}
.conflicts h3 {{ margin:0 0 8px; color:#92400e; font-size:.9rem; }}
.conflict {{ background:#fffbeb; border-left:4px solid #f59e0b; padding:9px 12px; margin-top:8px; font-size:.78rem; }}
.conflict-title {{ font-weight:700; color:#92400e; }}
.conflict ul {{ margin:6px 0 0; padding-left:20px; }}
code {{ background:#fff; padding:1px 4px; border-radius:3px; }}
@media (max-width:640px) {{ .annotated-text {{ font-size:1.2rem; }} .page {{ padding:24px 14px 40px; }} }}
</style>
</head>
<body><main class="page">
<h1>Annotation samples</h1>
<div class="subtitle">Corpus Hán cổ lịch sử · source: {html.escape(str(corpus_path))}</div>
<div class="summary"><strong>{len(samples)} representative sentences</strong><br>{html.escape(summary)}</div>
<div class="legend">{legend}</div>
<section class="grid">{"".join(cards)}</section>
</main></body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("build/corpus_preannotated.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("build/annotation_samples.html"))
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be >= 1")

    samples = choose_samples(collect_samples(args.input), args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_html(samples, args.input), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "samples": len(samples),
                "works": sorted({sample.title for sample in samples}),
                "labels": sorted({label for sample in samples for label in sample.labels}),
                "conflict_samples": sum(bool(sample.conflicts) for sample in samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
