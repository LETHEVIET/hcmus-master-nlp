"""Gold pilot sampler (Phase D1).

Hai split tách biệt:
- evaluation_random: stratified by work only, dùng cho metric chính.
- diagnostic_challenge: stratified theo dấu kết thúc / câu dài / 〈案…〉 /
  PERSON candidate / NUMBER+TIME / nguồn bất đồng / quyển — CHỈ error
  analysis, KHÔNG feed metric.

Default pilot size 200 câu (≈ 8.000 chars) cho session; production mở rộng
800–1.500 câu qua flag.

Cả hai split phải disjoint theo sentence_id.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class SentenceRecord:
    sentence_id: str
    record_id: str
    title: str
    volume_id: str | None
    text: str
    start: int
    end: int
    has_punct_end: bool
    is_long: bool
    has_annotation_案: bool
    n_entities: int
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class PilotManifest:
    seed: int
    pilot_size: int
    work_quota: dict[str, int]
    double_annotate_fraction: float
    evaluation_random_path: str
    diagnostic_challenge_path: str
    note: str = ""


_HAS_ANNOTATION_案_RE = re.compile(r"〈案[^〉]*〉")
_NUM_RE = re.compile(r"[0-9０-９一二三四五六七八九十百千万]+")
_TIME_LIKE_RE = re.compile(r"(元年|年|朝|月|日)")


def collect_sentences(
    corpus_path: Path,
) -> list[SentenceRecord]:
    """Đọc corpus_preannotated.jsonl và build list SentenceRecord."""
    out: list[SentenceRecord] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            record_id = r["id"]
            title = r.get("title", "")
            volume_id = r.get("volume_id")
            for sent in r.get("sentences", []):
                text = sent.get("text", "")
                entities_for_sent = [
                    e for e in r.get("entities", []) if e.get("sentence_id") == sent["sid"]
                ]
                labels = tuple(sorted({e["label"] for e in entities_for_sent}))
                out.append(
                    SentenceRecord(
                        sentence_id=sent["sid"],
                        record_id=record_id,
                        title=title,
                        volume_id=volume_id,
                        text=text,
                        start=sent["start"],
                        end=sent["end"],
                        has_punct_end=bool(text and text[-1] in "。！？!?.｡"),
                        is_long=len(text) > 80,
                        has_annotation_案=bool(_HAS_ANNOTATION_案_RE.search(text)),
                        n_entities=len(entities_for_sent),
                        labels=labels,
                    )
                )
    return out


def stratified_sample(
    sentences: Iterable[SentenceRecord],
    n: int,
    *,
    seed: int,
    work_quota: dict[str, int] | None = None,
) -> list[SentenceRecord]:
    """Sample stratified theo title (work).

    Nếu work_quota cho trước (vd `{"漢書": 30, "後漢書": 30}`), phân bổ đúng
    quota. Ngược lại chia đều cho các work.
    """
    rng = random.Random(seed)
    by_title: dict[str, list[SentenceRecord]] = defaultdict(list)
    for s in sentences:
        by_title[s.title].append(s)

    if work_quota is None:
        # Chia đều.
        per_work = max(1, n // max(1, len(by_title)))
        work_quota = {title: per_work for title in by_title}

    selected: list[SentenceRecord] = []
    for title, quota in work_quota.items():
        pool = by_title.get(title, [])
        rng.shuffle(pool)
        selected.extend(pool[:quota])

    # Nếu chưa đủ n, bù từ phần còn lại.
    if len(selected) < n:
        seen_ids = {s.sentence_id for s in selected}
        remaining = [s for s in sentences if s.sentence_id not in seen_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])

    return selected[:n]


def diagnostic_strata(sentences: list[SentenceRecord]) -> dict[str, list[SentenceRecord]]:
    """Chia sentence theo strata để error analysis (KHÔNG dùng cho metric)."""
    out: dict[str, list[SentenceRecord]] = {
        "no_punct_end": [],
        "long_sentence": [],
        "has_annotation_案": [],
        "has_person_candidate": [],
        "has_num_and_time": [],
        "multi_source": [],
    }
    for s in sentences:
        if not s.has_punct_end:
            out["no_punct_end"].append(s)
        if s.is_long:
            out["long_sentence"].append(s)
        if s.has_annotation_案:
            out["has_annotation_案"].append(s)
        if "PERSON" in s.labels:
            out["has_person_candidate"].append(s)
        if any(l.startswith("NUMBER") or l == "NUMBER" for l in s.labels) and any(
            l == "TIME" for l in s.labels
        ):
            out["has_num_and_time"].append(s)
        if len(s.labels) >= 2:
            out["multi_source"].append(s)
    return out


def sample_diagnostic_challenge(
    sentences: list[SentenceRecord],
    *,
    seed: int,
    per_stratum: int = 20,
) -> list[SentenceRecord]:
    """Lấy mẫu cho diagnostic set. Mỗi stratum tối đa `per_stratum` câu."""
    rng = random.Random(seed + 1)
    strata = diagnostic_strata(sentences)
    selected: list[SentenceRecord] = []
    seen: set[str] = set()
    for stratum in strata.values():
        rng.shuffle(stratum)
        for s in stratum[:per_stratum]:
            if s.sentence_id not in seen:
                seen.add(s.sentence_id)
                selected.append(s)
    return selected


def write_pilot_jsonl(
    sentences: Iterable[SentenceRecord],
    output_path: Path,
    *,
    double_annotate_fraction: float = 0.0,
    seed: int = 42,
) -> None:
    """Ghi pilot JSONL. Mỗi dòng `{sentence_id, text, double_annotate}`."""
    rng = random.Random(seed + 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for s in sentences:
            record = {
                "sentence_id": s.sentence_id,
                "text": s.text,
                "record_id": s.record_id,
                "title": s.title,
                "volume_id": s.volume_id,
                "double_annotate": rng.random() < double_annotate_fraction,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_pilot(
    corpus_path: Path,
    output_dir: Path,
    *,
    pilot_size: int = 200,
    seed: int = 42,
    double_annotate_fraction: float = 0.15,
    work_quota: dict[str, int] | None = None,
) -> PilotManifest:
    """Build pilot đầy đủ (random + diagnostic + manifest)."""
    sentences = collect_sentences(corpus_path)
    if not sentences:
        raise ValueError(f"No sentences in {corpus_path}")

    eval_random = stratified_sample(sentences, pilot_size, seed=seed, work_quota=work_quota)
    eval_ids = {s.sentence_id for s in eval_random}
    remaining = [s for s in sentences if s.sentence_id not in eval_ids]

    diagnostic = sample_diagnostic_challenge(remaining, seed=seed)
    diag_ids = {s.sentence_id for s in diagnostic}
    # Đảm bảo disjoint.
    assert not (eval_ids & diag_ids), "evaluation_random và diagnostic phải disjoint"

    eval_path = output_dir / "evaluation_random.jsonl"
    diag_path = output_dir / "diagnostic_challenge.jsonl"

    write_pilot_jsonl(
        eval_random, eval_path, double_annotate_fraction=double_annotate_fraction, seed=seed
    )
    write_pilot_jsonl(diagnostic, diag_path, double_annotate_fraction=0.0, seed=seed)

    # Work quota breakdown.
    quota_breakdown: dict[str, int] = dict(Counter(s.title for s in eval_random))

    manifest = PilotManifest(
        seed=seed,
        pilot_size=pilot_size,
        work_quota=quota_breakdown,
        double_annotate_fraction=double_annotate_fraction,
        evaluation_random_path=str(eval_path),
        diagnostic_challenge_path=str(diag_path),
        note=(
            "evaluation_random: metric chính (F1). "
            "diagnostic_challenge: error analysis only, KHÔNG dùng cho metric."
        ),
    )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


__all__ = [
    "PilotManifest",
    "SentenceRecord",
    "build_pilot",
    "collect_sentences",
    "diagnostic_strata",
    "sample_diagnostic_challenge",
    "stratified_sample",
    "write_pilot_jsonl",
]
