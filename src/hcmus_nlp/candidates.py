"""Trie + CandidateMerger (Phase C3+C4).

Trie là dict-of-dict, KHÔNG phải Aho-Corasick — chưa failure transitions.
`find_all()` emit **mọi match kể cả overlap** để CandidateMerger quyết định.
Với CBDB full (600k name), sẽ cần benchmark hoặc implement fail-link sau.

CandidateMerger áp dụng 8 rule từ plan v5:
1. Cùng (start, end, label) → union sources, max priority, 1 entity.
2. Cùng (start, end), label khác, compatible → 1 entity với preferred label.
3. Cùng (start, end), label khác, incompatible → unresolved_conflicts.
4. Strict nested cùng label → outer giữ (hoặc inner nếu critical_source).
5. Partial overlap khác label → unresolved_conflicts, không emit.
6. Strict nested khác label → unresolved_conflicts, không emit.
7. Tie same priority → deterministic theo (source asc, source_id asc).
8. Disjoint → cả hai emit.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from hcmus_nlp.labels import Mapping as LabelMapping
from hcmus_nlp.source_base import Candidate

# --- Trie (KHÔNG Aho-Corasick) -----------------------------------------------


class Trie:
    """Trie đơn giản. `find_all(text)` trả list match (start, end, term).

    Trie có thể được build từ dict[str, list[str]] hoặc dict[str, list[tuple[str, str]]]
    (term, alias).

    Ví dụ:
        t = Trie()
        t.insert("高祖", "PERSON")
        t.insert("漢書", "BOOK")
        t.find_all("高祖沛人也，《漢書》卷一。")
        # [(0, 2, "高祖", "PERSON"), (8, 10, "漢書", "BOOK")]
    """

    __slots__ = ("_root",)

    def __init__(self) -> None:
        # Mỗi node: {char: {..., '$payload': list[(term, label)]}}
        self._root: dict = {}

    def insert(self, term: str, label: str, *, alias: str | None = None) -> None:
        if not term:
            return
        node = self._root
        for char in term:
            node = node.setdefault(char, {})
        payload = node.setdefault("$payload", [])
        payload.append((term, label, alias or term))

    def find_all(self, text: str) -> list[tuple[int, int, str, str, str]]:
        """Trả list (start, end, term, label, alias). Có thể overlap."""
        results: list[tuple[int, int, str, str, str]] = []
        n = len(text)
        for i in range(n):
            node = self._root
            j = i
            while j < n and text[j] in node:
                node = node[text[j]]
                j += 1
                payload = node.get("$payload")
                if payload:
                    for term, label, alias in payload:
                        results.append((i, j, term, label, alias))
        return results

    def to_dict(self) -> dict:
        """Serialize cho cache. JSON-able."""
        return self._root

    @classmethod
    def from_dict(cls, data: dict) -> "Trie":
        t = cls()
        t._root = data
        return t


# --- CandidateMerger ---------------------------------------------------------


@dataclass(frozen=True)
class MergeConflict:
    """Một conflict unresolved từ merger."""

    kind: str  # "same_span_label" | "partial_overlap" | "nested_label"
    candidates: tuple[dict, ...]


@dataclass(frozen=True)
class MergeResult:
    """Kết quả merge: flat entities + conflicts."""

    entities: tuple[dict, ...]
    conflicts: tuple[MergeConflict, ...]


class CandidateMerger:
    """Gộp Candidate từ nhiều source thành flat NER + conflicts.

    Args:
        label_mapping: Mapping (config) để biết compatible groups và
            priority_label_order.
        critical_sources: tuple source name được ưu tiên khi strict nested
            cùng label (vd `("cbdb", "guwen_basic")`).
    """

    def __init__(
        self,
        label_mapping: LabelMapping,
        critical_sources: tuple[str, ...] = (),
    ) -> None:
        self._mapping = label_mapping
        self._critical = set(critical_sources)

    def merge(self, candidates: Iterable[Candidate]) -> MergeResult:
        # Sort deterministic.
        sorted_candidates = sorted(
            candidates,
            key=lambda c: (c.start, -c.end, c.label, c.source, c.source_id or ""),
        )
        flat: list[dict] = []
        conflicts: list[MergeConflict] = []

        for cand in sorted_candidates:
            placed = False
            drop_existing: list[int] = []
            new_flat: list[dict] = []
            for idx, existing in enumerate(flat):
                decision = self._decide(cand, existing, conflicts)
                if decision == "merge":
                    new_flat.append(self._merge_two(cand, existing))
                    placed = True
                elif decision == "conflict":
                    # Cùng (start, end) khác label không compatible → KHÔNG
                    # emit cả existing và cand; cả hai vào unresolved_conflicts.
                    conflicts.append(
                        MergeConflict(
                            kind="same_span_label",
                            candidates=(
                                self._candidate_to_dict(cand),
                                self._existing_to_dict(existing),
                            ),
                        )
                    )
                    placed = True
                    drop_existing.append(idx)
                else:  # "keep_both" — xét overlap/nested/disjoint.
                    relation = self._relation(cand, existing)
                    if relation == "disjoint":
                        new_flat.append(existing)
                    else:
                        # partial overlap hoặc nested khác label → conflict,
                        # KHÔNG emit cả cand và existing.
                        kind = (
                            "partial_overlap" if relation == "partial_overlap" else "nested_label"
                        )
                        # Strict nested cùng label: giữ outer (theo sort key
                        # end desc, outer đến trước); inner bị drop. Nếu
                        # inner từ critical_source → đánh dấu conflict để
                        # người duyệt quyết.
                        if kind == "nested_label" and cand.label == existing["label"]:
                            if cand.source in self._critical:
                                # Critical inner từ KB/model → conflict,
                                # người duyệt xử lý.
                                conflicts.append(
                                    MergeConflict(
                                        kind="nested_critical",
                                        candidates=(
                                            self._candidate_to_dict(cand),
                                            self._existing_to_dict(existing),
                                        ),
                                    )
                                )
                                placed = True
                                drop_existing.append(idx)
                                continue
                            # Không critical: outer giữ (existing), inner (cand) drop.
                            new_flat.append(existing)
                            placed = True
                            continue
                        # Các trường hợp overlap/nested khác label đã xử lý ở trên.
                        conflicts.append(
                            MergeConflict(
                                kind=kind,
                                candidates=(
                                    self._candidate_to_dict(cand),
                                    self._existing_to_dict(existing),
                                ),
                            )
                        )
                        placed = True
                        drop_existing.append(idx)
            if not placed:
                new_flat.append(self._candidate_to_emit(cand))
            # Apply drop_existing: các index trong drop_existing đã được
            # mark conflict → remove khỏi flat. Vì các index thuộc `flat`
            # cũ, không phải new_flat, đánh dấu bằng set.
            flat = [e for i, e in enumerate(flat) if i not in drop_existing]
            flat = new_flat

        return MergeResult(
            entities=tuple(flat),
            conflicts=tuple(conflicts),
        )

    # --- private helpers ----------------------------------------------------

    def _decide(self, cand: Candidate, existing: dict, conflicts: list[MergeConflict]) -> str:
        """Quyết định quan hệ giữa cand và existing.

        Trả: "merge" | "skip" | "conflict" | "keep_both".
        - "merge": cùng (start,end,label) → union.
        - "skip": cand bị nhập vào existing (cùng start,end,label).
        - "conflict": cùng start,end nhưng label khác không compatible.
        - "keep_both": quan hệ khác (sẽ xét thêm).
        """
        if (cand.start, cand.end) == (existing["start"], existing["end"]):
            if cand.label == existing["label"]:
                return "merge"
            if self._mapping.labels_compatible(cand.label, existing["label"]):
                return "merge"  # merger sẽ dùng preferred_label
            return "conflict"
        return "keep_both"

    def _merge_two(self, cand: Candidate, existing: dict) -> dict:
        """Union sources, lấy max priority, dùng preferred label nếu
        compatible-different-label."""
        label = existing["label"]
        if cand.label != label and self._mapping.labels_compatible(cand.label, label):
            label = self._mapping.preferred_label(cand.label, label)
        sources = sorted(set(existing.get("sources", [])) | {cand.source})
        score = max(existing.get("priority_score", 0.0), cand.priority_score)
        merged_from = sorted(set(existing.get("merged_from_labels", [])) | {cand.label})
        linking = sorted(set(existing.get("linking_candidates", ())) | set(cand.linking_candidates))
        linking_status = existing.get("linking_status") or cand.linking_status
        return {
            "text": existing["text"],
            "label": label,
            "start": existing["start"],
            "end": existing["end"],
            "sources": sources,
            "source_ids": sorted(
                set(existing.get("source_ids", []))
                | ({cand.source_id} if cand.source_id else set())
            ),
            "priority_score": score,
            "matched_alias": existing.get("matched_alias"),
            "merged_from_labels": merged_from,
            "linking_candidates": tuple(linking),
            "linking_status": linking_status,
            "review_status": "needs_review",
            "method": "merger",
        }

    def _candidate_to_emit(self, cand: Candidate) -> dict:
        return {
            "text": cand.text,
            "label": cand.label,
            "start": cand.start,
            "end": cand.end,
            "sources": [cand.source],
            "source_ids": [cand.source_id] if cand.source_id else [],
            "priority_score": cand.priority_score,
            "matched_alias": cand.matched_alias,
            "merged_from_labels": [cand.label],
            "linking_candidates": cand.linking_candidates,
            "linking_status": cand.linking_status,
            "review_status": "needs_review",
            "method": cand.source,
        }

    def _candidate_to_dict(self, cand: Candidate) -> dict:
        return {
            "text": cand.text,
            "label": cand.label,
            "start": cand.start,
            "end": cand.end,
            "sources": [cand.source],
            "source_id": cand.source_id,
        }

    def _existing_to_dict(self, existing: dict) -> dict:
        return {
            "text": existing["text"],
            "label": existing["label"],
            "start": existing["start"],
            "end": existing["end"],
            "sources": existing.get("sources", []),
            "source_ids": existing.get("source_ids", []),
        }

    def _relation(self, cand: Candidate, existing: dict) -> str:
        """Trả 'disjoint' | 'partial_overlap' | 'nested'."""
        cs, ce = cand.start, cand.end
        es, ee = existing["start"], existing["end"]
        if ce <= es or ee <= cs:
            return "disjoint"
        if cs <= es and ee <= ce:
            return "nested"  # existing nằm trong cand
        if es <= cs and ce <= ee:
            return "nested"  # cand nằm trong existing
        return "partial_overlap"
