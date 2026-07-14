"""CHGIS loader (Phase F3).

CHGIS có nhiều format phát hành (CSV, shapefile, SQLite). Loader này dùng
schema-detector: thử 1-2 header variants phổ biến. Không nhận → IngestError.

KHÔNG auto-download trong session này (license cần verify từng phiên bản).
User ingest bằng `scripts/build_kb.py ingest-chgis --input PATH/TO/places.csv`.

Period filter: KHÔNG lọc cứng theo `record["period"]` (plan v5 nhận xét #10).
Period chỉ là prior trong gazetteer runtime; nếu nhiều candidate CHGIS hợp
lệ → giữ linking_candidates để review.
"""

from __future__ import annotations

import csv
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hcmus_nlp._weights import KB_FULL
from hcmus_nlp.candidates import Trie
from hcmus_nlp.kb.manifest import KBManifest, sha256_of_file
from hcmus_nlp.source_base import AnnotationContext, Candidate, SourceKind


class CHGISError(RuntimeError):
    """Lỗi khi load/ingest CHGIS."""


@dataclass(frozen=True)
class CHGISPlace:
    place_id: str
    name: str
    begin: int  # năm bắt đầu
    end: int  # năm kết thúc
    type: str
    parent_id: str | None
    aliases: tuple[str, ...] = ()


# Header variants phổ biến của CHGIS CSV. Detector thử từng cái.
CHGIS_HEADER_VARIANTS: tuple[dict[str, str], ...] = (
    {
        "place_id": "PLACE_ID",
        "name": "NAME_CN",
        "begin": "BEGIN",
        "end": "END",
        "type": "TYPE",
        "parent": "PARENT_ID",
    },
    {
        "place_id": "id",
        "name": "name",
        "begin": "begin",
        "end": "end",
        "type": "type",
        "parent": "parent",
    },
    {
        "place_id": "CHGIS_ID",
        "name": "PLACE_NAME",
        "begin": "START_YEAR",
        "end": "END_YEAR",
        "type": "ADMIN_TYPE",
        "parent": "PARENT_CHGIS_ID",
    },
)


def detect_header(headers: list[str]) -> dict[str, str] | None:
    """Trả header map khớp với 1 trong variants, hoặc None.

    So sánh case-insensitive: cả variant values và headers được uppercase
    trước khi match.
    """
    headers_upper = {h.upper(): h for h in headers}
    for variant in CHGIS_HEADER_VARIANTS:
        if all(v.upper() in headers_upper for v in variant.values()):
            return {k: headers_upper[v.upper()] for k, v in variant.items()}
    return None


def load_places(csv_path: Path) -> list[CHGISPlace]:
    """Đọc CSV, dùng header detector."""
    if not csv_path.exists():
        raise CHGISError(f"CHGIS CSV not found: {csv_path}")
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise CHGISError("CSV không có header")
        header_map = detect_header(list(reader.fieldnames))
        if header_map is None:
            raise CHGISError(
                f"Không nhận diện được schema CHGIS. Header thực tế: {reader.fieldnames}. "
                "Có thể cần thêm variant vào CHGIS_HEADER_VARIANTS."
            )
        places: list[CHGISPlace] = []
        for row in reader:
            try:
                places.append(
                    CHGISPlace(
                        place_id=row[header_map["place_id"]],
                        name=row[header_map["name"]],
                        begin=int(row[header_map["begin"]]),
                        end=int(row[header_map["end"]]),
                        type=row[header_map["type"]],
                        parent_id=row.get(header_map["parent"]),
                    )
                )
            except (ValueError, KeyError) as e:
                # Skip row lỗi nhưng log warning.
                print(f"[chgis] skip row: {e}", file=__import__("sys").stderr)
                continue
    return places


def build_chgis_cache(
    csv_path: Path,
    cache_path: Path,
    *,
    version: str,
    source_url: str | None,
    license: str,
) -> KBManifest:
    """Ingest CHGIS CSV ra SQLite cache + manifest."""
    places = load_places(csv_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_path.unlink()
    conn = sqlite3.connect(cache_path)
    try:
        conn.execute(
            """
            CREATE TABLE place (
                place_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                begin INTEGER NOT NULL,
                end INTEGER NOT NULL,
                type TEXT,
                parent_id TEXT
            )
            """
        )
        conn.execute("CREATE INDEX idx_place_name ON place(name)")
        conn.execute("CREATE INDEX idx_place_begin ON place(begin)")
        conn.execute("CREATE INDEX idx_place_end ON place(end)")
        sorted_places = sorted(places, key=lambda p: (p.name, p.begin))
        for p in sorted_places:
            conn.execute(
                "INSERT INTO place VALUES (?, ?, ?, ?, ?, ?)",
                (p.place_id, p.name, p.begin, p.end, p.type, p.parent_id),
            )
        conn.commit()
    finally:
        conn.close()

    sha, size = sha256_of_file(cache_path)
    manifest = KBManifest(
        name="chgis",
        version=version,
        source_url=source_url,
        license=license,
        file_sha256=sha,
        file_size=size,
        row_counts={"place": len(places)},
    )
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    manifest.write(manifest_path)
    return manifest


class CHGISPlaceSource:
    """Source adapter cho CHGIS places.

    Period là prior, không filter cứng. Mỗi place match → 1 candidate với
    linking_candidates chứa tất cả place_id cùng surface trong khoảng thời
    gian plausible.
    """

    name = "chgis"
    kind = SourceKind.KB

    def __init__(self, places: list[CHGISPlace]):
        self._places = places
        self._trie = Trie()
        # Group places theo (name, begin-year) để detect linking ambiguity.
        for p in sorted(places, key=lambda x: (x.name, x.begin)):
            self._trie.insert(p.name, "LOCATION", alias=p.place_id)

    def candidates(self, text: str, ctx: AnnotationContext) -> Iterable[Candidate]:
        # Group matches theo (start, end, surface).
        grouped: dict[tuple[int, int, str], list[tuple[int, str]]] = {}
        for start, end, term, label, alias in self._trie.find_all(text):
            if term != label and label == "LOCATION":
                # alias ở đây là place_id.
                grouped.setdefault((start, end, term), []).append((0, alias))

        for (start, end, term), linking_ids in grouped.items():
            linking = tuple(pid for _, pid in sorted(linking_ids))
            status = "ambiguous" if len(linking) > 1 else "resolved"
            yield Candidate(
                text=term,
                label="LOCATION",
                start=start,
                end=end,
                source=self.name,
                source_id=f"chgis:{linking[0]}",
                priority_score=KB_FULL,
                matched_alias=term,
                linking_candidates=linking,
                linking_status=status,
            )

    def available(self) -> bool:
        return len(self._places) > 0


__all__ = [
    "CHGISError",
    "CHGISPlace",
    "CHGISPlaceSource",
    "build_chgis_cache",
    "detect_header",
    "load_places",
]
