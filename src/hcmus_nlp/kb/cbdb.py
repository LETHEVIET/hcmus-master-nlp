"""CBDB loader (Phase F2).

CBDB SQLite có schema gồm nhiều bảng. Loader introspect schema thực tế thay
vì assume schema cố định — robust với các phiên bản khác nhau.

KHÔNG auto-download trong session này (license CC BY-NC-SA 4.0). User cần
tải thủ công và ingest bằng `scripts/build_kb.py ingest-cbdb --input PATH`.

Runtime filter: 1-char name từ CBDB bị skip (min_len >= 2) vì ambiguous
với common nouns.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from hcmus_nlp._weights import KB_FULL
from hcmus_nlp.candidates import Trie
from hcmus_nlp.kb.manifest import KBManifest, sha256_of_file
from hcmus_nlp.source_base import AnnotationContext, Candidate, SourceKind


class CBDBError(RuntimeError):
    """Lỗi khi load/ingest CBDB."""


@dataclass(frozen=True)
class CBDBPerson:
    person_id: int
    name: str
    surname: str | None
    courtesy_name: str | None


# Bảng CBDB phổ biến (introspect thực tế khi ingest).
CBDB_PERSON_TABLE = "BIOG_MAIN"
CBDB_ALT_NAME_TABLE = "ALTNAME_DATA"
CBDB_OFFICE_TABLE = "OFFICE_CODES"
CBDB_TEXT_TABLE = "TEXT_CODES"


def introspect_schema(sqlite_path: Path) -> dict[str, list[str]]:
    """Trả dict table_name -> [column, ...]."""
    if not sqlite_path.exists():
        raise CBDBError(f"CBDB SQLite not found: {sqlite_path}")
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cur.fetchall()]
        schema: dict[str, list[str]] = {}
        for tbl in tables:
            cur = conn.execute(f"PRAGMA table_info({tbl})")
            cols = [row[1] for row in cur.fetchall()]
            schema[tbl] = cols
    finally:
        conn.close()
    return schema


def load_persons(sqlite_path: Path, *, min_len: int = 2) -> tuple[list[CBDBPerson], int]:
    """Load persons từ BIOG_MAIN. Filter min_len >= 2 cho name."""
    schema = introspect_schema(sqlite_path)
    if CBDB_PERSON_TABLE not in schema:
        raise CBDBError(
            f"CBDB schema không có bảng {CBDB_PERSON_TABLE!r}. Có: {sorted(schema.keys())[:10]}..."
        )
    cols = schema[CBDB_PERSON_TABLE]
    name_col = "c_name_chn" if "c_name_chn" in cols else cols[2] if len(cols) > 2 else cols[-1]
    surname_col = "c_surname_chn" if "c_surname_chn" in cols else None
    courtesy_col = "c_courtesy_chn" if "c_courtesy_chn" in cols else None

    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        select_cols = [f"{CBDB_PERSON_TABLE}.{name_col}"]
        if surname_col:
            select_cols.append(f"{CBDB_PERSON_TABLE}.{surname_col}")
        if courtesy_col:
            select_cols.append(f"{CBDB_PERSON_TABLE}.{courtesy_col}")
        # c_personid assumed column 0.
        select_cols.insert(0, f"{CBDB_PERSON_TABLE}.c_personid")
        sql = f"SELECT {', '.join(select_cols)} FROM {CBDB_PERSON_TABLE}"
        cur = conn.execute(sql)
        persons: list[CBDBPerson] = []
        total = 0
        for row in cur:
            total += 1
            person_id = row[0]
            name = row[1] or ""
            surname = row[2] if len(row) > 2 and surname_col else None
            courtesy = row[3] if len(row) > 3 and courtesy_col else None
            if not name or len(name) < min_len:
                continue
            persons.append(
                CBDBPerson(
                    person_id=person_id,
                    name=name,
                    surname=surname,
                    courtesy_name=courtesy,
                )
            )
    finally:
        conn.close()
    return persons, total


def build_cbdb_cache(
    sqlite_path: Path,
    cache_path: Path,
    *,
    version: str,
    source_url: str | None,
    license: str = "cc-by-nc-sa-4.0",
) -> KBManifest:
    """Ingest CBDB SQLite ra cache SQLite riêng + manifest.

    Cache giữ schema: person(person_id, name, surname, courtesy).
    """
    persons, total = load_persons(sqlite_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Tạo SQLite cache.
    if cache_path.exists():
        cache_path.unlink()
    conn = sqlite3.connect(cache_path)
    try:
        conn.execute(
            """
            CREATE TABLE person (
                person_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                surname TEXT,
                courtesy TEXT
            )
            """
        )
        conn.execute("CREATE INDEX idx_person_name ON person(name)")
        for p in sorted(persons, key=lambda x: (x.name, x.person_id)):
            conn.execute(
                "INSERT INTO person VALUES (?, ?, ?, ?)",
                (p.person_id, p.name, p.surname, p.courtesy_name),
            )
        conn.commit()
    finally:
        conn.close()

    sha, size = sha256_of_file(cache_path)
    manifest = KBManifest(
        name="cbdb",
        version=version,
        source_url=source_url,
        license=license,
        file_sha256=sha,
        file_size=size,
        row_counts={"person_total": total, "person_kept": len(persons)},
    )
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    manifest.write(manifest_path)
    return manifest


class CBDBPersonSource:
    """Source adapter cho CBDB person names."""

    name = "cbdb"
    kind = SourceKind.KB

    def __init__(self, persons: list[CBDBPerson]):
        self._persons = persons
        self._trie = Trie()
        for p in sorted(persons, key=lambda x: (x.name, x.person_id)):
            self._trie.insert(p.name, "PERSON", alias=p.name)
            if p.courtesy_name and len(p.courtesy_name) >= 2:
                self._trie.insert(p.courtesy_name, "PERSON", alias=p.courtesy_name)

    def candidates(self, text: str, ctx: AnnotationContext) -> Iterable[Candidate]:
        for start, end, term, label, alias in self._trie.find_all(text):
            yield Candidate(
                text=term,
                label=label,
                start=start,
                end=end,
                source=self.name,
                source_id=f"cbdb:{alias}",
                priority_score=KB_FULL,
                matched_alias=alias,
            )

    def available(self) -> bool:
        return len(self._persons) > 0


__all__ = [
    "CBDBError",
    "CBDBPerson",
    "CBDBPersonSource",
    "build_cbdb_cache",
    "introspect_schema",
    "load_persons",
]
