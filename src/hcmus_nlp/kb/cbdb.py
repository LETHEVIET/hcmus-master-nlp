"""CBDB offline ingest and PERSON candidate source.

The official CBDB SQLite schema changes between releases.  This module reads
the documented relational tables through schema introspection and writes a
small, stable cache used by the annotation pipeline.  It never downloads data
and never mutates the source database.

The cache contains primary Chinese names plus aliases from ``ALTNAME_DATA``.
One-character names, placeholders, glyph-composition strings, and non-Han
values are rejected because dictionary matching them has very low precision.
"""

from __future__ import annotations

import os
import sqlite3
import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from hcmus_nlp._weights import KB_FULL
from hcmus_nlp.candidates import Trie
from hcmus_nlp.kb.manifest import KBManifest, load_manifest, sha256_of_file
from hcmus_nlp.source_base import AnnotationContext, Candidate, SourceKind


class CBDBError(RuntimeError):
    """Raised when a CBDB source/cache is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class CBDBPerson:
    """Normalized person record retained for API compatibility and tests."""

    person_id: int
    name: str
    surname: str | None = None
    courtesy_name: str | None = None
    aliases: tuple[str, ...] = field(default_factory=tuple)
    dynasty_code: int | None = None
    birth_year: int | None = None
    death_year: int | None = None


@dataclass(frozen=True)
class CBDBNameEntry:
    """One surface form linked to one or more CBDB person IDs."""

    name: str
    person_ids: tuple[int, ...]
    kinds: tuple[str, ...] = field(default_factory=tuple)
    dynasty_codes: tuple[int | None, ...] = field(default_factory=tuple)


CBDB_PERSON_TABLE = "BIOG_MAIN"
CBDB_ALT_NAME_TABLE = "ALTNAME_DATA"

# Personal-name-like categories from ALTNAME_CODES. Titles, posthumous names,
# temple names, enfeoffments, honorifics, and unknown values are excluded: in
# running text they behave as TITLE or common vocabulary rather than PERSON.
CBDB_PERSONAL_ALT_NAME_TYPE_CODES = frozenset({3, 4, 5, 9, 10, 13, 17, 19, 20, 21})

_PLACEHOLDER_NAMES = {
    "不詳",
    "佚名",
    "失名",
    "未詳",
    "某人",
    "某某",
    "無名",
    "無名氏",
    "闕名",
}
_CHINESE_NUMERALS = frozenset("零〇一二三四五六七八九十百千萬万億亿兩两")
_PERSON_TITLE_SUFFIXES = (
    "夫人",
    "先生",
    "國師",
    "国师",
    "可汗",
    "皇帝",
    "皇后",
)


def _quoted_identifier(value: str) -> str:
    """Quote an identifier obtained from SQLite schema introspection."""
    return '"' + value.replace('"', '""') + '"'


def _is_han_character(char: str) -> bool:
    cp = ord(char)
    return (
        0x3400 <= cp <= 0x4DBF
        or 0x4E00 <= cp <= 0x9FFF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2FA1F
        or 0x30000 <= cp <= 0x323AF
    )


def normalize_cbdb_name(
    value: object,
    *,
    min_len: int = 2,
    max_len: int = 6,
) -> str | None:
    """Return a safe Han-only CBDB surface form, or ``None``.

    CBDB also stores romanization, missing-data markers, and glyph composition
    such as ``(放+山)㟚``.  Those values must not enter a surface-form matcher.
    """
    if not isinstance(value, str):
        return None
    name = unicodedata.normalize("NFC", value).strip()
    if name in _PLACEHOLDER_NAMES or not min_len <= len(name) <= max_len:
        return None
    if not all(_is_han_character(char) for char in name):
        return None
    if all(char in _CHINESE_NUMERALS for char in name):
        return None
    if name.endswith(_PERSON_TITLE_SUFFIXES):
        return None
    return name


def _load_nonperson_surfaces(
    conn: sqlite3.Connection,
    schema: dict[str, list[str]],
) -> set[str]:
    """Load era names that are especially dangerous as person surfaces."""
    blocked: set[str] = set()
    if "NIAN_HAO" not in schema:
        return blocked
    columns = set(schema["NIAN_HAO"])
    if not {"c_dynasty_chn", "c_nianhao_chn"}.issubset(columns):
        return blocked
    for dynasty, era in conn.execute("SELECT c_dynasty_chn, c_nianhao_chn FROM NIAN_HAO"):
        normalized_era = normalize_cbdb_name(era)
        normalized_dynasty = normalize_cbdb_name(dynasty, min_len=1)
        if normalized_era:
            blocked.add(normalized_era)
            if normalized_dynasty:
                combined = normalize_cbdb_name(normalized_dynasty + normalized_era)
                if combined:
                    blocked.add(combined)
    return blocked


def introspect_schema(sqlite_path: Path) -> dict[str, list[str]]:
    """Return ``table_name -> columns`` from a read-only SQLite database."""
    if not sqlite_path.exists():
        raise CBDBError(f"CBDB SQLite not found: {sqlite_path}")
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = [
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        ]
        return {
            table: [
                row[1] for row in conn.execute(f"PRAGMA table_info({_quoted_identifier(table)})")
            ]
            for table in tables
        }
    except sqlite3.DatabaseError as exc:
        raise CBDBError(f"Cannot introspect CBDB SQLite {sqlite_path}: {exc}") from exc
    finally:
        conn.close()


def _require_columns(
    schema: dict[str, list[str]],
    table: str,
    required: tuple[str, ...],
) -> None:
    if table not in schema:
        raise CBDBError(f"CBDB schema has no table {table!r}; available: {sorted(schema)[:12]}")
    missing = [column for column in required if column not in schema[table]]
    if missing:
        raise CBDBError(f"CBDB table {table!r} is missing columns: {missing}")


def load_persons(sqlite_path: Path, *, min_len: int = 2) -> tuple[list[CBDBPerson], int]:
    """Load normalized primary names from ``BIOG_MAIN``.

    This public helper is intentionally simple.  Production ingest additionally
    reads aliases and writes them to the normalized cache.
    """
    schema = introspect_schema(sqlite_path)
    _require_columns(schema, CBDB_PERSON_TABLE, ("c_personid", "c_name_chn"))
    columns = set(schema[CBDB_PERSON_TABLE])
    optional = {
        "surname": "c_surname_chn",
        "dynasty": "c_dy",
        "birth": "c_birthyear",
        "death": "c_deathyear",
    }
    selected = ["c_personid", "c_name_chn"] + [
        column for column in optional.values() if column in columns
    ]
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT "
            + ", ".join(_quoted_identifier(column) for column in selected)
            + f" FROM {_quoted_identifier(CBDB_PERSON_TABLE)} ORDER BY c_personid"
        )
        people: list[CBDBPerson] = []
        total = 0
        positions = {column: index for index, column in enumerate(selected)}
        for row in rows:
            total += 1
            name = normalize_cbdb_name(row[positions["c_name_chn"]], min_len=min_len)
            if name is None:
                continue
            people.append(
                CBDBPerson(
                    person_id=int(row[positions["c_personid"]]),
                    name=name,
                    surname=(
                        row[positions["c_surname_chn"]] if "c_surname_chn" in positions else None
                    ),
                    dynasty_code=(row[positions["c_dy"]] if "c_dy" in positions else None),
                    birth_year=(
                        row[positions["c_birthyear"]] if "c_birthyear" in positions else None
                    ),
                    death_year=(
                        row[positions["c_deathyear"]] if "c_deathyear" in positions else None
                    ),
                )
            )
    finally:
        conn.close()
    return people, total


def _source_person_rows(conn: sqlite3.Connection, columns: set[str]) -> Iterator[tuple]:
    optional = [
        column
        for column in ("c_surname_chn", "c_dy", "c_birthyear", "c_deathyear")
        if column in columns
    ]
    selected = ["c_personid", "c_name_chn", *optional]
    sql = (
        "SELECT "
        + ", ".join(_quoted_identifier(column) for column in selected)
        + f" FROM {_quoted_identifier(CBDB_PERSON_TABLE)} ORDER BY c_personid"
    )
    yield from conn.execute(sql)


def build_cbdb_cache(
    sqlite_path: Path,
    cache_path: Path,
    *,
    version: str,
    source_url: str | None,
    license: str = "cc-by-nc-sa-4.0",
    min_len: int = 2,
    max_len: int = 6,
) -> KBManifest:
    """Build a normalized, atomic CBDB cache with primary and alternate names."""
    schema = introspect_schema(sqlite_path)
    _require_columns(schema, CBDB_PERSON_TABLE, ("c_personid", "c_name_chn"))

    source = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    target = sqlite3.connect(tmp_path)

    person_total = 0
    person_kept = 0
    primary_kept = 0
    alias_total = 0
    alias_allowed_type = 0
    alias_kept = 0
    blocked_nonperson = 0
    kept_ids: set[int] = set()

    try:
        target.executescript(
            """
            PRAGMA journal_mode=DELETE;
            CREATE TABLE person (
                person_id INTEGER PRIMARY KEY,
                primary_name TEXT NOT NULL,
                surname TEXT,
                dynasty_code INTEGER,
                birth_year INTEGER,
                death_year INTEGER
            );
            CREATE TABLE name (
                name TEXT NOT NULL,
                person_id INTEGER NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('primary', 'alias')),
                PRIMARY KEY (name, person_id, kind)
            );
            """
        )

        blocked_surfaces = _load_nonperson_surfaces(source, schema)
        person_columns = set(schema[CBDB_PERSON_TABLE])
        selected = ["c_personid", "c_name_chn"] + [
            column
            for column in ("c_surname_chn", "c_dy", "c_birthyear", "c_deathyear")
            if column in person_columns
        ]
        positions = {column: index for index, column in enumerate(selected)}
        for row in _source_person_rows(source, person_columns):
            person_total += 1
            name = normalize_cbdb_name(
                row[positions["c_name_chn"]], min_len=min_len, max_len=max_len
            )
            if name is None:
                continue
            if name in blocked_surfaces:
                blocked_nonperson += 1
                continue
            person_id = int(row[positions["c_personid"]])
            surname = row[positions["c_surname_chn"]] if "c_surname_chn" in positions else None
            dynasty = row[positions["c_dy"]] if "c_dy" in positions else None
            birth = row[positions["c_birthyear"]] if "c_birthyear" in positions else None
            death = row[positions["c_deathyear"]] if "c_deathyear" in positions else None
            target.execute(
                "INSERT INTO person VALUES (?, ?, ?, ?, ?, ?)",
                (person_id, name, surname, dynasty, birth, death),
            )
            target.execute("INSERT INTO name VALUES (?, ?, 'primary')", (name, person_id))
            kept_ids.add(person_id)
            person_kept += 1
            primary_kept += 1

        if CBDB_ALT_NAME_TABLE in schema:
            _require_columns(
                schema,
                CBDB_ALT_NAME_TABLE,
                ("c_personid", "c_alt_name_chn"),
            )
            alt_has_type = "c_alt_name_type_code" in schema[CBDB_ALT_NAME_TABLE]
            alt_columns = "c_personid, c_alt_name_chn"
            if alt_has_type:
                alt_columns += ", c_alt_name_type_code"
            alt_sql = (
                f"SELECT {alt_columns} FROM "
                f"{_quoted_identifier(CBDB_ALT_NAME_TABLE)} "
                "ORDER BY c_personid, c_alt_name_chn"
            )
            for row in source.execute(alt_sql):
                person_id, raw_alias = row[0], row[1]
                alias_total += 1
                if alt_has_type and row[2] not in CBDB_PERSONAL_ALT_NAME_TYPE_CODES:
                    continue
                alias_allowed_type += 1
                if person_id not in kept_ids:
                    continue
                alias = normalize_cbdb_name(raw_alias, min_len=min_len, max_len=max_len)
                if alias is None:
                    continue
                if alias in blocked_surfaces:
                    blocked_nonperson += 1
                    continue
                before = target.total_changes
                target.execute(
                    "INSERT OR IGNORE INTO name VALUES (?, ?, 'alias')",
                    (alias, int(person_id)),
                )
                if target.total_changes > before:
                    alias_kept += 1

        target.executescript(
            """
            CREATE INDEX idx_name_surface ON name(name);
            CREATE INDEX idx_name_person ON name(person_id);
            """
        )
        target.commit()
        unique_names = int(target.execute("SELECT COUNT(DISTINCT name) FROM name").fetchone()[0])
    except (sqlite3.DatabaseError, OSError) as exc:
        target.rollback()
        raise CBDBError(f"Failed to build CBDB cache: {exc}") from exc
    finally:
        source.close()
        target.close()

    os.replace(tmp_path, cache_path)
    sha, size = sha256_of_file(cache_path)
    manifest = KBManifest(
        name="cbdb",
        version=version,
        source_url=source_url,
        license=license,
        file_sha256=sha,
        file_size=size,
        row_counts={
            "person_total": person_total,
            "person_kept": person_kept,
            "primary_names": primary_kept,
            "alias_total": alias_total,
            "alias_allowed_type": alias_allowed_type,
            "alias_names": alias_kept,
            "unique_names": unique_names,
            "blocked_nonperson": blocked_nonperson,
        },
    )
    manifest.write(cache_path.with_suffix(cache_path.suffix + ".manifest.json"))
    return manifest


def load_cbdb_cache(
    cache_path: Path,
    *,
    max_ambiguity: int = 10,
) -> tuple[list[CBDBNameEntry], KBManifest]:
    """Verify and load name surfaces from a normalized CBDB cache.

    Names linked to more than ``max_ambiguity`` people are omitted because a
    bare dictionary hit cannot link them usefully and is likely to be noisy.
    """
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".manifest.json")
    if not cache_path.exists():
        raise CBDBError(f"CBDB cache not found: {cache_path}")
    if not manifest_path.exists():
        raise CBDBError(f"CBDB cache manifest not found: {manifest_path}")
    manifest = load_manifest(manifest_path)
    sha, size = sha256_of_file(cache_path)
    if sha != manifest.file_sha256 or size != manifest.file_size:
        raise CBDBError(f"CBDB cache SHA-256 mismatch: expected {manifest.file_sha256}, got {sha}")

    schema = introspect_schema(cache_path)
    _require_columns(schema, "name", ("name", "person_id", "kind"))
    conn = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    entries: list[CBDBNameEntry] = []
    try:
        current_name: str | None = None
        ids: set[int] = set()
        kinds: set[str] = set()
        dynasties: dict[int, int | None] = {}

        def emit() -> None:
            if current_name is None or not ids or len(ids) > max_ambiguity:
                return
            sorted_ids = tuple(sorted(ids))
            entries.append(
                CBDBNameEntry(
                    name=current_name,
                    person_ids=sorted_ids,
                    kinds=tuple(sorted(kinds)),
                    dynasty_codes=tuple(dynasties.get(person_id) for person_id in sorted_ids),
                )
            )

        rows = conn.execute(
            """
            SELECT n.name, n.person_id, n.kind, p.dynasty_code
            FROM name AS n
            JOIN person AS p ON p.person_id = n.person_id
            ORDER BY n.name, n.person_id, n.kind
            """
        )
        for surface, person_id, kind, dynasty_code in rows:
            if current_name is not None and surface != current_name:
                emit()
                ids.clear()
                kinds.clear()
                dynasties.clear()
            current_name = surface
            ids.add(int(person_id))
            kinds.add(str(kind))
            dynasties[int(person_id)] = dynasty_code
        emit()
    except sqlite3.DatabaseError as exc:
        raise CBDBError(f"Cannot load CBDB cache {cache_path}: {exc}") from exc
    finally:
        conn.close()
    return entries, manifest


class CBDBPersonSource:
    """Dictionary source for CBDB person names and aliases."""

    name = "cbdb"
    kind = SourceKind.KB

    _CUES_BEFORE = frozenset("帝王公侯君后妃氏姓名召拜遣命殺見謂將")
    _CUES_AFTER = ("曰", "傳", "字", "卒", "死", "者")

    # Work-level period is a prior, not a hard historical truth: chronicles
    # often mention earlier people. ``prefer`` narrows linking IDs when a
    # compatible record exists, and requires explicit person context otherwise.
    _PERIOD_DYNASTY_CODES = {
        "西漢": frozenset({2, 29, 83}),
        "東漢": frozenset({2, 25, 83}),
        "南北朝": frozenset(
            {
                4,
                23,
                24,
                27,
                28,
                30,
                31,
                32,
                35,
                37,
                39,
                40,
                41,
                44,
                50,
                51,
                56,
                60,
                62,
                63,
                64,
                65,
                68,
                69,
                70,
                71,
                72,
                73,
                74,
                76,
                82,
                87,
            }
        ),
        "五代": frozenset({7, 8, 9, 10, 11, 12, 13, 34, 36, 38, 47, 48, 49, 52, 55, 66, 75}),
        "唐": frozenset({6, 77}),
        "宋": frozenset({15}),
    }

    def __init__(
        self,
        entries: Iterable[CBDBNameEntry | CBDBPerson],
        *,
        short_name_policy: str = "exclude",
        period_policy: str = "strict",
    ) -> None:
        if short_name_policy not in {"all", "context", "exclude"}:
            raise ValueError("short_name_policy must be one of: all, context, exclude")
        if period_policy not in {"off", "prefer", "strict"}:
            raise ValueError("period_policy must be one of: off, prefer, strict")
        grouped: dict[str, dict[int, int | None]] = {}
        for entry in entries:
            if isinstance(entry, CBDBPerson):
                surfaces = (entry.name, *entry.aliases)
                if entry.courtesy_name:
                    surfaces = (*surfaces, entry.courtesy_name)
                for surface in surfaces:
                    normalized = normalize_cbdb_name(surface)
                    if normalized:
                        grouped.setdefault(normalized, {})[entry.person_id] = entry.dynasty_code
            else:
                dynasty_codes = entry.dynasty_codes or (None,) * len(entry.person_ids)
                for person_id, dynasty_code in zip(entry.person_ids, dynasty_codes, strict=True):
                    grouped.setdefault(entry.name, {})[person_id] = dynasty_code

        self._short_name_policy = short_name_policy
        self._period_policy = period_policy
        self._entry_count = len(grouped)
        self._trie = Trie()
        for surface, people in sorted(grouped.items()):
            payload = ",".join(
                f"{person_id}@{'' if dynasty is None else dynasty}"
                for person_id, dynasty in sorted(people.items())
            )
            self._trie.insert(surface, "PERSON", alias=payload)

    @classmethod
    def from_cache(
        cls,
        cache_path: Path,
        *,
        max_ambiguity: int = 10,
        short_name_policy: str = "exclude",
        period_policy: str = "strict",
    ) -> "CBDBPersonSource":
        entries, _manifest = load_cbdb_cache(cache_path, max_ambiguity=max_ambiguity)
        return cls(
            entries,
            short_name_policy=short_name_policy,
            period_policy=period_policy,
        )

    def candidates(self, text: str, ctx: AnnotationContext) -> Iterable[Candidate]:
        for start, end, term, label, encoded_people in self._trie.find_all(text):
            if len(term) == 2:
                if self._short_name_policy == "exclude":
                    continue
                if self._short_name_policy == "context" and not self._has_short_name_context(
                    text, start, end
                ):
                    continue
            people: list[tuple[int, int | None]] = []
            for encoded in encoded_people.split(","):
                raw_id, raw_dynasty = encoded.split("@", 1)
                people.append((int(raw_id), int(raw_dynasty) if raw_dynasty else None))
            people = self._apply_period_policy(people, ctx.period, text, start, end)
            if not people:
                continue
            ids = tuple(f"cbdb:{person_id}" for person_id, _dynasty in people)
            yield Candidate(
                text=text[start:end],
                label=label,
                start=start,
                end=end,
                source=self.name,
                source_id=ids[0] if len(ids) == 1 else f"cbdb:name:{term}",
                priority_score=KB_FULL,
                matched_alias=term,
                linking_candidates=ids,
                linking_status="linked" if len(ids) == 1 else "ambiguous",
            )

    @classmethod
    def _has_short_name_context(cls, text: str, start: int, end: int) -> bool:
        before = text[start - 1] if start > 0 else ""
        after = text[end : end + 1]
        if before in cls._CUES_BEFORE:
            return True
        if any(text.startswith(cue, end) for cue in cls._CUES_AFTER):
            return True
        # Enumeration in biographies: 張三、李四 or punctuation-delimited name.
        if before in "，。、；：" and after in "，。、；：":
            return True
        return False

    @classmethod
    def _has_person_context(cls, text: str, start: int, end: int) -> bool:
        if cls._has_short_name_context(text, start, end):
            return True
        before = text[max(0, start - 3) : start]
        after = text[end : min(len(text), end + 2)]
        title_suffixes = ("將軍", "太守", "刺史", "丞相", "尚書", "侍郎", "校尉", "縣令")
        action_prefixes = ("為", "序", "討", "帥", "奔", "降", "薨", "卒", "曰")
        return before.endswith(title_suffixes) or after.startswith(action_prefixes)

    def _apply_period_policy(
        self,
        people: list[tuple[int, int | None]],
        period: str | None,
        text: str,
        start: int,
        end: int,
    ) -> list[tuple[int, int | None]]:
        if self._period_policy == "off" or not period:
            return people
        allowed = self._PERIOD_DYNASTY_CODES.get(period)
        if not allowed:
            return people
        compatible = [person for person in people if person[1] in allowed]
        if compatible:
            return compatible
        if self._period_policy == "strict":
            return []
        # ``prefer`` still permits a cross-period reference, but only when the
        # surrounding text strongly looks like a person mention.
        return people if self._has_person_context(text, start, end) else []

    def available(self) -> bool:
        return self._entry_count > 0


__all__ = [
    "CBDBError",
    "CBDBNameEntry",
    "CBDB_PERSONAL_ALT_NAME_TYPE_CODES",
    "CBDBPerson",
    "CBDBPersonSource",
    "build_cbdb_cache",
    "introspect_schema",
    "load_cbdb_cache",
    "load_persons",
    "normalize_cbdb_name",
]
