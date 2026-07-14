"""Pytest config + shared fixtures.

Regression mỗi test bảo vệ:
- test_volume.py: 10 edge case heading quyển + injectivity + integration
- test_cleaning.py: inline OCR placeholder + URL/footer leak + ellipsis
- test_labels.py: mapping TOML coverage + confirmed flag fatal
- test_candidates.py: partial overlap / nested / tie / deterministic / TIME-vs-NUM
- test_doccano.py: 8 case roundtrip (pass-through / add / delete / relabel /
  duplicate-text / empty / duplicate-id / atomic)
- test_gold.py: random reproducibility + diagnostic disjoint + scale-up rule
- test_eval.py: strict P/R/F1 + boundary-only + per-label + baseline diff
- test_compliance.py: final-fail-when-unchecked + no unresolved + ID unique

Core runtime (scripts/ + src/) chỉ dùng stdlib. Test cần `dev` extra
(`uv sync --extra dev`).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Repo root = parent của tests/
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
DATASET_DIR = REPO_ROOT / "dataset"
BUILD_DIR = REPO_ROOT / "build"
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def dataset_dir() -> Path:
    return DATASET_DIR


@pytest.fixture(scope="session")
def build_dir() -> Path:
    return BUILD_DIR


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def works() -> list[str]:
    """Danh sách 8 tác phẩm raw."""
    return [
        "北史_full.txt",
        "北齊書_full.txt",
        "Chư Phiên Chí - 諸蕃志_full.txt",
        "舊唐書_full.txt",
        "舊五代史_full.txt",
        "Đông Quan Hán Ký - 東觀漢記_(四庫全書本)_full.txt",
        "漢書_full.txt",
        "後漢書_full.txt",
    ]


@pytest.fixture(scope="session")
def sample_record(dataset_dir: Path, works: list[str]) -> dict:
    """Lấy 1 record đầu tiên từ corpus đã prepare (nếu có) hoặc 1 raw line.

    Trả về dict có shape tối thiểu để test dùng. Dùng cho các test không cần
    full pipeline output.
    """
    corpus_path = BUILD_DIR / "corpus.jsonl"
    if corpus_path.exists():
        with corpus_path.open(encoding="utf-8") as handle:
            return json.loads(next(handle))
    # Fallback: đọc 1 raw line đầu từ 漢書_full.txt
    fallback = dataset_dir / "漢書_full.txt"
    text = fallback.read_text(encoding="utf-8").splitlines()
    for line in text:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return {
                "id": "fallback-1",
                "title": "漢書",
                "text": stripped,
                "volume": None,
                "section": None,
                "source_file": "漢書_full.txt",
                "source_line": 0,
            }
    raise RuntimeError("No usable fixture text")


@pytest.fixture(scope="session")
def sample_records(n: int = 5, dataset_dir: Path = DATASET_DIR) -> list[dict]:
    """Lấy n record đầu từ corpus.jsonl nếu có; dùng cho integration test."""
    corpus_path = BUILD_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        return []
    records: list[dict] = []
    with corpus_path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def real_mapping_path(repo_root: Path) -> Path:
    """Path tới file mapping.toml trong repo."""
    return repo_root / "config" / "mapping.toml"
