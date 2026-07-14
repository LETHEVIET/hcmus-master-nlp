"""Smoke test — xác nhận test scaffold chạy được và cả 2 cách gọi CLI OK.

Regression: nếu pytest config / pythonpath / _bootstrap lỗi, smoke test fail
đầu tiên. Nếu CLI chỉ chạy được 1 trong 2 cách (direct script vs -m),
test_subprocess_cli_both_modes fail.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pytest_runs():
    assert True


def test_scripts_bootstrap_importable():
    from scripts._bootstrap import ensure_src_on_path

    ensure_src_on_path()
    import hcmus_nlp  # noqa: F401


def test_hcmus_nlp_importable():
    import hcmus_nlp

    assert hcmus_nlp.__doc__ is not None


def test_repo_root_contains_pyproject(repo_root):
    assert (repo_root / "pyproject.toml").exists()
    assert (repo_root / "uv.lock").exists()
    assert (repo_root / ".python-version").exists()


def test_dataset_has_eight_works(dataset_dir, works):
    existing = sorted(p.name for p in dataset_dir.glob("*.txt"))
    # Match theo NFC normalization để tránh lệch giữa NFD/NFC Unicode.
    import unicodedata

    def nfc(s: str) -> str:
        return unicodedata.normalize("NFC", s)

    existing_nfc = {nfc(name) for name in existing}
    missing = [w for w in works if nfc(w) not in existing_nfc]
    assert not missing, f"Missing dataset files: {missing}\nFound: {existing}"


def test_subprocess_cli_both_modes(repo_root: Path):
    """Chạy `python3 scripts/X.py` và `python -m scripts.X` qua subprocess.

    Verify cả 2 cách đều hoạt động trong environment không editable-install
    (tức là đường dẫn script thẳng, không qua `pip install -e`).
    """
    import os

    env = os.environ.copy()
    # Bỏ PYTHONPATH để test pure CLI invocation.
    env.pop("PYTHONPATH", None)
    # Chạy từ repo_root để script tìm được relative paths.

    # Test 1: python3 scripts/prepare_corpus.py --help
    r1 = subprocess.run(
        [sys.executable, "scripts/prepare_corpus.py", "--help"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r1.returncode == 0, f"direct script failed: {r1.stderr}"
    assert "Build a structured monolingual" in r1.stdout or "INPUT" in r1.stdout.upper()

    # Test 2: python -m scripts.prepare_corpus --help
    r2 = subprocess.run(
        [sys.executable, "-m", "scripts.prepare_corpus", "--help"],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r2.returncode == 0, f"python -m failed: {r2.stderr}"
    assert "Build a structured monolingual" in r2.stdout or "INPUT" in r2.stdout.upper()

    # Cả 2 phải có output giống nhau (về cấu trúc argparse).
    # So sánh chỉ phần usage line (argparse in usage ra argv[0] khác nhau).
    r1_lines = [l for l in r1.stdout.split("\n") if not l.startswith("usage:")]
    r2_lines = [l for l in r2.stdout.split("\n") if not l.startswith("usage:")]
    assert r1_lines == r2_lines, (
        f"CLI outputs khác nhau:\ndirect script:\n{r1.stdout[:500]}\npython -m:\n{r2.stdout[:500]}"
    )
