"""Bootstrap cho direct-script mode.

Khi chạy `python3 scripts/<name>.py`, file entrypoint không tự động có `src/` trong
sys.path. Module này thêm `<repo>/src` vào sys.path **một lần** để cho phép
`import hcmus_nlp`.

Cách dùng đầu mỗi entrypoint:

    from scripts._bootstrap import ensure_src_on_path
    ensure_src_on_path()

    from hcmus_nlp import ...   # OK ở cả `python3 scripts/X.py` và `python -m scripts.X`

Khi gọi qua `python -m scripts.X`, sys.path đã có `src/` (qua pyproject.toml
pythonpath / editable install) nên hàm này không cần thêm gì — chỉ no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BOOTSTRAPPED = False


def ensure_src_on_path() -> None:
    """Thêm `<repo>/src` vào sys.path nếu chưa có. Idempotent."""
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    # scripts/_bootstrap.py → scripts/ → <repo> → src/
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    _BOOTSTRAPPED = True
