# uv setup & workflow

## Tại sao uv?

`uv` (https://docs.astral.sh/uv/) là build tool bắt buộc của project này:
- Lock file `uv.lock` được commit để môi trường tái lập byte-for-byte.
- Extra dependency tách module: `dev`, `ner-basic`, `ner-crf`, `sentseg-models`,
  `ner-eval` — chỉ sync khi cần.
- Core runtime (`src/hcmus_nlp/` + `scripts/`) chỉ dùng Python standard
  library. Có thể chạy pipeline mặc định mà không cần `torch`/`transformers`.

## Cài uv (nếu chưa có)

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# hoặc qua pipx
pipx install uv
```

## Cài môi trường

```bash
# Core runtime (chỉ stdlib) + dev extra (pytest)
uv sync --extra dev

# Thêm optional models (transformers + torch)
uv sync --extra ner-basic

# Kết hợp nhiều extras
uv sync --extra dev --extra ner-basic
```

`uv.lock` được commit. Sau khi đổi `pyproject.toml`, chạy:

```bash
uv lock                      # regenerate uv.lock
git add uv.lock pyproject.toml
```

## Chạy pipeline

```bash
# Cách 1: qua uv run (khuyến nghị)
uv run python3 scripts/prepare_corpus.py
uv run python3 scripts/annotate_corpus.py
uv run python3 scripts/validate_corpus.py
uv run python3 scripts/export_submission.py --mode draft

# Cách 2: qua python -m
uv run python -m scripts.prepare_corpus

# Cách 3: trực tiếp (đã có scripts/_bootstrap.py)
uv run python3 scripts/prepare_corpus.py
```

Cả 3 cách đều chạy được nhờ `scripts/_bootstrap.py` thêm `<repo>/src` vào
`sys.path` khi gọi direct-script.

## Chạy test

```bash
uv run --extra dev pytest tests/         # full
uv run --extra dev pytest tests/test_volume.py -v   # 1 file
uv run --extra dev pytest -k "test_same_span"        # theo tên
```

## Cập nhật dependencies

```bash
# Sau khi sửa [project.optional-dependencies] trong pyproject.toml
uv lock

# Kiểm tra conflict
uv sync --extra dev --extra ner-basic
```

## Lock + reproducibility

`uv.lock` chứa hash từng package + URL index. Hai máy khác nhau cùng
commit sẽ resolve cùng version. Khi upstream publish version mới, lock
không tự update — phải `uv lock` thủ công.

## Troubleshooting

- **`No module named 'hcmus_nlp'`**: chạy qua `uv run` hoặc `scripts/_bootstrap`.
- **`ModuleNotFoundError: No module named 'pytest'`**: `uv sync --extra dev`.
- **`ModuleNotFoundError: No module named 'torch'`**: model extras chưa sync.
  Đây là intentional — default pipeline không cần model.

## Cấu trúc extras

| Extra | Mục đích | Khi nào dùng |
|---|---|---|
| `dev` | pytest | Test, CI |
| `ner-basic` | transformers + torch | Thử Guwen NER basic |
| `ner-crf` | + pytorch-crf | Thử Guwen NER CRF (experimental) |
| `sentseg-models` | transformers + torch | Thử Koichi sentence segmenter |
| `ner-eval` | seqeval | Token-level metric |

`uv sync` không có `--extra` → chỉ core runtime (stdlib). Pipeline mặc định
chạy được.