"""Thin entrypoints package.

Mỗi script có thể chạy theo hai cách:
- `python3 scripts/<name>.py ...` (cách cũ, README cũ dùng) — yêu cầu _bootstrap
- `python -m scripts.<name> ...` (cách mới, chuẩn package) — dùng sys.path mặc định

Khi chạy trực tiếp, entrypoint phải gọi scripts._bootstrap trước khi import
`hcmus_nlp.*` (xem scripts/_bootstrap.py).
"""
