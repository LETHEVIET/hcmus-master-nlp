# External KB sources

## CBDB — China Biographical Database

**URL**: https://cbdb.hsites.harvard.edu/download-cbdb-standalone-database

**License**: CC BY-NC-SA 4.0 — ghi attribution và lưu ý phi thương mại.

**Release đã kiểm tra**: SQLite `cbdb_20260704.sqlite3` (2026-07-04), SHA-256
`6b18f1e7f90a823d8dde5ecbd8a9eac70f092fe643efb8dca193986949a913df`.

**Tải thủ công** (pipeline không tải trong lúc annotate):

1. Truy cập URL trên, tải bản SQLite mới nhất.
2. Lưu SQLite vào `data/external/` (gitignored).
3. Ingest:

   ```bash
   uv run python scripts/build_kb.py ingest-cbdb \
       --input data/external/cbdb_20260704.sqlite3 \
       --version 2026.07.04 \
       --source-url https://huggingface.co/datasets/cbdb/cbdb-sqlite/resolve/main/history/cbdb_202607/cbdb_20260704.zip \
       --license cc-by-nc-sa-4.0 \
       --expected-sha256 6b18f1e7f90a823d8dde5ecbd8a9eac70f092fe643efb8dca193986949a913df
   ```

**Cache**: `build/kb/cbdb.sqlite` (SQLite) + `build/kb/cbdb.sqlite.manifest.json`.

**Schema ingest**: loader tự dò `BIOG_MAIN`, `ALTNAME_DATA` và `NIAN_HAO` qua
`PRAGMA table_info`. Cache giữ tên chính, các loại alias mang tính tên người,
CBDB person ID, triều đại và năm sinh/mất. Các alias dạng tước hiệu, miếu hiệu,
thụy hiệu và giá trị không xác định không được đưa vào matcher.

**Bộ lọc precision mặc định**:

- loại tên 1 ký tự, chuỗi không thuần chữ Hán, placeholder và chuỗi số;
- loại niên hiệu, `triều đại + niên hiệu` và hậu tố tước hiệu thường gặp;
- bỏ surface liên kết với hơn 10 person ID;
- không emit tên 2 ký tự vì dictionary match loại này có nhiều false positive;
- chỉ giữ person ID có triều đại tương thích với thời kỳ tác phẩm.

Có thể nới lỏng khi nghiên cứu recall:

```bash
uv run python scripts/annotate_corpus.py \
    --cbdb-short-names context \
    --cbdb-period-policy prefer \
    --cbdb-max-ambiguity 50
```

Khi cache tồn tại và hash khớp manifest, `annotate_corpus.py` tự thêm nguồn
`cbdb`. Dùng `--no-cbdb` để chạy baseline regex + seed.

## CHGIS — China Historical GIS

**URL**: https://chgis.fas.harvard.edu/

**License**: thay đổi theo phiên bản; kiểm tra trước khi dùng.

**Tải thủ công**:

1. Truy cập CHGIS data portal, tải bản CSV (places).
2. Lưu vào `data/external/chgis_places.csv`.
3. Ingest (BẮT BUỘC truyền `--source-url` và `--license`):

   ```bash
   uv run python scripts/build_kb.py ingest-chgis \
       --input data/external/chgis_places.csv \
       --version 2024.06 \
       --source-url https://chgis.fas.harvard.edu/... \
       --license chgis-terms
   ```

**Cache**: `build/kb/chgis.sqlite` + `build/kb/chgis.sqlite.manifest.json`.

**Schema detector**: thử 3 header variants phổ biến. Không khớp → `CHGISError`.

**Period là prior, không filter cứng** (plan v5 nhận xét #10). Nếu nhiều
CHGIS record hợp lệ cho cùng surface → emit 1 NER entity với
`linking_candidates=[place_id...]` để review.

## Seed gazetteer (built-in)

Không cần external download. Dùng default seed list:

```bash
uv run python scripts/build_kb.py ingest-seed
```

**Cache**: `build/kb/seed.jsonl.gz` + `build/kb/seed.jsonl.gz.manifest.json`.

**Schema**: mỗi dòng `{term, label, alias}`.

## Manifest contract

Mỗi KB khi ingest sẽ ghi manifest gồm:

```json
{
  "name": "cbdb" | "chgis" | "seed",
  "version": "...",
  "source_url": "...",
  "license": "...",
  "file_sha256": "...",
  "file_size": 12345,
  "row_counts": {...}
}
```

KHÔNG có `built_at` (timestamp) để reproducible.

## Accept terms trước khi ingest

CBDB và CHGIS đều có license riêng. Trước khi ingest:

1. Đọc license từng nguồn.
2. Ghi `--license` đúng tên license.
3. Nếu license yêu cầu attribution, ghi vào docs/acknowledgement.md
   (TODO: tạo file này khi cần).
4. KHÔNG phân phối raw SQLite hoặc cache CBDB ra public repo. Chỉ manifest
   provenance được phép version hóa trong repository này.

## License manifest summary

Tổng hợp license các KB trong `build/kb/*.manifest.json`:

```bash
uv run python -c "
import json, glob
for f in glob.glob('build/kb/*.manifest.json'):
    m = json.load(open(f))
    print(f'{m[\"name\"]}: {m[\"license\"]} (version {m[\"version\"]})')
"
```

## Chưa có external data?

Default pipeline chạy được không cần KB ngoài. Chỉ cần seed (built-in).

```bash
uv run python scripts/build_kb.py ingest-seed
uv run python scripts/prepare_corpus.py
uv run python scripts/annotate_corpus.py
```

Nếu không có `build/kb/cbdb.sqlite`, pipeline vẫn chạy regex + seed. Khi cache
CBDB hợp lệ tồn tại, nguồn PERSON được bật tự động; không có network request
trong lúc annotation.
