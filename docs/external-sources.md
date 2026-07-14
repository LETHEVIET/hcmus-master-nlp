# External KB sources

## CBDB — China Biographical Database

**URL**: https://cbdb.hsites.harvard.edu/download-cbdb-standalone-database

**License**: CC BY-NC-SA 4.0 — ghi attribution và lưu ý phi thương mại.

**Tải thủ công** (KHÔNG auto-download trong session này):

1. Truy cập URL trên, tải bản SQLite mới nhất.
2. Lưu vào `data/external/cbdb.sqlite` (gitignored).
3. Ingest:

   ```bash
   uv run python scripts/build_kb.py ingest-cbdb \
       --input data/external/cbdb.sqlite \
       --version 2024.06 \
       --source-url https://... \
       --license cc-by-nc-sa-4.0
   ```

**Cache**: `build/kb/cbdb.sqlite` (SQLite) + `build/kb/cbdb.sqlite.manifest.json`.

**Schema introspect**: loader tự dò table `BIOG_MAIN` qua `PRAGMA table_info`.
Robust với các phiên bản schema khác nhau.

**Runtime filter**: 1-char name bị skip (`min_len >= 2`) vì ambiguous.

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
4. KHÔNG phân phối cache ra public repo nếu license không cho phép.

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

## Chưa có data thật?

Default pipeline chạy được không cần KB ngoài. Chỉ cần seed (built-in).

```bash
uv run python scripts/build_kb.py ingest-seed
uv run python scripts/prepare_corpus.py
uv run python scripts/annotate_corpus.py
```

Regex + seed gazetteer đủ cho pilot đầu tiên. CBDB/CHGIS dùng để mở rộng
sang các nguồn chính thống sau khi có giấy phép và dữ liệu.