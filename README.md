# Xây dựng ngữ liệu đơn ngữ chữ Hán chuyên ngành lịch sử Trung Quốc

Corpus chữ Hán cổ/văn ngôn về lịch sử Trung Quốc thời phong kiến. “Đơn ngữ”
nghĩa là phần văn bản nghiên cứu chỉ có tiếng Hán, không ghép song song với
bản dịch tiếng Việt.

## Phạm vi

Đề tài tập trung vào việc tạo corpus, không huấn luyện hoặc fine-tune model.
Model/công cụ có sẵn chỉ được dùng để hỗ trợ tách câu hoặc tiền-gán nhãn NER;
nhãn cuối phải được kiểm tra và chỉnh sửa thủ công.

OCR không thuộc pipeline hiện tại vì dataset chỉ có văn bản text, không có ảnh.

## Cấu trúc thư mục

```text
dataset/                         # dữ liệu raw, không ghi đè
scripts/prepare_corpus.py       # làm sạch và tạo corpus JSONL
scripts/annotate_corpus.py      # tách câu + tiền-gán nhãn NER
scripts/validate_corpus.py      # kiểm tra offset và tính toàn vẹn
scripts/export_submission.py    # xuất đúng format nộp bài theo quyển
docs/data-cleaning.md           # quy tắc làm sạch
docs/annotation-guideline.md    # guideline tách câu và NER
docs/references.md              # tài liệu tham khảo
build/                          # dữ liệu dẫn xuất, tự tạo
```

Dataset hiện có 8 tác phẩm chữ Hán lịch sử. Dữ liệu gốc được mô tả trong
[`dataset/README.md`](dataset/README.md) và luôn được giữ nguyên.

## Cài đặt

Pipeline dùng [`uv`](https://docs.astral.sh/uv/) làm build tool. Cài `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Sau đó từ thư mục repo:

```bash
uv sync --extra dev       # core + pytest
```

`uv.lock` được commit — môi trường tái lập giữa các máy.

Khuyến nghị Python 3.11 (xem `.python-version`). Chi tiết xem
[`docs/uv-setup.md`](docs/uv-setup.md).

## Chạy toàn bộ pipeline

Từ thư mục gốc của repository:

```bash
uv run python3 scripts/prepare_corpus.py
uv run python3 scripts/annotate_corpus.py
uv run python3 scripts/validate_corpus.py
uv run python3 scripts/export_submission.py --mode draft
```

Nếu lệnh validate kết thúc với mã `0` và hiển thị `"valid": true`, các offset
câu/entity hợp lệ.

### Các mode export

```bash
# Format check, không phải submission thật (cho phép needs_review)
uv run python3 scripts/export_submission.py --mode draft

# Xuất gold pilot để review
uv run python3 scripts/export_submission.py --mode pilot \
    --pilot build/pilot/evaluation_random.jsonl

# Submission thật — yêu cầu mapping.confirmed=true + mọi sentence checked
uv run python3 scripts/export_submission.py --mode final
```

### Compliance check (submission artifact)

```bash
uv run python3 scripts/compliance_check.py \
    --submission build/submission \
    --mode draft
```

### Tạo gold pilot

```bash
uv run python3 scripts/create_gold_pilot.py \
    --input build/corpus_preannotated.jsonl \
    --output build/pilot \
    --pilot-size 200 \
    --seed 42
```

Output:
- `build/pilot/evaluation_random.jsonl` (200 câu, stratified theo work) — dùng cho metric.
- `build/pilot/diagnostic_challenge.jsonl` (stratified theo nhiều tiêu chí) — chỉ error analysis.
- `build/pilot/manifest.json` (seed, quota, double_annotate fraction).

### Đánh giá trên gold

Tạo baseline regex-only riêng, không dùng chung artifact với prediction hiện tại:

```bash
uv run python3 scripts/annotate_corpus.py \
    --input build/corpus.jsonl \
    --output build/baseline_regex.jsonl \
    --stats build/baseline_regex_statistics.json \
    --no-seed
```

```bash
uv run python3 scripts/evaluate.py \
    --gold build/gold/pilot.checked.jsonl \
    --pred build/corpus_preannotated.jsonl \
    --filter build/pilot/evaluation_random.jsonl \
    --baseline build/baseline_regex.jsonl
```

`baseline_regex.jsonl` chỉ dùng `RegexSource`; `corpus_preannotated.jsonl`
dùng pipeline source hiện tại (mặc định regex + seed cache nếu có). Vì hai
artifact độc lập, regression check mới có ý nghĩa.

Output: `build/pilot/eval_report.json` với strict F1, boundary F1, per-label
F1, label accuracy, confusion matrix. So sánh với baseline; exit code 2 nếu
regression.

### Roundtrip Doccano

```bash
# Export corpus → Doccano JSONL
uv run python3 scripts/doccano_io.py to-doccano

# Sau khi review trong Doccano:
uv run python3 scripts/doccano_io.py from-doccano \
    --doccano build/doccano/export.jsonl \
    --input build/corpus_preannotated.jsonl \
    --output build/gold/pilot.checked.jsonl
```

### Bước 1 — Làm sạch và cấu trúc hóa

```bash
python3 scripts/prepare_corpus.py
```

Đầu ra:

- `build/corpus.jsonl`: mỗi dòng là một đoạn văn kèm metadata.
- `build/metadata.json`: thống kê theo tác phẩm.
- `build/statistics.json`: thống kê tổng thể và số dòng bị loại bỏ.

Script loại bỏ metadata/trang bản quyền, tiêu đề kỹ thuật, dòng trống và
chuẩn hóa khoảng trắng. Chú thích học thuật vẫn được giữ lại. Chi tiết xem
[`docs/data-cleaning.md`](docs/data-cleaning.md).

### Bước 2 — Tách câu và tiền-gán nhãn NER

```bash
python3 scripts/annotate_corpus.py
```

Đầu ra:

- `build/corpus_preannotated.jsonl`: corpus có `sentences`, `entities` (kèm provenance từ CandidateMerger) và `unresolved_conflicts`.
- `build/annotation_statistics.json`: số câu và entity theo nhãn.

Tách câu hiện tại dùng rule-based trên các dấu `。`, `！`, `？`, đồng thời bảo
vệ nội dung bên trong các cặp dấu như `〈〉`, `《》`, `（）`, `「」`, `『』` và
`【】`.

NER hiện tại là heuristic pre-annotation với các nhãn:

```text
BOOK
TIME
OFFICIAL_TITLE
LOCATION
POLITY
```

Mọi entity đều có `review_status: "needs_review"`; đây chưa phải gold corpus.
Tên người (`PERSON`) cần được bổ sung trong bước review thủ công vì heuristic
không đủ đáng tin để tự nhận diện tên người Hán cổ.

### Bước 4 — Xuất format theo yêu cầu đề tài

```bash
python3 scripts/export_submission.py
```

Đầu ra nằm trong `build/submission/`:

```text
build/submission/
├── HCH_001/
│   ├── HCH_001_01/
│   │   ├── HCH_001_01_seg.tsv
│   │   └── HCH_001_01_ner.json
│   └── ...
├── HCH_002/
└── manifest.json
```

Mỗi dòng trong `_seg.tsv` có đúng format:

```text
sentence_id<TAB>sentence
```

Mỗi phần tử trong `_ner.json` có format:

```json
{
  "sentence_id": "HCH_001_01_000001",
  "sentence": "高祖，沛豐邑中陽里人也。",
  "entities": [
    {"text": "沛豐邑", "label": "LOC", "start": 3, "end": 6}
  ]
}
```

Schema xuất ra hỗ trợ tối thiểu các nhãn bắt buộc:

```text
PER  LOC  ORG  TITLE  TME  NUM
```

Ánh xạ hiện tại là `PERSON→PER`, `LOCATION→LOC`, `POLITY→ORG`,
`BOOK/OFFICIAL_TITLE→TITLE`, `TIME→TME`. Vì corpus là đầu vào text nên không
tạo file `_raw.txt` riêng; dữ liệu raw vẫn được bảo toàn trong `dataset/`.

### Bước 3 — Kiểm tra dữ liệu

```bash
python3 scripts/validate_corpus.py
```

Validator kiểm tra:

- JSONL có hợp lệ không.
- Span câu có khớp với `text` không.
- Span entity có khớp với `text` không.
- Entity có nằm trong câu tương ứng không.
- Có khoảng trống không hợp lệ giữa các câu không.

Có thể kiểm tra file khác:

```bash
python3 scripts/validate_corpus.py path/to/corpus_preannotated.jsonl
```

## Định dạng JSONL

Ví dụ một record:

```json
{
  "id": "hanshu-000001",
  "title": "漢書",
  "volume": "卷一",
  "section": "高帝紀",
  "text": "高祖，沛豐邑中陽里人也。",
  "sentences": [
    {
      "sid": "hanshu-000001-s1",
      "start": 0,
      "end": 12,
      "text": "高祖，沛豐邑中陽里人也。",
      "method": "rule",
      "review_status": "needs_review"
    }
  ],
  "entities": [
    {
      "eid": "hanshu-000001-e1",
      "sentence_id": "hanshu-000001-s1",
      "start": 3,
      "end": 6,
      "text": "沛豐邑",
      "label": "LOCATION",
      "method": "heuristic",
      "review_status": "needs_review"
    }
  ]
}
```

Offset dùng quy ước `start` inclusive, `end` exclusive theo chỉ số Unicode
Python.

## Quy trình review thủ công

1. Review các câu có `review_status: "needs_review"`.
2. Sửa hoặc xóa entity heuristic bị sai.
3. Bổ sung `PERSON` và entity bị bỏ sót.
4. Đổi trạng thái các record đã kiểm tra sang `checked`.
5. Chỉ sau bước này mới gọi dữ liệu là gold/curated corpus.

Guideline chi tiết nằm tại
[`docs/annotation-guideline.md`](docs/annotation-guideline.md).

## Kết quả hiện tại

Sau khi chạy pipeline hiện tại:

- 57.474 record văn bản.
- 271.471 câu được tách bằng rule.
- 184.670 entity candidates.
- Trạng thái: `preannotation_needs_review`.

Đây là corpus dẫn xuất để review, chưa phải phiên bản cuối đã được con người
kiểm duyệt hoàn toàn.

## Tài liệu tham khảo

- [docs/data-cleaning.md](docs/data-cleaning.md)
- [docs/annotation-guideline.md](docs/annotation-guideline.md)
- [docs/references.md](docs/references.md)
