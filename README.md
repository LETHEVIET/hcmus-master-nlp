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
docs/data-cleaning.md           # quy tắc làm sạch
docs/annotation-guideline.md    # guideline tách câu và NER
docs/references.md              # tài liệu tham khảo
build/                          # dữ liệu dẫn xuất, tự tạo
```

Dataset hiện có 8 tác phẩm chữ Hán lịch sử. Dữ liệu gốc được mô tả trong
[`dataset/README.md`](dataset/README.md) và luôn được giữ nguyên.

## Cài đặt

Pipeline hiện tại chỉ dùng Python standard library, không cần cài package:

```bash
python3 --version
```

Khuyến nghị Python 3.10 trở lên.

## Chạy toàn bộ pipeline

Từ thư mục gốc của repository:

```bash
python3 scripts/prepare_corpus.py
python3 scripts/annotate_corpus.py
python3 scripts/validate_corpus.py
```

Nếu lệnh validate kết thúc với mã `0` và hiển thị `"valid": true`, các offset
câu/entity hợp lệ.

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

- `build/corpus_annotated.jsonl`: corpus có `sentences`, `entities` và offset.
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
python3 scripts/validate_corpus.py path/to/corpus_annotated.jsonl
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
