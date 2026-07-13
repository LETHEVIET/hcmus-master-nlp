# Hướng dẫn tách câu và gán nhãn NER phiên bản 0.1

## Phạm vi

Đây là guideline để tạo corpus chữ Hán lịch sử Trung Quốc. Không huấn luyện
model trong phạm vi đề tài. Kết quả tiền gán nhãn phải được con người kiểm tra
trước khi gọi là dữ liệu vàng.

## Tách câu

- Tách sau `。`, `！`, `？` và các dạng tương đương.
- Không tách dấu câu nằm bên trong `〈...〉`, `《...》`, `（...）`, `「...」`,
  `『...』`, `【...】`.
- Tạm thời không tự động tách sau `；` và `：`; đây là ranh giới mệnh đề hoặc
  dấu dẫn nhập, cần kiểm tra ngữ cảnh.
- Không sửa ký tự gốc. Offset dùng `start` inclusive và `end` exclusive theo
  chỉ số Unicode Python.
- Nếu đoạn không có dấu câu kết thúc, giữ nguyên đoạn và đánh dấu
  `review_status: needs_review`.

## Nhãn NER phiên bản đầu

| Nhãn | Ý nghĩa | Ví dụ |
|---|---|---|
| `PERSON` | Tên người, hiệu hoặc cách gọi quy chiếu một người | 高祖, 蘇秦 |
| `BOOK` | Tên sách/văn bản trong `《...》` hoặc tên điển tịch rõ ràng | 《漢書》 |
| `OFFICIAL_TITLE` | Chức quan, chức tước | 太守, 刺史, 將軍 |
| `LOCATION` | Địa danh tự nhiên hoặc hành chính | 長安, 南陽郡 |
| `TIME` | Năm, niên hiệu, mốc thời gian | 建武元年, 三年 |
| `POLITY` | Quốc hiệu, nước, chính thể | 漢國, 齊國 |

Các nhãn `DYNASTY`, `EVENT`, `ETHNIC_GROUP` để dành cho phiên bản mở rộng.

## Quy tắc gán nhãn

- Gán span dài nhất có ý nghĩa trong ngữ cảnh, nhưng không gộp phần mô tả
  không thuộc thực thể.
- Không gán nhãn mọi danh từ chung chỉ vì có hậu tố như `城`, `州`, `王`.
- Nếu một chuỗi vừa có thể là triều đại vừa là mốc thời gian, ưu tiên nhãn
  theo vai trò trong câu và ghi chú trường hợp không chắc chắn.
- Không gán thực thể nằm hoàn toàn trong chú thích `〈案 ...〉` vào bản gold nếu
  dự án quyết định loại chú thích khỏi lớp văn bản chính; trước mắt vẫn giữ
  chú thích và đánh dấu để review.
- Mọi nhãn do script sinh ra đều là `heuristic` và `needs_review`.

## Định dạng đầu ra

`build/corpus_annotated.jsonl` giữ nguyên `text`, thêm:

- `sentences`: các span câu.
- `entities`: các span entity ở offset toàn văn bản.
- `annotation`: phiên bản guideline và trạng thái review.

Script tạo tiền gán nhãn bằng:

```bash
python3 scripts/annotate_corpus.py
```

Không được xem `build/corpus_annotated.jsonl` là gold corpus cho tới khi đã
review và sửa các bản ghi `needs_review`.
