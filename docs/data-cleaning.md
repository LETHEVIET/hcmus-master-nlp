# Quy trình loại bỏ dữ liệu nhiễu

## Mục tiêu

Các file trong `dataset/` được xem là dữ liệu gốc và không bị chỉnh sửa. Quy
trình làm sạch tạo ra bản dẫn xuất trong `build/corpus.jsonl`, đồng thời giữ
`source_file` và `source_line` để có thể truy ngược mỗi đoạn về file gốc.

Mục tiêu của bước này là loại bỏ thành phần do trang nguồn hoặc định dạng
trích xuất sinh ra, không phải hiện đại hóa hay hiệu đính nội dung chữ Hán.

Quy trình này phục vụ việc tạo corpus. Không có yêu cầu huấn luyện hoặc
fine-tune mô hình trong phạm vi hiện tại.

## Các loại dữ liệu bị loại bỏ

Script `scripts/prepare_corpus.py` loại bỏ các trường hợp sau:

| Loại | Mẫu nhận diện | Xử lý |
|---|---|---|
| Metadata của trang nguồn | Dòng bắt đầu bằng `姊妹计划` | Loại bỏ |
| Nhãn bản quyền | Dòng bắt đầu bằng `Public domain` | Loại bỏ |
| Thông báo bản quyền tiếng Trung | Dòng bắt đầu bằng `本作品在全世界都属于` | Loại bỏ |
| Ghi chú về dấu câu của bản nguồn | Dòng bắt đầu bằng `本作品 原文没有標點` | Loại bỏ |
| Tiêu đề Markdown/kỹ thuật | Dòng bắt đầu bằng `#`, `##`, ... | Loại bỏ khỏi `text`, chuyển thành metadata |
| Tiêu đề tác phẩm lặp lại | Dòng đầu trùng tên tác phẩm | Loại bỏ |
| Dòng trống | Dòng không có nội dung sau khi chuẩn hóa | Loại bỏ |

Các dòng tiêu đề có dạng `卷...` được lưu vào trường `volume`; các tiêu đề khác
được lưu vào trường `section`. Như vậy thông tin cấu trúc không bị mất khi
tiêu đề bị loại khỏi nội dung văn bản.

## Các thành phần được giữ lại

Các nội dung sau không bị tự động xóa:

- Văn bản chữ Hán lịch sử.
- Dấu câu nguyên bản.
- Chú thích học thuật, ví dụ `〈案 ...〉`.
- Dị thể chữ và cách viết cổ.
- Lời tựa hoặc phần khảo chứng nếu chúng nằm trong cùng văn bản nguồn.

Lý do là việc tự động xóa các thành phần này có thể làm mất thông tin nghiên
cứu. Nếu đề tài yêu cầu corpus chỉ chứa văn bản chính, chú thích cần được tách
thành một lớp dữ liệu riêng và kiểm tra thủ công trước khi loại bỏ.

## Chuẩn hóa định dạng

Quy trình chỉ thực hiện các thay đổi kỹ thuật tối thiểu:

- Đọc file dưới dạng UTF-8.
- Xóa xuống dòng ở cuối dòng.
- Xóa khoảng trắng đầu/cuối dòng.
- Gộp chuỗi khoảng trắng ngang thành một khoảng trắng.
- Không chuyển phồn thể sang giản thể.
- Không dịch sang tiếng Việt.
- Không sửa nội dung chữ Hán hoặc tự động sửa lỗi OCR.

## Tái tạo kết quả

Chạy từ thư mục gốc của dự án:

```bash
rm -rf build
python3 scripts/prepare_corpus.py
```

Kết quả được ghi vào:

- `build/corpus.jsonl`: dữ liệu đã làm sạch và phân đoạn.
- `build/metadata.json`: thống kê theo từng tác phẩm.
- `build/statistics.json`: tổng số bản ghi, ký tự và số dòng bị loại bỏ theo loại.

Trong lần chạy hiện tại, quy trình tạo 57.474 bản ghi và loại bỏ 3.809 dòng
boilerplate, 819 dòng tiêu đề kỹ thuật và 8 tiêu đề tác phẩm lặp lại.
