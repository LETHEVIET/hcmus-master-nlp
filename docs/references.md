# Tài liệu tham khảo

Các liên kết dưới đây được cung cấp và kiểm tra trong phiên nghiên cứu ngày
13/07/2026. Chúng phục vụ việc thiết kế corpus Hán cổ lịch sử Trung Quốc.
Đề tài hiện tại không yêu cầu huấn luyện model; các model chỉ được tham khảo
hoặc dùng để hỗ trợ tiền gán nhãn.

## Ưu tiên đọc

### Tách câu và dấu câu Hán cổ

1. [EvaHan2024 — Overview of Ancient Chinese Sentence Segmentation and Punctuation](https://aclanthology.org/2024.lt4hala-1.27/)
2. [EvaHan2024 — PDF](https://aclanthology.org/2024.lt4hala-1.27.pdf)
3. [EvaHan2024 shared task](https://circse.github.io/LT4HALA/2024/EvaHan.html)
4. [Automatic Traditional Ancient Chinese Text Segmentation and Punctuation](https://aclanthology.org/2021.ccl-1.61/)
5. [Classical Chinese Sentence Segmentation — Huang et al.](https://aclanthology.org/W10-4103.pdf)
6. [SPEADO](https://aclanthology.org/2024.lt4hala-1.32.pdf)
7. [Sentence Segmentation and Punctuation Based on XunziALLM](https://aclanthology.org/2024.lt4hala-1.30/)

Các nguồn này hỗ trợ việc quyết định giữa rule-based, model sentence
segmentation và xử lý các đoạn thiếu hoặc có dấu câu không đáng tin cậy.

### NER Hán cổ và lịch sử

1. [EvaHan2025 — Overview of Ancient Chinese Named Entity Recognition](https://aclanthology.org/2025.alp-1.19/)
2. [EvaHan2025 — PDF](https://aclanthology.org/2025.alp-1.19.pdf)
3. [EvaHan repository](https://github.com/GoThereGit/EvaHan)
4. [GuNER2023 — NER trên Nhị thập tứ sử](https://aclanthology.org/2023.ccl-3.4/)
5. [WYWEB](https://aclanthology.org/2023.findings-acl.204/)
6. [Nghiên cứu kết hợp segmentation và NER bằng SikuBERT](https://aclanthology.org/2022.nlp4dh-1.21/)

EvaHan2025 và GuNER2023 là nguồn chính để tham khảo schema nhãn, guideline
gán nhãn và whole-entity evaluation. WYWEB chỉ nên dùng để tham khảo benchmark;
schema GLNER của WYWEB quá thô để làm schema cuối cho corpus này.

## Công cụ hỗ trợ tách câu

- [KoichiYasuoka Classical Chinese sentence segmentation — base](https://huggingface.co/KoichiYasuoka/roberta-classical-chinese-base-sentence-segmentation)
- [KoichiYasuoka Classical Chinese sentence segmentation — large](https://huggingface.co/KoichiYasuoka/roberta-classical-chinese-large-sentence-segmentation)
- [XunziALLM](https://github.com/Xunzi-LLM-of-Chinese-classics/XunziALLM)
- [HanLP sentence boundary API](https://hanlp.hankcs.com/docs/api/hanlp/components/eos.html)
- [HanLP data format](https://hanlp.hankcs.com/docs/data_format.html)

Khuyến nghị ban đầu cho corpus là rule-based dựa trên dấu câu hiện có, có lớp
bảo vệ ngoặc/trích dẫn/chú thích. Model Classical Chinese chỉ nên dùng cho
đoạn khó và đầu ra phải được kiểm tra để không thay đổi nguyên văn.

## Công cụ hỗ trợ tiền gán nhãn NER

- [SikuBERT](https://huggingface.co/SIKU-BERT/sikubert)
- [SikuBERT repository](https://github.com/hsc748NLP/SikuBERT-for-digital-humanities-and-classical-Chinese-information-processing)
- [GuwenBERT](https://huggingface.co/ethanyt/guwenbert-base)
- [GuwenBERT repository](https://github.com/Ethan-yt/guwenbert)
- [Guwen NER](https://huggingface.co/ethanyt/guwen-ner)
- [GujiRoBERTa Jian-Fan](https://huggingface.co/hsc748NLP/GujiRoBERTa_jian_fan)
- [HanLP NER documentation](https://hanlp.hankcs.com/docs/annotations/ner/index.html)
- [CKIP Transformers](https://ckip-transformers.readthedocs.io/en/stable/main/readme.html)
- [Stanza NER](https://stanfordnlp.github.io/stanza/ner_models.html)
- [spaCy Chinese models](https://spacy.io/models/zh)

Các công cụ NER tiếng Trung hiện đại như HanLP, CKIP, Stanza và spaCy chỉ nên
dùng làm baseline hoặc annotation helper. Không nên xem đầu ra của chúng là
nhãn vàng cho văn bản Hán cổ nếu chưa kiểm tra thủ công.

## Schema NER tham khảo

EvaHan2025 và GuNER2023 gợi ý các nhóm nhãn sau:

```text
PERSON          Tên người
LOCATION        Địa danh
POLITY          Quốc hiệu/chính thể/nước chư hầu
DYNASTY         Triều đại
OFFICIAL_TITLE  Chức quan, tước vị
TIME            Niên hiệu, năm, thời điểm
BOOK            Tên sách, kinh điển, văn bản
EVENT           Chiến dịch, loạn, biến cố
ETHNIC_GROUP    Tộc danh, cộng đồng lịch sử
```

Đây là schema đề xuất, không phải một chuẩn duy nhất. Nên bắt đầu bằng pilot
với 6 nhãn: `PERSON`, `LOCATION`, `POLITY`, `TIME`, `BOOK`,
`OFFICIAL_TITLE`; sau đó mới mở rộng nếu người gán nhãn phân biệt ổn định được
`DYNASTY`, `EVENT` và `ETHNIC_GROUP`.

## Framework và định dạng dữ liệu

- [Hugging Face token classification](https://huggingface.co/docs/transformers/main/tasks/token_classification)
- [HanLP data format](https://hanlp.hankcs.com/docs/data_format.html)

Nên lưu gold corpus ở dạng span với `start` inclusive và `end` exclusive. Từ
định dạng này có thể sinh BIOES/BMES hoặc định dạng riêng của HanLP khi cần,
nhưng không cần huấn luyện model trong phạm vi đề tài.

## Giấy phép cần kiểm tra

Trước khi phân phối code hoặc corpus dẫn xuất, cần kiểm tra riêng license của
từng model/dataset. Đặc biệt không suy ra license của pretrained model chỉ từ
license của framework. HanLP có thể có điều kiện khác nhau giữa code,
pretrained resources và REST API; CKIP Transformers được ghi nhận là GPL-3.0;
Stanza là Apache-2.0; spaCy là MIT. Các thông tin này cần được kiểm tra lại trên
model card và repository tại thời điểm phát hành.
