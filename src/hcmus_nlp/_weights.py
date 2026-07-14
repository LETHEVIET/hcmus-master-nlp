"""priority_score seeds — CHƯA CALIBRATE.

Đây là priority-for-review, KHÔNG phải xác suất đúng. Sau khi có gold pilot,
calibrate từ dev split (KHÔNG dùng test split) bằng command riêng (chưa viết).

Các hằng số chỉ là seed để các source có thứ tự ưu tiên tạm thời khi chưa có
gold. Khi review, human annotator có thể override priority bằng cách sửa
trực tiếp entity hoặc đánh `review_status=checked`.
"""

# Source kind → priority score (cho review queue).
REGEX = 0.55
SEED_GAZETTEER = 0.70
KB_FULL = 0.85
MODEL_BASIC = 0.65
MODEL_CRF = 0.75

# Human annotation = highest priority (already verified).
HUMAN = 1.0


def priority_for_source(source_name: str) -> float:
    """Lookup priority_score theo source name. Trả 0.5 nếu unknown."""
    table = {
        "regex": REGEX,
        "seed": SEED_GAZETTEER,
        "gazetteer": SEED_GAZETTEER,
        "kb": KB_FULL,
        "kb_full": KB_FULL,
        "model_basic": MODEL_BASIC,
        "guwen_basic": MODEL_BASIC,
        "model_crf": MODEL_CRF,
        "guwen_crf": MODEL_CRF,
        "human": HUMAN,
    }
    return table.get(source_name, 0.5)
