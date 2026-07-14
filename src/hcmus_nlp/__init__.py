"""hcmus_nlp — pipeline NER cho corpus Hán cổ lịch sử.

Pipeline gồm các phase (xem plan):
- volume.parse_volume_heading: chuẩn hóa heading quyển
- cleaning.audit: phát hiện boilerplate / inline OCR placeholder / URL leak
- labels.load_mapping: load mapping internal → submission
- sentence.segment: tách câu rule-based
- sources.*: SourceAdapter Protocol + regex/gazetteer/model
- candidates.{Trie, CandidateMerger}: gộp + làm phẳng candidate
- gold.sample: tạo evaluation_random + diagnostic_challenge
- eval.*: entity_strict / boundary_only / per_label / label_acc
- doccano_io: roundtrip Doccano 1.8.x
- kb.*: ingest seed / CBDB / CHGIS

Core runtime chỉ dùng Python standard library (tomllib, dataclasses, re, json,
sqlite3, gzip, hashlib, urllib). Optional deps (transformers, torch, pytest,
seqeval) nằm trong [project.optional-dependencies] của pyproject.toml.
"""
