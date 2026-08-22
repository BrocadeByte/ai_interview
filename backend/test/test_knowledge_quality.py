import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.knowledge_quality import KnowledgeQualityError, analyze_knowledge_text, validate_knowledge_text


def test_clean_mixed_language_document_is_accepted() -> None:
    report = validate_knowledge_text(
        "Python backend scoring rubric\n\n"
        "\u5019\u9009\u4eba\u5e94\u8bf4\u660e JWT \u8fc7\u671f\u3001\u5237\u65b0\u548c\u5b89\u5168\u53d6\u820d\u3002"
    )

    assert report.accepted is True
    assert report.mojibake_ratio == 0
    assert report.abnormal_character_ratio == 0


def test_normal_question_marks_are_not_abnormal_characters() -> None:
    report = validate_knowledge_text("Why this design? What changed? What was the result?")

    assert report.accepted is True
    assert report.abnormal_character_ratio == 0


def test_mojibake_document_is_rejected() -> None:
    text = ("\u951f\u65a4\u62f7 broken encoding content " * 20)

    with pytest.raises(KnowledgeQualityError, match="mojibake ratio"):
        validate_knowledge_text(text)


def test_common_chinese_mojibake_sequence_is_rejected() -> None:
    text = ("\u93c2\u56e8 legacy decoded text " * 20)

    with pytest.raises(KnowledgeQualityError, match="mojibake ratio"):
        validate_knowledge_text(text)


def test_abnormal_control_character_ratio_is_rejected() -> None:
    text = "valid" + ("\x00\x01\x02" * 20)

    with pytest.raises(KnowledgeQualityError, match="abnormal character ratio"):
        validate_knowledge_text(text)


def test_excessive_empty_pdf_pages_are_rejected() -> None:
    text = "Page 1\nJWT rubric\n\nPage 2\n\nPage 3\n\nPage 4\n"
    report = analyze_knowledge_text(text)

    assert report.empty_page_count == 3
    assert report.empty_page_ratio == 0.75
    assert report.accepted is False


def test_duplicate_content_blocks_are_rejected() -> None:
    block = "This is the same repeated scoring paragraph with enough characters."
    text = "\n\n".join([block] * 6)

    with pytest.raises(KnowledgeQualityError, match="duplicate block ratio"):
        validate_knowledge_text(text)
