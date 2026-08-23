import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.models.question_review import QuestionReview
from app.models.score import InterviewScore
from app.services.question_review_service import (
    build_grounded_sample_answer,
    build_question_review_payload,
    question_review_is_incomplete,
)


def test_missing_score_fields_use_complete_deterministic_fallbacks() -> None:
    score = InterviewScore(
        id=9,
        session_id=3,
        question_index=0,
        question="",
        answer="",
        dimension="",
        score=64,
        sub_scores="not-json",
        reason="",
        weaknesses="[]",
        suggestions="[]",
    )

    payload = build_question_review_payload(score)

    assert payload["question_index"] == 1
    assert payload["dimension"] == "综合表现"
    assert payload["question"]
    assert payload["answer"]
    assert json.loads(payload["sub_scores_json"]) == {"综合评分": 64}
    assert json.loads(payload["deduction_reasons_json"])
    assert json.loads(payload["suggested_structure_json"])
    assert payload["sample_answer"]
    assert "当前信息不足" in payload["sample_answer"]
    assert "[真实" in payload["sample_answer"]
    assert json.loads(payload["practice_seed_json"])["source_score_id"] == 9


def test_sample_answer_reuses_candidate_text_and_keeps_missing_facts_as_placeholders() -> None:
    answer = "我在项目中使用 FastAPI 实现了登录鉴权。"

    sample = build_grounded_sample_answer(answer, ["补充真实结果数据"])

    assert answer in sample
    assert "[真实背景、目标与约束]" in sample
    assert "[真实结果、验证方式或如实说明暂无量化指标]" in sample
    assert "某公司" not in sample
    assert "提升了 50%" not in sample


def test_incomplete_persisted_review_is_detected_for_repair() -> None:
    review = QuestionReview(
        id=4,
        report_id=2,
        session_id=3,
        score_id=9,
        question_index=1,
        dimension="项目经验",
        question="请介绍项目。",
        answer="我的回答。",
        score=70,
        sub_scores_json="{}",
        deduction_reasons_json="[]",
        suggested_structure_json="[]",
        sample_answer="",
        weaknesses_json="[]",
        weakness_key="question_1",
        practice_seed_json="{}",
    )

    assert question_review_is_incomplete(review) is True
