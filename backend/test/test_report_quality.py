import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.agents.nodes.report_generator as report_generator
import app.services.report_service as report_service
from app.models.report import InterviewReport
from app.schemas.llm_outputs import InterviewReportOutput


EMPTY_REPORT = {
    "total_score": 70,
    "summary": "无内容",
    "strengths": [],
    "weaknesses": [],
    "suggestions": [],
    "learning_path": [],
    "sample_answer": "无内容",
}

SCORES = [{
    "question_index": 1,
    "question": "Seata AT 模式如何处理库存与全局锁？",
    "answer": "使用全局事务处理。",
    "dimension": "项目经验",
    "score": 70,
    "reason": "具备基础认识，但混淆了业务库存冻结和 Seata 全局锁。",
    "weaknesses": ["混淆 Seata AT 全局锁与业务库存冻结。"],
    "suggestions": ["系统学习 TM、TC、RM、undo_log 与全局锁生命周期。"],
    "citations": [],
}]

REPORT_STATS = {
    "total_score": 70,
    "dimension_scores": [{
        "dimension": "项目经验",
        "score": 70,
        "question_indexes": [1],
        "focus": "分布式事务的项目实践",
        "weaknesses": ["混淆 Seata AT 全局锁与业务库存冻结。"],
        "suggestions": ["系统学习 TM、TC、RM、undo_log 与全局锁生命周期。"],
        "weight": 1,
    }],
    "top_dimensions": ["项目经验"],
    "lowest_dimensions": ["项目经验"],
    "all_weaknesses": ["混淆 Seata AT 全局锁与业务库存冻结。"],
    "all_suggestions": ["系统学习 TM、TC、RM、undo_log 与全局锁生命周期。"],
}


class _FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.response)


class _StoredScore:
    def model_dump(self) -> dict:
        return SCORES[0]


class _RepairDb:
    async def flush(self) -> None:
        return None


def test_report_schema_rejects_placeholders_and_empty_sections() -> None:
    with pytest.raises(ValidationError):
        InterviewReportOutput.model_validate(EMPTY_REPORT)


@pytest.mark.anyio
async def test_semantically_empty_llm_report_is_repaired_and_filled(monkeypatch) -> None:
    fake_llm = _FakeLLM(json.dumps(EMPTY_REPORT, ensure_ascii=False))

    async def knowledge(*args, **kwargs):
        return {"text": "Seata AT scoring rubric", "citations": []}

    monkeypatch.setattr(report_generator, "llm", fake_llm)
    monkeypatch.setattr(report_generator, "format_knowledge_context", knowledge)

    report = await report_generator.generate_report([], SCORES, REPORT_STATS, "Java 开发")

    assert fake_llm.calls == 2
    assert report["total_score"] == 70
    assert "70 分" in report["summary"]
    assert report["strengths"]
    assert report["weaknesses"] == REPORT_STATS["all_weaknesses"]
    assert report["suggestions"] == REPORT_STATS["all_suggestions"]
    assert report["learning_path"]
    assert "Seata AT" in report["sample_answer"]
    assert report["citations"] == []


def test_completeness_keeps_useful_model_fields_and_fills_only_gaps() -> None:
    completed = report_generator.ensure_report_completeness(
        report_generator.normalize_report({
            **EMPTY_REPORT,
            "summary": "模型生成的有效总体评价。",
            "strengths": ["能够说明基础事务流程。"],
        }),
        scores=SCORES,
        report_stats=REPORT_STATS,
        target_position="Java 开发",
    )

    assert completed["summary"] == "模型生成的有效总体评价。"
    assert completed["strengths"] == ["能够说明基础事务流程。"]
    assert completed["weaknesses"] == REPORT_STATS["all_weaknesses"]
    assert completed["suggestions"] == REPORT_STATS["all_suggestions"]
    assert completed["learning_path"]
    assert completed["sample_answer"] != "无内容"


@pytest.mark.anyio
async def test_existing_empty_report_is_repaired_without_changing_citations(monkeypatch) -> None:
    citations_json = '[{"document_id":48,"chunk_id":"48-2-1"}]'
    report = InterviewReport(
        id=23,
        session_id=51,
        total_score=70,
        summary="无内容",
        strengths="[]",
        weaknesses="[]",
        suggestions="[]",
        dimension_scores_json=json.dumps(REPORT_STATS["dimension_scores"], ensure_ascii=False),
        citations_json=citations_json,
        learning_path="[]",
        sample_answer="无内容",
        created_at=datetime(2026, 8, 11, 12, 0, 0),
        updated_at=datetime(2026, 8, 11, 12, 0, 0),
    )
    session = SimpleNamespace(target_position="Java 开发", interview_plan_json=None)

    async def stored_scores(db, session_id):
        return [_StoredScore()]

    monkeypatch.setattr(report_service, "list_scores", stored_scores)
    repaired = await report_service.repair_report_if_incomplete(_RepairDb(), report, session)

    assert repaired is True
    assert report.citations_json == citations_json
    assert "70 分" in report.summary
    assert json.loads(report.strengths)
    assert json.loads(report.weaknesses)
    assert json.loads(report.suggestions)
    assert json.loads(report.learning_path)
    assert report.sample_answer != "无内容"
