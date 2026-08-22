import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from types import SimpleNamespace


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.schemas.llm_outputs import AnswerPipelineOutput
from app.services.llm_json import as_dict, as_list, clamp_int, parse_json_model_with_repair, parse_json_object, parse_json_object_with_repair


class FakeRepairLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return SimpleNamespace(content=self.responses[self.calls - 1])


def test_parse_json_object_accepts_plain_json() -> None:
    assert parse_json_object('{"summary": "ok"}') == {"summary": "ok"}


def test_parse_json_object_accepts_fenced_json() -> None:
    assert parse_json_object('```json\n{"summary": "ok"}\n```') == {"summary": "ok"}


def test_parse_json_object_extracts_embedded_json() -> None:
    assert parse_json_object('结果如下：{"summary": "ok"} 请查收') == {"summary": "ok"}


def test_parse_json_object_rejects_non_object_json() -> None:
    with pytest.raises(ValueError):
        parse_json_object('["not", "object"]')


@pytest.mark.anyio
async def test_parse_json_object_with_repair_retries_once() -> None:
    fake_llm = FakeRepairLLM(['{"summary": "fixed"}'])

    result = await parse_json_object_with_repair(
        "not-json",
        llm=fake_llm,
        expected_schema='{"summary": "..."}',
        max_retries=1,
    )

    assert result == {"summary": "fixed"}
    assert fake_llm.calls == 1


@pytest.mark.anyio
async def test_parse_json_object_with_repair_respects_retry_limit() -> None:
    fake_llm = FakeRepairLLM(["still not json"])

    with pytest.raises(ValueError):
        await parse_json_object_with_repair(
            "not-json",
            llm=fake_llm,
            expected_schema='{"summary": "..."}',
            max_retries=1,
        )

    assert fake_llm.calls == 1


def test_normalizers_keep_business_fields_safe() -> None:
    assert clamp_int("120") == 100
    assert clamp_int("bad", default=60) == 60
    assert as_list("one") == ["one"]
    assert as_list(None) == []
    assert as_dict({"a": 1}) == {"a": 1}
    assert as_dict("bad") == {}


def valid_answer_pipeline_data() -> dict:
    return {
        "needs_followup": False,
        "decision_reason": "回答完整。",
        "question": "下一题",
        "score": 80,
        "sub_scores": {
            "专业准确性": 80,
            "表达清晰度": 80,
            "项目真实性": 80,
            "岗位匹配度": 80,
        },
        "reason": "评分合理。",
        "weaknesses": [],
        "suggestions": [],
    }


def test_answer_output_rejects_string_boolean() -> None:
    data = valid_answer_pipeline_data()
    data["needs_followup"] = "false"

    with pytest.raises(ValidationError):
        AnswerPipelineOutput.model_validate(data)


def test_answer_output_rejects_missing_required_sub_scores() -> None:
    data = valid_answer_pipeline_data()
    data["sub_scores"] = {"专业准确性": 80}

    with pytest.raises(ValidationError):
        AnswerPipelineOutput.model_validate(data)


@pytest.mark.anyio
async def test_parse_json_model_repairs_schema_error() -> None:
    import json

    repaired = valid_answer_pipeline_data()
    fake_llm = FakeRepairLLM([json.dumps(repaired, ensure_ascii=False)])
    invalid = valid_answer_pipeline_data()
    invalid["needs_followup"] = "false"

    result = await parse_json_model_with_repair(
        json.dumps(invalid, ensure_ascii=False),
        llm=fake_llm,
        output_model=AnswerPipelineOutput,
        max_retries=1,
    )

    assert result.needs_followup is False
    assert fake_llm.calls == 1
