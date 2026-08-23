import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.agents.nodes.interview_planner as interview_planner
from app.agents.state import create_initial_state


def test_type_specific_fallback_plans_match_product_constraints() -> None:
    hr_plan = interview_planner.get_default_interview_plan("hr", "full_interview")
    project_plan = interview_planner.get_default_interview_plan("project_deep_dive", "full_interview")
    technical_plan = interview_planner.get_default_interview_plan("technical_basics", "full_interview")
    system_design_plan = interview_planner.get_default_interview_plan("system_design", "full_interview")

    assert sum(item["question_count"] for item in hr_plan) == 8
    assert all("技术" not in item["dimension"] for item in hr_plan)
    assert {item["dimension"] for item in hr_plan} == {
        "求职动机与岗位认知",
        "经历与稳定性",
        "沟通协作与冲突处理",
        "职业规划与自我认知",
    }
    assert "简历真实项目" in project_plan[0]["focus"]
    assert technical_plan[0]["dimension"] == "JD 必备技能"
    assert "must_have_skills" in technical_plan[0]["focus"]
    assert any("初级岗位" in item["focus"] for item in system_design_plan)
    assert abs(sum(item["weight"] for item in system_design_plan) - 1) < 1e-9


@pytest.mark.parametrize(
    ("session_purpose", "expected_count"),
    [
        ("full_interview", 8),
        ("weakness_practice", 4),
        ("retest", 2),
    ],
)
def test_plan_is_rebalanced_to_session_purpose_question_count(
    session_purpose: str,
    expected_count: int,
) -> None:
    raw_plan = [
        {"dimension": "能力一", "question_count": 5, "weight": 0.5, "focus": "重点一"},
        {"dimension": "能力二", "question_count": 5, "weight": 0.5, "focus": "重点二"},
    ]

    plan = interview_planner.normalize_interview_plan(
        raw_plan,
        interview_type="technical_basics",
        session_purpose=session_purpose,
    )

    assert sum(item["question_count"] for item in plan) == expected_count
    assert interview_planner.get_interview_question_count(plan) == expected_count
    assert abs(sum(item["weight"] for item in plan) - 1) < 1e-9


def test_legacy_or_invalid_plan_keeps_eight_question_default() -> None:
    assert interview_planner.get_interview_question_count([]) == 8
    assert interview_planner.get_interview_question_count(None) == 8
    assert interview_planner.get_interview_question_count([{"question_count": "invalid"}]) == 8


@pytest.mark.anyio
async def test_planner_prompt_contains_type_mode_resume_jd_and_practice_context(monkeypatch) -> None:
    captured_messages = []
    captured_query = {}

    async def fake_knowledge(**kwargs) -> str:
        captured_query.update(kwargs)
        return "岗位知识"

    async def fake_invoke(_llm, messages, *, field):
        captured_messages.extend(messages)
        assert field == "question"
        return SimpleNamespace(
            content=(
                '{"question":"请说明你会如何改进该能力点。","plan":['
                '{"dimension":"专项基础","question_count":3,"weight":0.5,"focus":"基础理解"},'
                '{"dimension":"专项应用","question_count":3,"weight":0.5,"focus":"实际应用"}]}'
            )
        )

    monkeypatch.setattr(interview_planner, "format_knowledge_context", fake_knowledge)
    monkeypatch.setattr(interview_planner, "invoke_json_with_streaming_field", fake_invoke)
    state = create_initial_state(
        user_id=1,
        session_id=9,
        target_position="Python 后端工程师",
        difficulty="easy",
        profile={"experience_years": 1},
        mode="mock",
        interview_type="system_design",
        resume={"parsed": {"projects": [{"name": "AI 面试系统"}]}},
        target_job={
            "parsed": {
                "seniority": "初级",
                "experience_requirements": "1-2 年",
                "must_have_skills": ["Python", "MySQL"],
            }
        },
        parent_session_id=3,
        source_report_id=5,
        source_weakness_key="architecture_tradeoff",
        session_purpose="weakness_practice",
    )

    result = await interview_planner.plan_interview_node(state)

    assert sum(item["question_count"] for item in result["interview_plan"]) == 4
    assert captured_query["purpose"] == "planning"
    assert "系统设计" in captured_query["query"]
    system_prompt = str(captured_messages[0].content)
    user_prompt = str(captured_messages[1].content)
    assert "HR：" in system_prompt and "不得生成纯技术知识背诵题" in system_prompt
    assert "项目深挖：优先引用简历中的真实项目" in system_prompt
    assert "parsed.must_have_skills" in system_prompt
    assert "初级岗位" in system_prompt
    assert "实战模式" in user_prompt and "不在题面中透露评分" in user_prompt
    assert "本场主问题数：4" in user_prompt
    assert '"source": "resume_snapshot"' in user_prompt and '"projects"' in user_prompt
    assert '"source": "job_description_snapshot"' in user_prompt and "1-2" in user_prompt
    assert '"source": "source_weakness_key"' in user_prompt and "architecture_tradeoff" in user_prompt


@pytest.mark.anyio
async def test_type_specific_fallback_question_is_used_when_planner_fails(monkeypatch) -> None:
    async def fake_knowledge(**_kwargs) -> str:
        return ""

    async def fail_invoke(*_args, **_kwargs):
        raise RuntimeError("planner unavailable")

    monkeypatch.setattr(interview_planner, "format_knowledge_context", fake_knowledge)
    monkeypatch.setattr(interview_planner, "invoke_json_with_streaming_field", fail_invoke)
    state = create_initial_state(
        user_id=1,
        session_id=10,
        target_position="前端工程师",
        difficulty="medium",
        profile={},
        interview_type="hr",
    )

    result = await interview_planner.plan_interview_node(state)

    assert "为什么选择这个目标岗位" in result["current_question"]
    assert result["current_dimension"] == "求职动机与岗位认知"
    assert sum(item["question_count"] for item in result["interview_plan"]) == 8
