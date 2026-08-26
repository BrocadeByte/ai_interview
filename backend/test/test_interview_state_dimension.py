import json

from app.agents.nodes.interview_planner import normalize_interview_plan
from app.api.interviews import attach_current_plan_fields
from app.models.interview import InterviewMessage, InterviewSession
from app.services.interview_state_service import build_state_from_session


def build_session(*, message_dimension: str | None) -> InterviewSession:
    session = InterviewSession(
        id=1,
        user_id=1,
        target_position="Python 后端工程师",
        difficulty="medium",
        status="active",
        current_question_index=3,
        interview_plan_json=json.dumps(
            [
                {"dimension": "项目经验", "question_count": 2, "weight": 0.5, "focus": "项目复盘"},
                {"dimension": "系统设计", "question_count": 2, "weight": 0.5, "focus": "架构设计"},
            ],
            ensure_ascii=False,
        ),
    )
    session.messages = [
        InterviewMessage(
            id=10,
            session_id=1,
            role="assistant",
            content="请设计一个高可用的面试会话服务。",
            question_index=3,
            dimension=message_dimension,
            is_followup=0,
            followup_index=0,
        )
    ]
    session.memories = []
    return session


def test_state_restores_persisted_question_dimension() -> None:
    state = build_state_from_session(build_session(message_dimension="自定义架构能力"), profile=None)

    assert state["current_dimension"] == "自定义架构能力"
    assert state["current_question"] == "请设计一个高可用的面试会话服务。"
    assert state["messages"][0].additional_kwargs["dimension"] == "自定义架构能力"


def test_state_derives_dimension_from_plan_for_legacy_message() -> None:
    state = build_state_from_session(build_session(message_dimension=None), profile=None)

    assert state["current_dimension"] == "系统设计"


def test_plan_with_ten_dimensions_is_rebalanced_to_eight_questions() -> None:
    plan = normalize_interview_plan(
        [
            {
                "dimension": f"维度 {index}",
                "question_count": 1,
                "weight": 0.1,
                "focus": f"重点 {index}",
            }
            for index in range(10)
        ]
    )

    assert len(plan) == 8
    assert sum(item["question_count"] for item in plan) == 8
    assert abs(sum(item["weight"] for item in plan) - 1) < 1e-9


def test_session_response_uses_persisted_short_plan_question_count() -> None:
    session = build_session(message_dimension=None)
    session.interview_plan_json = json.dumps(
        [{"dimension": "来源短板再测", "question_count": 2, "weight": 1.0, "focus": "同类能力点"}],
        ensure_ascii=False,
    )

    attach_current_plan_fields(session)

    assert session.total_question_count == 2
    assert session.current_dimension == "来源短板再测"
