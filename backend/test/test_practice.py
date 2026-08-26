import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401, E402
from app.api import interviews as interviews_api  # noqa: E402
from app.api import practice as practice_api  # noqa: E402
from app.api.deps import get_current_user  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.models.interview import InterviewSession  # noqa: E402
from app.models.practice import PracticeSession  # noqa: E402
from app.models.question_review import QuestionReview  # noqa: E402
from app.models.report import InterviewReport  # noqa: E402
from app.models.score import InterviewScore  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.question_review_service import build_question_review_payload  # noqa: E402
from app.services.interview_state_service import build_state_from_session  # noqa: E402
from app.services.practice_service import (  # noqa: E402
    _compare_weaknesses,
    _same_weakness,
    sync_practice_for_interview,
)


practice_app = FastAPI()
practice_app.include_router(interviews_api.router, prefix="/api")
practice_app.include_router(practice_api.router, prefix="/api")


def test_weakness_comparison_treats_synonymous_wording_as_the_same_issue() -> None:
    assert _same_weakness("缺少量化结果", "没有给出可量化的结果")

    resolved, remaining = _compare_weaknesses(
        ["缺少量化结果"],
        ["没有给出可量化的结果"],
        target_title="缺少量化结果",
        target_improved=True,
    )

    assert resolved == []
    assert remaining == ["没有给出可量化的结果"]


def test_weakness_comparison_requires_target_improvement_and_keeps_sets_disjoint() -> None:
    unresolved, remaining = _compare_weaknesses(
        ["缺少量化结果"],
        [],
        target_title="缺少量化结果",
        target_improved=False,
    )
    assert unresolved == []
    assert remaining == ["缺少量化结果"]

    resolved, remaining = _compare_weaknesses(
        ["缺少量化结果"],
        [],
        target_title="缺少量化结果",
        target_improved=True,
    )
    assert resolved == ["缺少量化结果"]
    assert remaining == []


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_context(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'practice.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as db:
        db.add_all([
            User(id=1, email="practice-1@example.com", username="practice-1", password_hash="hash"),
            User(id=2, email="practice-2@example.com", username="practice-2", password_hash="hash"),
        ])
        await db.commit()
        source_session = InterviewSession(
            user_id=1,
            target_position="Python 后端工程师",
            difficulty="medium",
            mode="mock",
            interview_type="project_deep_dive",
            status="finished",
            resume_snapshot_json=json.dumps({"parsed": {"skills": ["Python"]}}, ensure_ascii=False),
            job_description_snapshot_json=json.dumps(
                {"parsed": {"must_have_skills": ["Python", "MySQL"]}}, ensure_ascii=False
            ),
        )
        foreign_session = InterviewSession(
            user_id=2,
            target_position="Go 后端工程师",
            difficulty="medium",
            status="finished",
        )
        db.add_all([source_session, foreign_session])
        await db.commit()

        source_score = InterviewScore(
            session_id=source_session.id,
            question_index=1,
            question="请说明项目结果以及你如何验证效果。",
            answer="我完成了接口改造，但没有补充结果指标。",
            dimension="项目结果与复盘",
            score=60,
            sub_scores=json.dumps({"专业准确性": 62, "表达清晰度": 58}, ensure_ascii=False),
            reason="回答缺少量化结果。",
            weaknesses=json.dumps(["缺少量化结果", "方案取舍不清晰"], ensure_ascii=False),
            suggestions=json.dumps(["补充真实指标与验证方法"], ensure_ascii=False),
        )
        foreign_score = InterviewScore(
            session_id=foreign_session.id,
            question_index=1,
            question="Go 问题",
            answer="回答",
            dimension="技术基础",
            score=80,
            sub_scores="{}",
            reason="理由",
            weaknesses="[]",
            suggestions="[]",
        )
        db.add_all([source_score, foreign_score])
        await db.commit()

        report = InterviewReport(
            session_id=source_session.id,
            total_score=60,
            summary="原始报告总结",
            strengths=json.dumps(["基础知识清楚"], ensure_ascii=False),
            weaknesses=json.dumps(["缺少量化结果", "方案取舍不清晰"], ensure_ascii=False),
            suggestions=json.dumps(["补充指标"], ensure_ascii=False),
            dimension_scores_json=json.dumps(
                [{"dimension": "项目结果与复盘", "score": 60}], ensure_ascii=False
            ),
            learning_path=json.dumps(["练习 STAR"], ensure_ascii=False),
            sample_answer="示范",
        )
        foreign_report = InterviewReport(
            session_id=foreign_session.id,
            total_score=80,
            summary="其他报告",
            strengths="[]",
            weaknesses=json.dumps(["其他短板"], ensure_ascii=False),
            suggestions="[]",
            learning_path="[]",
            sample_answer="示范",
        )
        db.add_all([report, foreign_report])
        await db.commit()

        review = QuestionReview(
            report_id=report.id,
            session_id=source_session.id,
            score_id=source_score.id,
            **build_question_review_payload(source_score),
        )
        foreign_review = QuestionReview(
            report_id=foreign_report.id,
            session_id=foreign_session.id,
            score_id=foreign_score.id,
            **build_question_review_payload(foreign_score),
        )
        db.add_all([review, foreign_review])
        await db.commit()
        source_ids = {
            "session": source_session.id,
            "score": source_score.id,
            "report": report.id,
            "review": review.id,
            "foreign_report": foreign_report.id,
            "foreign_review": foreign_review.id,
        }
        original_report = {
            "summary": report.summary,
            "weaknesses": report.weaknesses,
            "total_score": report.total_score,
        }

    current = {"user_id": 1}

    async def override_get_db():
        async with session_factory() as db:
            yield db

    async def override_current_user():
        return SimpleNamespace(id=current["user_id"])

    practice_app.dependency_overrides[get_db] = override_get_db
    practice_app.dependency_overrides[get_current_user] = override_current_user
    transport = ASGITransport(app=practice_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory, source_ids, current, original_report
    practice_app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_report_weakness_creates_isolated_practice_with_frozen_source(test_context) -> None:
    client, session_factory, source_ids, _current, original_report = test_context
    response = await client.post(
        "/api/practice/from-report",
        json={
            "report_id": source_ids["report"],
            "weakness_key": "report_weakness_1",
            "weakness_title": "缺少量化结果",
        },
    )

    assert response.status_code == 201
    created = response.json()
    assert created["status"] == "not_started"
    assert created["practice_session_id"]
    comparison = (await client.get(f"/api/practice/{created['id']}/comparison")).json()
    assert comparison["status"] == "not_started"
    assert comparison["practice_session_id"] == created["practice_session_id"]
    assert comparison["retest_session_id"] is None
    async with session_factory() as db:
        practice = await db.get(PracticeSession, created["id"])
        session = await db.get(InterviewSession, created["practice_session_id"])
        report = await db.get(InterviewReport, source_ids["report"])
        snapshot = json.loads(practice.source_snapshot_json)
        assert session.session_purpose == "weakness_practice"
        assert session.parent_session_id == source_ids["session"]
        assert session.source_report_id == source_ids["report"]
        assert session.mode == "training"
        assert json.loads(session.practice_context_json) == snapshot
        state = build_state_from_session(session, None, messages=[])
        assert state["practice_context"] == snapshot
        assert snapshot["source_question"] == "请说明项目结果以及你如何验证效果。"
        assert snapshot["source_answer"] == "我完成了接口改造，但没有补充结果指标。"
        assert snapshot["deduction_reasons"]
        assert snapshot["target_position"] == "Python 后端工程师"
        assert report.summary == original_report["summary"]
        assert report.weaknesses == original_report["weaknesses"]
        assert report.total_score == original_report["total_score"]

    started = await client.post(f"/api/practice/{created['id']}/start")
    assert started.status_code == 200
    assert started.json()["practice_id"] == created["id"]
    assert started.json()["source_practice_id"] == created["id"]
    fetched_session = await client.get(f"/api/interviews/{created['practice_session_id']}")
    assert fetched_session.status_code == 200
    assert fetched_session.json()["practice_id"] == created["id"]
    listed = (await client.get("/api/practice")).json()
    assert listed[0]["status"] == "practicing"


@pytest.mark.anyio
async def test_question_practice_validates_saved_source_fields(test_context) -> None:
    client, _session_factory, source_ids, _current, _original_report = test_context
    payload = {
        "report_id": source_ids["report"],
        "question_review_id": source_ids["review"],
        "score_id": source_ids["score"],
        "question_index": 1,
        "weakness_key": "question_1",
        "weakness_title": "缺少量化结果",
        "practice_mode": "similar_question",
    }

    assert (await client.post(
        "/api/practice/from-question-review", json={**payload, "score_id": source_ids["score"] + 99}
    )).status_code == 409
    assert (await client.post(
        "/api/practice/from-question-review", json={**payload, "weakness_key": "forged"}
    )).status_code == 422
    response = await client.post("/api/practice/from-question-review", json=payload)
    assert response.status_code == 201


@pytest.mark.anyio
async def test_retest_comparison_is_aligned_and_immutable(test_context) -> None:
    client, session_factory, source_ids, _current, original_report = test_context
    created = (await client.post(
        "/api/practice/from-question-review",
        json={
            "report_id": source_ids["report"],
            "question_review_id": source_ids["review"],
            "score_id": source_ids["score"],
            "question_index": 1,
            "weakness_key": "question_1",
            "weakness_title": "缺少量化结果",
            "practice_mode": "repeat_question",
        },
    )).json()
    practice_id = created["id"]

    blocked = await client.post(f"/api/practice/{practice_id}/start-retest")
    assert blocked.status_code == 409

    async with session_factory() as db:
        practice_session = await db.get(InterviewSession, created["practice_session_id"])
        practice_session.status = "finished"
        await sync_practice_for_interview(db, practice_session)
        practice = await db.get(PracticeSession, practice_id)
        assert practice.status == "ready_for_retest"
        await db.commit()

    first_retest = await client.post(f"/api/practice/{practice_id}/start-retest")
    assert first_retest.status_code == 200
    retest = first_retest.json()
    assert retest["session_purpose"] == "retest"
    assert retest["mode"] == "mock"
    assert retest["practice_id"] == practice_id
    assert (await client.post(f"/api/practice/{practice_id}/start-retest")).json()["id"] == retest["id"]

    async with session_factory() as db:
        retest_session = await db.get(InterviewSession, retest["id"])
        retest_session.status = "finished"
        retest_scores = [
            InterviewScore(
                session_id=retest["id"],
                question_index=1,
                question="同类场景一",
                answer="背景是接口延迟，我采用缓存方案，因为读多写少，最终延迟降低 30%。",
                dimension="项目结果与复盘",
                score=78,
                sub_scores=json.dumps({"专业准确性": 80, "表达清晰度": 76}, ensure_ascii=False),
                reason="结果较完整",
                weaknesses=json.dumps(["方案取舍不清晰"], ensure_ascii=False),
                suggestions="[]",
            ),
            InterviewScore(
                session_id=retest["id"],
                question_index=2,
                question="同类场景二",
                answer="我负责验证，最终通过压测确认指标，并复盘了异常路径。",
                dimension="项目结果与复盘",
                score=82,
                sub_scores=json.dumps({"专业准确性": 84, "表达清晰度": 80}, ensure_ascii=False),
                reason="验证充分",
                weaknesses=json.dumps(["方案取舍不清晰"], ensure_ascii=False),
                suggestions="[]",
            ),
        ]
        db.add_all(retest_scores)
        await db.flush()
        await sync_practice_for_interview(db, retest_session)
        practice = await db.get(PracticeSession, practice_id)
        assert practice.status == "completed"
        assert practice.comparison_json
        await db.commit()

    comparison_response = await client.get(f"/api/practice/{practice_id}/comparison")
    assert comparison_response.status_code == 200
    comparison = comparison_response.json()
    assert comparison["status"] == "completed"
    assert comparison["before"]["session_id"] == source_ids["session"]
    assert comparison["before"]["score"] == 60
    assert comparison["after"]["session_id"] == retest["id"]
    assert comparison["after"]["score"] == 80
    assert comparison["delta"]["score"] == 20
    assert "缺少量化结果" in comparison["delta"]["resolved_weaknesses"]
    assert comparison["delta"]["remaining_weaknesses"] == ["方案取舍不清晰"]
    assert comparison["delta"]["sub_scores"] == {"专业准确性": 20, "表达清晰度": 20}
    assert comparison["after"]["answer_structure"]

    async with session_factory() as db:
        source_score = await db.get(InterviewScore, source_ids["score"])
        source_score.score = 5
        source_score.answer = "后续被修改的原回答"
        changed_retest_score = await db.scalar(
            select(InterviewScore).where(InterviewScore.session_id == retest["id"])
        )
        changed_retest_score.score = 10
        report = await db.get(InterviewReport, source_ids["report"])
        report.weaknesses = json.dumps(["后续修改的短板"], ensure_ascii=False)
        await db.commit()

    assert (await client.get(f"/api/practice/{practice_id}/comparison")).json() == comparison
    async with session_factory() as db:
        practice = await db.get(PracticeSession, practice_id)
        report = await db.get(InterviewReport, source_ids["report"])
        assert practice.after_score == 80
        assert json.loads(practice.comparison_json) == comparison
        assert report.summary == original_report["summary"]
        assert report.total_score == original_report["total_score"]


@pytest.mark.anyio
async def test_practice_permissions_are_isolated(test_context) -> None:
    client, _session_factory, source_ids, current, _original_report = test_context
    created = (await client.post(
        "/api/practice/from-report",
        json={
            "report_id": source_ids["report"],
            "weakness_key": "report_weakness_1",
            "weakness_title": "缺少量化结果",
        },
    )).json()
    current["user_id"] = 2

    assert (await client.get(f"/api/practice/{created['id']}/comparison")).status_code == 404
    assert (await client.post(f"/api/practice/{created['id']}/start")).status_code == 404
    assert (await client.post(f"/api/practice/{created['id']}/start-retest")).status_code == 404
    assert (await client.post(
        "/api/practice/from-report",
        json={
            "report_id": source_ids["report"],
            "weakness_key": "report_weakness_1",
            "weakness_title": "缺少量化结果",
        },
    )).status_code == 404
    assert all(item["id"] != created["id"] for item in (await client.get("/api/practice")).json())
