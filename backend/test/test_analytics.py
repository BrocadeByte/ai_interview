import sys
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401, E402
from app.api import analytics as analytics_api  # noqa: E402
from app.api.deps import get_current_admin_user, get_current_user  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.models.analytics import AnalyticsEvent  # noqa: E402
from app.models.interview import InterviewSession  # noqa: E402
from app.models.job_description import JobDescription  # noqa: E402
from app.models.question_review import QuestionReview  # noqa: E402
from app.models.report import InterviewReport  # noqa: E402
from app.models.resume import Resume  # noqa: E402
from app.models.score import InterviewScore  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.analytics_service import (  # noqa: E402
    build_analytics_metrics,
    record_analytics_event,
)


analytics_app = FastAPI()
analytics_app.include_router(analytics_api.router, prefix="/api")


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def analytics_context(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'analytics.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    current = {"user_id": 1}

    async def override_get_db():
        async with session_factory() as db:
            yield db

    async def override_current_user():
        async with session_factory() as db:
            return await db.get(User, current["user_id"])

    async def override_current_admin():
        async with session_factory() as db:
            return await db.get(User, 1)

    async with session_factory() as db:
        db.add_all(
            [
                User(
                    id=1,
                    email="analytics-1@example.com",
                    username="analytics-1",
                    password_hash="hash",
                    is_admin=True,
                ),
                User(
                    id=2,
                    email="analytics-2@example.com",
                    username="analytics-2",
                    password_hash="hash",
                ),
            ]
        )
        await db.commit()
        session = InterviewSession(
            user_id=1,
            target_position="Python 后端工程师",
            difficulty="medium",
            status="finished",
        )
        foreign_session = InterviewSession(
            user_id=2,
            target_position="Go 后端工程师",
            difficulty="medium",
            status="finished",
        )
        db.add_all([session, foreign_session])
        await db.commit()
        score = InterviewScore(
            session_id=session.id,
            question_index=1,
            question="请介绍项目。",
            answer="项目回答。",
            dimension="项目经验",
            score=70,
            sub_scores="{}",
            reason="缺少指标",
            weaknesses="[]",
            suggestions="[]",
        )
        db.add(score)
        await db.commit()
        report = InterviewReport(
            session_id=session.id,
            total_score=70,
            summary="总结",
            strengths="[]",
            weaknesses="[]",
            suggestions="[]",
            learning_path="[]",
            sample_answer="示范",
        )
        foreign_report = InterviewReport(
            session_id=foreign_session.id,
            total_score=80,
            summary="总结",
            strengths="[]",
            weaknesses="[]",
            suggestions="[]",
            learning_path="[]",
            sample_answer="示范",
        )
        db.add_all([report, foreign_report])
        await db.commit()
        review = QuestionReview(
            report_id=report.id,
            session_id=session.id,
            score_id=score.id,
            question_index=1,
            dimension="项目经验",
            question="请介绍项目。",
            answer="项目回答。",
            score=70,
            sub_scores_json="{}",
            deduction_reasons_json='["缺少指标"]',
            suggested_structure_json='["背景"]',
            sample_answer="使用真实经历作答。",
            weaknesses_json='["缺少指标"]',
            weakness_key="project_metrics",
            practice_seed_json="{}",
        )
        resume = Resume(
            user_id=1,
            title="简历",
            source_type="paste",
            raw_text="Python 项目",
            status="parsed",
        )
        job_description = JobDescription(
            user_id=1,
            title="后端岗位",
            raw_text="熟悉 Python",
            target_position="Python 后端工程师",
            status="parsed",
        )
        db.add_all([review, resume, job_description])
        await db.commit()
        ids = {
            "report": report.id,
            "review": review.id,
            "foreign_report": foreign_report.id,
            "resume": resume.id,
            "job_description": job_description.id,
        }

    analytics_app.dependency_overrides[get_db] = override_get_db
    analytics_app.dependency_overrides[get_current_user] = override_current_user
    analytics_app.dependency_overrides[get_current_admin_user] = override_current_admin
    transport = ASGITransport(app=analytics_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory, current, ids

    analytics_app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_client_events_are_idempotent_and_validate_owned_subjects(analytics_context) -> None:
    client, session_factory, current, ids = analytics_context
    event_id = "f89ce807-8afa-4db4-99f3-6f7f15b7d609"
    payload = {
        "event_name": "question_review_expanded",
        "client_event_id": event_id,
        "report_id": ids["report"],
        "question_review_id": ids["review"],
    }
    assert (await client.post("/api/analytics/events", json=payload)).status_code == 204
    assert (await client.post("/api/analytics/events", json=payload)).status_code == 204

    profile_event = await client.post(
        "/api/analytics/events",
        json={
            "event_name": "profile_applied",
            "client_event_id": "6e97bb75-5b7d-4e64-81fa-44bed2b8fcec",
            "resume_id": ids["resume"],
            "job_description_id": ids["job_description"],
        },
    )
    assert profile_event.status_code == 204

    current["user_id"] = 2
    forbidden = await client.post(
        "/api/analytics/events",
        json={**payload, "client_event_id": "71f71463-14cc-4d3c-a218-777eaf7e5101"},
    )
    assert forbidden.status_code == 404
    current["user_id"] = 1

    spoofed = await client.post(
        "/api/analytics/events",
        json={
            "event_name": "interview_started",
            "client_event_id": "67fa714d-c89e-4b74-8da5-f82f2ac80172",
        },
    )
    assert spoofed.status_code == 422

    async with session_factory() as db:
        count = await db.scalar(
            select(func.count(AnalyticsEvent.id)).where(
                AnalyticsEvent.event_name == "question_review_expanded"
            )
        )
        assert count == 1


@pytest.mark.anyio
async def test_metrics_cover_required_funnels_and_seven_day_repractice(tmp_path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'metrics.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    now = datetime(2026, 8, 23, 12)

    async with session_factory() as db:
        users = [
            User(id=11, email="metric-11@example.com", username="m11", password_hash="hash", created_at=datetime(2026, 8, 1)),
            User(id=12, email="metric-12@example.com", username="m12", password_hash="hash", created_at=datetime(2026, 8, 3)),
            User(id=13, email="metric-13@example.com", username="m13", password_hash="hash", created_at=datetime(2026, 8, 7)),
            User(id=14, email="metric-14@example.com", username="m14", password_hash="hash", created_at=datetime(2026, 7, 30)),
        ]
        db.add_all(users)
        await db.commit()
        sessions = [
            InterviewSession(id=101, user_id=11, target_position="后端", difficulty="medium", status="finished", session_purpose="full_interview"),
            InterviewSession(id=102, user_id=11, target_position="后端", difficulty="medium", status="finished", session_purpose="weakness_practice"),
            InterviewSession(id=103, user_id=12, target_position="前端", difficulty="medium", status="finished", session_purpose="full_interview"),
            InterviewSession(id=104, user_id=14, target_position="测试", difficulty="medium", status="finished", session_purpose="full_interview"),
        ]
        db.add_all(sessions)
        await db.commit()

        events = [
            ("resume_pasted", 11, None, None, datetime(2026, 8, 1, 9)),
            ("profile_auto_generated", 11, None, None, datetime(2026, 8, 1, 10)),
            ("profile_applied", 11, None, None, datetime(2026, 8, 1, 11)),
            ("interview_created", 11, 101, None, datetime(2026, 8, 1, 12)),
            ("interview_started", 11, 101, None, datetime(2026, 8, 1, 13)),
            ("interview_finished", 11, 101, None, datetime(2026, 8, 2, 13)),
            ("report_viewed", 11, 101, 201, datetime(2026, 8, 2, 14)),
            ("practice_created", 11, 102, 201, datetime(2026, 8, 3, 9)),
            ("interview_started", 11, 102, 201, datetime(2026, 8, 5, 9)),
            ("interview_created", 12, 103, None, datetime(2026, 8, 4, 8)),
            ("interview_started", 12, 103, None, datetime(2026, 8, 4, 9)),
            ("interview_finished", 12, 103, None, datetime(2026, 8, 6, 9)),
            ("report_viewed", 12, 103, 202, datetime(2026, 8, 6, 10)),
            ("interview_finished", 14, 104, None, datetime(2026, 8, 20, 9)),
        ]
        for index, (name, user_id, session_id, report_id, occurred_at) in enumerate(events):
            await record_analytics_event(
                db,
                event_name=name,
                user_id=user_id,
                session_id=session_id,
                report_id=report_id,
                deduplication_key=f"metric:{index}",
                occurred_at=occurred_at,
            )
        await db.commit()

        metrics = await build_analytics_metrics(
            db,
            period_start=date(2026, 8, 1),
            period_end=date(2026, 8, 22),
            now=now,
        )

    assert metrics.first_experience_funnel.registered_users == 3
    assert metrics.first_experience_funnel.first_interview_started_users == 2
    assert metrics.registration_to_first_interview_start.model_dump() == {
        "denominator": 3,
        "numerator": 2,
        "rate": 0.6667,
    }
    assert metrics.report_to_practice.model_dump() == {
        "denominator": 2,
        "numerator": 1,
        "rate": 0.5,
    }
    assert metrics.seven_day_repractice.model_dump() == {
        "denominator": 2,
        "numerator": 1,
        "rate": 0.5,
    }
    assert metrics.event_counts["interview_started"] == 3
    await engine.dispose()
