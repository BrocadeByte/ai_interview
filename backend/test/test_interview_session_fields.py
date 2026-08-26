import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401, E402
from app.api import interviews as interviews_api  # noqa: E402
from app.api.deps import get_current_user  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.models.interview import InterviewSession  # noqa: E402
from app.models.job_description import JobDescription  # noqa: E402
from app.models.report import InterviewReport  # noqa: E402
from app.models.resume import Resume  # noqa: E402
from app.models.user import User  # noqa: E402
from app.services.interview_state_service import build_state_from_session  # noqa: E402


interview_app = FastAPI()
interview_app.include_router(interviews_api.router, prefix="/api")


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_context(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'interview-fields.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with session_factory() as db:
        db.add_all(
            [
                User(id=1, email="session-1@example.com", username="session-1", password_hash="hash"),
                User(id=2, email="session-2@example.com", username="session-2", password_hash="hash"),
            ]
        )
        await db.commit()

        parsed_resume = {
            "skills": ["Python", "FastAPI"],
            "education": ["本科"],
            "projects": [],
            "work_experience": [],
            "experience_summary": "两年后端经验",
        }
        resume = Resume(
            user_id=1,
            title="后端简历",
            source_type="paste",
            raw_text="候选人掌握 Python 和 FastAPI。",
            parsed_json=json.dumps(parsed_resume, ensure_ascii=False),
            profile_patch_json=json.dumps({"skills": "Python, FastAPI"}, ensure_ascii=False),
            status="parsed",
        )
        other_resume = Resume(
            user_id=2,
            title="其他用户简历",
            source_type="paste",
            raw_text="Go",
            parsed_json=json.dumps({**parsed_resume, "skills": ["Go"]}, ensure_ascii=False),
            profile_patch_json=json.dumps({"skills": "Go"}, ensure_ascii=False),
            status="parsed",
        )
        jd = JobDescription(
            user_id=1,
            title="Python 后端",
            raw_text="要求 Python、FastAPI 与 MySQL。",
            parsed_json=json.dumps(
                {
                    "target_position": "Python 后端工程师",
                    "seniority": "中级",
                    "experience_requirements": "2 年以上",
                    "must_have_skills": ["Python", "FastAPI", "MySQL"],
                    "nice_to_have_skills": [],
                    "responsibilities": ["开发 API"],
                    "hard_requirements": [],
                    "interview_focus": ["项目深挖"],
                    "risk_points": [],
                },
                ensure_ascii=False,
            ),
            target_position="Python 后端工程师",
            status="parsed",
        )
        db.add_all([resume, other_resume, jd])
        await db.commit()

        parent = InterviewSession(
            user_id=1,
            target_position="Python 后端工程师",
            difficulty="medium",
            status="finished",
        )
        other_parent = InterviewSession(
            user_id=2,
            target_position="Go 后端工程师",
            difficulty="medium",
            status="finished",
        )
        db.add_all([parent, other_parent])
        await db.commit()
        report = InterviewReport(
            session_id=parent.id,
            total_score=70,
            summary="总结",
            strengths="优势",
            weaknesses="缺少量化结果",
            suggestions="补充指标",
            learning_path="学习路径",
            sample_answer="示范回答",
        )
        other_report = InterviewReport(
            session_id=other_parent.id,
            total_score=80,
            summary="总结",
            strengths="优势",
            weaknesses="短板",
            suggestions="建议",
            learning_path="学习路径",
            sample_answer="示范回答",
        )
        db.add_all([report, other_report])
        await db.commit()
        source_ids = {
            "resume": resume.id,
            "other_resume": other_resume.id,
            "jd": jd.id,
            "parent": parent.id,
            "other_parent": other_parent.id,
            "report": report.id,
            "other_report": other_report.id,
        }

    current = {"user_id": 1}

    async def override_get_db():
        async with session_factory() as db:
            yield db

    async def override_current_user():
        return SimpleNamespace(id=current["user_id"])

    interview_app.dependency_overrides[get_db] = override_get_db
    interview_app.dependency_overrides[get_current_user] = override_current_user
    transport = ASGITransport(app=interview_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory, source_ids
    interview_app.dependency_overrides.clear()
    await engine.dispose()


@pytest.mark.anyio
async def test_create_extended_session_freezes_sources_and_injects_graph_state(test_context) -> None:
    client, session_factory, source_ids = test_context
    response = await client.post(
        "/api/interviews",
        json={
            "target_position": "Python 后端工程师",
            "difficulty": "hard",
            "mode": "mock",
            "interview_type": "project_deep_dive",
            "resume_id": source_ids["resume"],
            "job_description_id": source_ids["jd"],
            "parent_session_id": source_ids["parent"],
            "source_report_id": source_ids["report"],
            "source_weakness_key": "project_metrics",
            "session_purpose": "retest",
            "comparison_group_id": "comparison-2026-08",
        },
    )

    assert response.status_code == 201
    data = response.json()
    assert data["mode"] == "mock"
    assert data["interview_type"] == "project_deep_dive"
    assert data["resume_id"] == source_ids["resume"]
    assert data["job_description_id"] == source_ids["jd"]
    assert data["parent_session_id"] == source_ids["parent"]
    assert data["source_report_id"] == source_ids["report"]
    assert data["source_weakness_key"] == "project_metrics"
    assert data["session_purpose"] == "retest"
    assert data["comparison_group_id"] == "comparison-2026-08"

    async with session_factory() as db:
        resume = await db.get(Resume, source_ids["resume"])
        resume.raw_text = "后续被修改的简历"
        resume.parsed_json = json.dumps(
            {
                "skills": ["Go"],
                "education": [],
                "projects": [],
                "work_experience": [],
                "experience_summary": "已修改",
            },
            ensure_ascii=False,
        )
        await db.commit()
        session = await db.get(InterviewSession, data["id"])
        state = build_state_from_session(session, profile=None, messages=[])

    assert state["mode"] == "mock"
    assert state["interview_type"] == "project_deep_dive"
    assert state["resume"]["raw_text"] == "候选人掌握 Python 和 FastAPI。"
    assert state["resume"]["parsed"]["skills"] == ["Python", "FastAPI"]
    assert state["target_job"]["parsed"]["must_have_skills"] == ["Python", "FastAPI", "MySQL"]
    assert state["parent_session_id"] == source_ids["parent"]
    assert state["source_report_id"] == source_ids["report"]
    assert state["source_weakness_key"] == "project_metrics"
    assert state["session_purpose"] == "retest"
    assert state["comparison_group_id"] == "comparison-2026-08"


@pytest.mark.anyio
async def test_legacy_create_and_state_use_compatible_defaults(test_context) -> None:
    client, session_factory, _source_ids = test_context
    response = await client.post(
        "/api/interviews",
        json={"target_position": "通用后端工程师", "difficulty": "easy"},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["mode"] == "training"
    assert data["interview_type"] == "mixed"
    assert data["session_purpose"] == "full_interview"
    assert data["resume_id"] is None
    assert data["job_description_id"] is None

    async with session_factory() as db:
        session = await db.get(InterviewSession, data["id"])
        state = build_state_from_session(session, profile=None, messages=[])
    assert state["mode"] == "training"
    assert state["interview_type"] == "mixed"
    assert state["resume"] is None
    assert state["target_job"] == {"position": "通用后端工程师"}
    assert state["session_purpose"] == "full_interview"

    listed = await client.get("/api/interviews")
    assert listed.status_code == 200
    legacy_item = next(item for item in listed.json() if item["id"] == data["id"])
    assert legacy_item["mode"] == "training"
    assert legacy_item["interview_type"] == "mixed"


@pytest.mark.anyio
async def test_extended_session_rejects_invalid_values_and_cross_user_sources(test_context) -> None:
    client, _session_factory, source_ids = test_context
    base = {"target_position": "后端工程师", "difficulty": "medium"}

    invalid_mode = await client.post("/api/interviews", json={**base, "mode": "live"})
    invalid_type = await client.post("/api/interviews", json={**base, "interview_type": "algorithm"})
    invalid_purpose = await client.post("/api/interviews", json={**base, "session_purpose": "practice"})
    assert invalid_mode.status_code == 422
    assert invalid_type.status_code == 422
    assert invalid_purpose.status_code == 422

    foreign_resume = await client.post(
        "/api/interviews", json={**base, "resume_id": source_ids["other_resume"]}
    )
    foreign_parent = await client.post(
        "/api/interviews", json={**base, "parent_session_id": source_ids["other_parent"]}
    )
    foreign_report = await client.post(
        "/api/interviews", json={**base, "source_report_id": source_ids["other_report"]}
    )
    assert foreign_resume.status_code == 404
    assert foreign_parent.status_code == 404
    assert foreign_report.status_code == 404
