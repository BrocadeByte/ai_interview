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
from app.api import profiles as profiles_api  # noqa: E402
from app.api.deps import get_current_user  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.models.job_description import JobDescription  # noqa: E402
from app.models.profile import UserProfile  # noqa: E402
from app.models.resume import Resume  # noqa: E402
from app.models.user import User  # noqa: E402


profile_app = FastAPI()
profile_app.include_router(profiles_api.router, prefix="/api")


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_context(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'profiles.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db:
        db.add_all(
            [
                User(id=1, email="profile-1@example.com", username="profile-1", password_hash="hash"),
                User(id=2, email="profile-2@example.com", username="profile-2", password_hash="hash"),
            ]
        )
        await db.commit()

    state = {"user_id": 1}

    async def override_get_db():
        async with session_factory() as db:
            yield db

    async def override_current_user():
        return SimpleNamespace(id=state["user_id"])

    profile_app.dependency_overrides[get_db] = override_get_db
    profile_app.dependency_overrides[get_current_user] = override_current_user
    transport = ASGITransport(app=profile_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, session_factory, state
    profile_app.dependency_overrides.clear()
    await engine.dispose()


def resume_profile_patch() -> dict:
    return {
        "education": "本科",
        "major": "计算机科学与技术",
        "experience_years": 3,
        "target_position": "后端开发工程师",
        "skills": "Python, FastAPI, MySQL",
        "projects": "AI 面试系统：负责 API 和工作流开发",
        "self_evaluation": "具备后端和 AI 应用开发经验",
    }


def parsed_resume() -> dict:
    return {
        "skills": ["Python", "FastAPI", "MySQL"],
        "education": ["计算机科学与技术本科"],
        "projects": [
            {
                "name": "AI 面试系统",
                "role": "后端开发",
                "description": "负责 API 和工作流开发",
                "tech_stack": ["Python", "FastAPI"],
                "highlights": [],
            }
        ],
        "work_experience": [],
        "experience_summary": "具备后端项目经验",
    }


def parsed_jd() -> dict:
    return {
        "target_position": "Python 后端工程师",
        "seniority": "中级",
        "experience_requirements": "3 年以上",
        "must_have_skills": ["Python", "FastAPI", "Redis"],
        "nice_to_have_skills": ["LangGraph"],
        "responsibilities": ["负责 API 和 AI 工作流开发"],
        "hard_requirements": ["熟悉关系型数据库"],
        "interview_focus": ["项目深挖", "技术基础"],
        "risk_points": ["需要验证高并发经验"],
    }


async def seed_sources(session_factory, *, user_id: int = 1) -> tuple[int, int]:
    async with session_factory() as db:
        resume = Resume(
            user_id=user_id,
            title="候选人简历",
            source_type="paste",
            raw_text="候选人真实简历正文",
            parsed_json=json.dumps(parsed_resume(), ensure_ascii=False),
            profile_patch_json=json.dumps(resume_profile_patch(), ensure_ascii=False),
            status="parsed",
        )
        jd = JobDescription(
            user_id=user_id,
            title="Python 后端工程师",
            raw_text="岗位 JD 正文",
            parsed_json=json.dumps(parsed_jd(), ensure_ascii=False),
            target_position="Python 后端工程师",
            status="parsed",
        )
        db.add_all([resume, jd])
        await db.commit()
        await db.refresh(resume)
        await db.refresh(jd)
        return resume.id, jd.id


@pytest.mark.anyio
async def test_auto_generate_merges_sources_without_persisting_profile(test_context) -> None:
    client, session_factory, _state = test_context
    async with session_factory() as db:
        db.add(
            UserProfile(
                user_id=1,
                target_position="手动目标岗位",
                target_city="上海",
                expected_salary="25k-30k",
                skills="手动技能",
            )
        )
        await db.commit()
    resume_id, jd_id = await seed_sources(session_factory)

    response = await client.post(
        "/api/profile/auto-generate",
        json={
            "resume_id": resume_id,
            "job_description_id": jd_id,
            "target_position": "AI 应用后端工程师",
        },
    )

    assert response.status_code == 200
    data = response.json()
    patch = data["profile_patch"]
    assert patch["target_position"] == "AI 应用后端工程师"
    assert patch["skills"] == "Python, FastAPI, MySQL"
    assert patch["target_city"] == "上海"
    assert patch["expected_salary"] == "25k-30k"
    assert data["completeness"] == 100
    assert data["missing_fields"] == []
    assert data["field_sources"]["skills"] == "resume"
    assert data["field_sources"]["target_city"] == "manual"
    assert data["field_sources"]["target_position"] == "request"
    assert any("Redis" in warning for warning in data["warnings"])
    assert any("高并发" in warning for warning in data["warnings"])
    assert "Redis" not in patch["skills"]

    async with session_factory() as db:
        unchanged = await db.scalar(select(UserProfile).where(UserProfile.user_id == 1))
        assert unchanged is not None
        assert unchanged.target_position == "手动目标岗位"
        assert unchanged.skills == "手动技能"


@pytest.mark.anyio
async def test_partial_profile_update_keeps_unselected_fields(test_context) -> None:
    client, session_factory, _state = test_context
    async with session_factory() as db:
        profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == 1))
        if profile is None:
            profile = UserProfile(user_id=1)
            db.add(profile)
        profile.target_city = "北京"
        profile.skills = "Python"
        await db.commit()

    response = await client.put("/api/profile/me", json={"skills": "Python, FastAPI"})

    assert response.status_code == 200
    assert response.json()["skills"] == "Python, FastAPI"
    assert response.json()["target_city"] == "北京"


@pytest.mark.anyio
async def test_low_completeness_returns_actionable_missing_fields(test_context) -> None:
    client, _session_factory, state = test_context
    state["user_id"] = 2

    response = await client.post(
        "/api/profile/auto-generate",
        json={"target_position": "前端开发工程师"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["completeness"] == 20
    assert "skills" in data["missing_fields"]
    assert "projects" in data["missing_fields"]
    assert any("低于 70%" in warning and "技能栈" in warning for warning in data["warnings"])
    assert any("未提供已解析简历" in warning for warning in data["warnings"])


@pytest.mark.anyio
async def test_auto_generate_enforces_source_ownership(test_context) -> None:
    client, session_factory, state = test_context
    state["user_id"] = 1
    resume_id, jd_id = await seed_sources(session_factory)
    state["user_id"] = 2

    assert (
        await client.post("/api/profile/auto-generate", json={"resume_id": resume_id})
    ).status_code == 404
    assert (
        await client.post(
            "/api/profile/auto-generate", json={"job_description_id": jd_id}
        )
    ).status_code == 404


@pytest.mark.anyio
async def test_auto_generate_rejects_unparsed_or_invalid_snapshots(test_context) -> None:
    client, session_factory, state = test_context
    state["user_id"] = 1
    async with session_factory() as db:
        pending_resume = Resume(
            user_id=1,
            title="待解析简历",
            source_type="paste",
            raw_text="正文",
            status="pending",
        )
        invalid_jd = JobDescription(
            user_id=1,
            title="损坏 JD",
            raw_text="正文",
            parsed_json="not-json",
            target_position="后端工程师",
            status="parsed",
        )
        db.add_all([pending_resume, invalid_jd])
        await db.commit()
        await db.refresh(pending_resume)
        await db.refresh(invalid_jd)

    pending_response = await client.post(
        "/api/profile/auto-generate", json={"resume_id": pending_resume.id}
    )
    invalid_response = await client.post(
        "/api/profile/auto-generate", json={"job_description_id": invalid_jd.id}
    )

    assert pending_response.status_code == 409
    assert invalid_response.status_code == 409


@pytest.mark.anyio
async def test_profile_write_rejects_oversized_generated_fields(test_context) -> None:
    client, _session_factory, state = test_context
    state["user_id"] = 1

    response = await client.put("/api/profile/me", json={"skills": "x" * 4_001})

    assert response.status_code == 422
