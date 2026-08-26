import json
import sys
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401, E402
from app.core.database import Base  # noqa: E402
from app.models.profile import UserProfile  # noqa: E402
from app.models.resume import Resume  # noqa: E402
from app.models.user import User  # noqa: E402
from app.schemas.resume import ResumeParseOutput  # noqa: E402
from app.services.resume_service import ResumeParseTimeoutError  # noqa: E402
from app.workers import resume_worker  # noqa: E402


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def worker_context(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'worker.db').as_posix()}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with session_factory() as db:
        db.add_all(
            [
                User(id=1, email="worker-1@example.com", username="worker-1", password_hash="hash"),
                User(id=2, email="worker-2@example.com", username="worker-2", password_hash="hash"),
                UserProfile(user_id=1, skills="Existing skill", target_position="Existing position"),
            ]
        )
        await db.commit()
    try:
        yield session_factory
    finally:
        await engine.dispose()


def parse_output() -> ResumeParseOutput:
    return ResumeParseOutput.model_validate(
        {
            "parsed": {
                "skills": ["Python"],
                "education": [],
                "projects": [],
                "work_experience": [],
                "experience_summary": "Python 后端经验",
            },
            "profile_patch": {
                "target_position": "Python 后端工程师",
                "skills": "Python",
            },
        }
    )


async def seed_resume(
    session_factory,
    *,
    user_id: int = 1,
    status: str = "pending",
    parsed_json: str | None = None,
    profile_patch_json: str | None = None,
) -> int:
    async with session_factory() as db:
        resume = Resume(
            user_id=user_id,
            title="Worker resume",
            source_type="paste",
            raw_text="Python FastAPI 项目经验",
            status=status,
            parsed_json=parsed_json,
            profile_patch_json=profile_patch_json,
        )
        db.add(resume)
        await db.commit()
        await db.refresh(resume)
        return resume.id


@pytest.mark.anyio
async def test_worker_success_parses_resume_without_changing_profile(worker_context, monkeypatch) -> None:
    resume_id = await seed_resume(worker_context)
    calls: list[str] = []

    async def fake_parse(raw_text: str) -> ResumeParseOutput:
        calls.append(raw_text)
        return parse_output()

    monkeypatch.setattr(resume_worker, "parse_resume_text", fake_parse)
    async with worker_context() as db:
        result = await resume_worker.process_resume_task(
            db,
            resume_id=resume_id,
            user_id=1,
            attempt=1,
        )

    assert result == resume_worker.ResumeTaskResult.PARSED
    assert calls == ["Python FastAPI 项目经验"]
    async with worker_context() as db:
        resume = await db.get(Resume, resume_id)
        profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == 1))
        assert resume is not None and resume.status == "parsed"
        assert json.loads(resume.parsed_json or "{}")["skills"] == ["Python"]
        assert json.loads(resume.profile_patch_json or "{}")["target_position"] == "Python 后端工程师"
        assert resume.error_message is None
        assert profile is not None
        assert profile.skills == "Existing skill"
        assert profile.target_position == "Existing position"


@pytest.mark.anyio
async def test_invalid_json_failure_clears_stale_parse_fields(worker_context, monkeypatch) -> None:
    resume_id = await seed_resume(
        worker_context,
        parsed_json='{"stale": true}',
        profile_patch_json='{"skills": "stale"}',
    )

    async def invalid_json(_raw_text: str) -> ResumeParseOutput:
        raise json.JSONDecodeError("invalid LLM JSON", "not-json", 0)

    monkeypatch.setattr(resume_worker, "parse_resume_text", invalid_json)
    async with worker_context() as db:
        result = await resume_worker.process_resume_task(
            db,
            resume_id=resume_id,
            user_id=1,
            attempt=1,
        )

    assert result == resume_worker.ResumeTaskResult.FAILED
    async with worker_context() as db:
        resume = await db.get(Resume, resume_id)
        profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == 1))
        assert resume is not None and resume.status == "failed"
        assert resume.error_message == "AI 未能识别简历内容，请检查文件内容后重试"
        assert resume.parsed_json is None
        assert resume.profile_patch_json is None
        assert profile is not None and profile.skills == "Existing skill"


@pytest.mark.anyio
async def test_full_parse_timeout_fails_without_exceeding_frontend_budget(
    worker_context,
    monkeypatch,
) -> None:
    resume_id = await seed_resume(worker_context)
    parse_calls = 0
    retry_messages: list[tuple[int, int, int]] = []
    dead_messages: list[tuple[int, int, int]] = []

    async def timeout_parse(_raw_text: str) -> ResumeParseOutput:
        nonlocal parse_calls
        parse_calls += 1
        raise ResumeParseTimeoutError("private provider timeout details")

    async def fake_retry(resume_id: int, user_id: int, attempt: int) -> None:
        retry_messages.append((resume_id, user_id, attempt))

    async def fake_dead(resume_id: int, user_id: int, attempt: int) -> None:
        dead_messages.append((resume_id, user_id, attempt))

    monkeypatch.setattr(resume_worker, "parse_resume_text", timeout_parse)
    monkeypatch.setattr(resume_worker, "publish_resume_parse_retry", fake_retry)
    monkeypatch.setattr(resume_worker, "publish_resume_parse_dead_letter", fake_dead)

    result = await resume_worker.handle_resume_task(
        resume_id=resume_id,
        user_id=1,
        attempt=1,
        session_factory=worker_context,
    )

    assert result == resume_worker.ResumeTaskResult.FAILED
    assert parse_calls == 1
    assert retry_messages == []
    assert dead_messages == [(resume_id, 1, 1)]
    async with worker_context() as db:
        failed = await db.get(Resume, resume_id)
        assert failed is not None and failed.status == "failed"
        assert failed.error_message == "AI 简历解析超时，请稍后重试"
        assert failed.parsed_json is None
        assert failed.profile_patch_json is None


@pytest.mark.anyio
async def test_connection_error_retries_once_then_fails(worker_context, monkeypatch) -> None:
    resume_id = await seed_resume(worker_context)
    parse_calls = 0
    retry_messages: list[tuple[int, int, int]] = []
    dead_messages: list[tuple[int, int, int]] = []

    class APIConnectionError(Exception):
        pass

    async def connection_failure(_raw_text: str) -> ResumeParseOutput:
        nonlocal parse_calls
        parse_calls += 1
        raise APIConnectionError("private connection details")

    async def fake_retry(resume_id: int, user_id: int, attempt: int) -> None:
        retry_messages.append((resume_id, user_id, attempt))

    async def fake_dead(resume_id: int, user_id: int, attempt: int) -> None:
        dead_messages.append((resume_id, user_id, attempt))

    monkeypatch.setattr(resume_worker, "parse_resume_text", connection_failure)
    monkeypatch.setattr(resume_worker, "publish_resume_parse_retry", fake_retry)
    monkeypatch.setattr(resume_worker, "publish_resume_parse_dead_letter", fake_dead)

    first = await resume_worker.handle_resume_task(
        resume_id=resume_id,
        user_id=1,
        attempt=1,
        session_factory=worker_context,
    )
    second = await resume_worker.handle_resume_task(
        resume_id=resume_id,
        user_id=1,
        attempt=2,
        session_factory=worker_context,
    )

    assert first == resume_worker.ResumeTaskResult.RETRY
    assert second == resume_worker.ResumeTaskResult.FAILED
    assert parse_calls == 2
    assert retry_messages == [(resume_id, 1, 2)]
    assert dead_messages == [(resume_id, 1, 2)]
    async with worker_context() as db:
        failed = await db.get(Resume, resume_id)
        assert failed is not None and failed.status == "failed"
        assert failed.error_message == "AI 简历解析服务暂时不可用，请稍后重试"


@pytest.mark.anyio
async def test_duplicate_delivery_of_parsed_resume_does_not_call_model(worker_context, monkeypatch) -> None:
    output = parse_output()
    resume_id = await seed_resume(
        worker_context,
        status="parsed",
        parsed_json=json.dumps(output.parsed.model_dump(mode="json"), ensure_ascii=False),
        profile_patch_json=json.dumps(output.profile_patch.model_dump(mode="json"), ensure_ascii=False),
    )
    calls = 0

    async def should_not_parse(_raw_text: str) -> ResumeParseOutput:
        nonlocal calls
        calls += 1
        return output

    monkeypatch.setattr(resume_worker, "parse_resume_text", should_not_parse)
    async with worker_context() as db:
        result = await resume_worker.process_resume_task(
            db,
            resume_id=resume_id,
            user_id=1,
            attempt=2,
        )

    assert result == resume_worker.ResumeTaskResult.IGNORED
    assert calls == 0


@pytest.mark.anyio
async def test_worker_does_not_process_another_users_resume(worker_context, monkeypatch) -> None:
    resume_id = await seed_resume(worker_context, user_id=1)
    calls = 0

    async def should_not_parse(_raw_text: str) -> ResumeParseOutput:
        nonlocal calls
        calls += 1
        return parse_output()

    monkeypatch.setattr(resume_worker, "parse_resume_text", should_not_parse)
    async with worker_context() as db:
        result = await resume_worker.process_resume_task(
            db,
            resume_id=resume_id,
            user_id=2,
            attempt=1,
        )

    assert result == resume_worker.ResumeTaskResult.IGNORED
    assert calls == 0
    async with worker_context() as db:
        resume = await db.get(Resume, resume_id)
        assert resume is not None and resume.status == "pending"


@pytest.mark.anyio
async def test_retry_publish_failure_marks_resume_failed(worker_context, monkeypatch) -> None:
    resume_id = await seed_resume(
        worker_context,
        parsed_json='{"stale": true}',
        profile_patch_json='{"skills": "stale"}',
    )

    class APIConnectionError(Exception):
        pass

    async def connection_failure(_raw_text: str) -> ResumeParseOutput:
        raise APIConnectionError("private connection details")

    async def retry_publish_failure(_resume_id: int, _user_id: int, _attempt: int) -> None:
        raise ConnectionError("RabbitMQ unavailable")

    monkeypatch.setattr(resume_worker, "parse_resume_text", connection_failure)
    monkeypatch.setattr(resume_worker, "publish_resume_parse_retry", retry_publish_failure)

    with pytest.raises(ConnectionError, match="RabbitMQ unavailable"):
        await resume_worker.handle_resume_task(
            resume_id=resume_id,
            user_id=1,
            attempt=1,
            session_factory=worker_context,
        )

    async with worker_context() as db:
        failed = await db.get(Resume, resume_id)
        assert failed is not None and failed.status == "failed"
        assert failed.error_message == resume_worker.RETRY_PUBLISH_ERROR
        assert failed.parsed_json is None
        assert failed.profile_patch_json is None
