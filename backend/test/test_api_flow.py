import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.agents.nodes.answer_evaluator as answer_evaluator
import app.agents.nodes.answer_pipeline as answer_pipeline
import app.agents.nodes.followup_decider as followup_decider
import app.agents.nodes.interview_planner as interview_planner
import app.agents.nodes.question_generator as question_generator
import app.agents.nodes.report_generator as report_generator
import app.api.interviews as interviews_api
import app.models  # noqa: F401
from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app
from app.models.interview import InterviewMemory
from app.services import interview_memory_service


class AsyncKnowledgeContext:
    def __init__(self, value: str) -> None:
        self.value = value
        self.calls = 0

    async def __call__(self, query: str, limit: int = 5, **kwargs) -> str:
        self.calls += 1
        return self.value


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def ainvoke(self, messages):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return SimpleNamespace(content=self.responses[index])
    async def astream(self, messages):
        index = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        content = self.responses[index]
        midpoint = max(1, len(content) // 2)
        for chunk in (content[:midpoint], content[midpoint:]):
            yield SimpleNamespace(content=chunk)


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_session_factory():
    source_url = make_url(settings.database_url)
    test_database = f"ai_interview_test_{uuid4().hex[:12]}"
    admin_url = source_url.set(database=None)
    test_url = source_url.set(database=test_database)

    admin_engine = create_async_engine(admin_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f"CREATE DATABASE `{test_database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    await admin_engine.dispose()

    test_engine = create_async_engine(test_url)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    try:
        yield session_factory
    finally:
        await test_engine.dispose()
        cleanup_engine = create_async_engine(admin_url)
        async with cleanup_engine.begin() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS `{test_database}`"))
        await cleanup_engine.dispose()


@pytest.fixture
async def client(test_session_factory, monkeypatch):
    async def override_get_db():
        async with test_session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db

    monkeypatch.setattr(interview_planner, "llm", FakeLLM([
        '{"question":"请介绍一个你做过的后端项目。","plan":[{"dimension":"项目经验","question_count":2,"weight":0.25,"focus":"考察项目职责和结果"},{"dimension":"专业基础","question_count":2,"weight":0.25,"focus":"考察 FastAPI 和数据库基础"},{"dimension":"系统设计","question_count":2,"weight":0.25,"focus":"考察接口、安全和扩展性"},{"dimension":"问题排查与协作","question_count":2,"weight":0.25,"focus":"考察排查和沟通"}]}'
    ]))
    monkeypatch.setattr(question_generator, "llm", FakeLLM([
        '{"question":"你如何处理接口鉴权和 token 过期？","dimension":"登录鉴权","reason":"继续考察后端能力"}',
    ]))
    monkeypatch.setattr(answer_evaluator, "llm", FakeLLM([
        '{"score":82,"sub_scores":{"专业准确性":82,"表达清晰度":80,"项目真实性":85,"岗位匹配度":81},"reason":"回答包含关键实现。","weaknesses":["缺少指标"],"suggestions":["补充结果数据"]}'
    ]))
    monkeypatch.setattr(followup_decider, "llm", FakeLLM([
        '{"needs_followup":false,"reason":"回答基本完整，可以进入下一题。","followup_question":""}'
    ]))
    monkeypatch.setattr(answer_pipeline, "llm", FakeLLM([
        '{"needs_followup":false,"decision_reason":"回答基本完整，可以进入下一题。","question":"你如何处理接口鉴权和 token 过期？","score":82,"sub_scores":{"专业准确性":82,"表达清晰度":80,"项目真实性":85,"岗位匹配度":81},"reason":"回答包含关键实现。","weaknesses":["缺少指标"],"suggestions":["补充结果数据"]}'
    ]))
    monkeypatch.setattr(report_generator, "llm", FakeLLM([
        '{"total_score":82,"summary":"整体表现良好。","strengths":["项目表达清楚"],"weaknesses":["指标不足"],"suggestions":["补充量化结果"],"learning_path":["复习鉴权安全"],"sample_answer":"可以按背景、方案、结果组织回答。"}'
    ]))
    monkeypatch.setattr(interview_planner, "format_knowledge_context", AsyncKnowledgeContext("测试计划知识库"))
    monkeypatch.setattr(question_generator, "format_knowledge_context", AsyncKnowledgeContext("测试知识库上下文"))
    monkeypatch.setattr(answer_evaluator, "format_knowledge_context", AsyncKnowledgeContext("测试评分标准"))
    monkeypatch.setattr(answer_pipeline, "format_knowledge_context", AsyncKnowledgeContext("测试合并检索上下文"))
    monkeypatch.setattr(report_generator, "format_knowledge_context", AsyncKnowledgeContext("测试报告知识库"))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client

    app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_register_profile_interview_answer_report_flow(client: AsyncClient, test_session_factory) -> None:
    headers = await register_and_fill_profile(client)

    create_response = await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python 后端工程师", "difficulty": "medium"},
    )
    assert create_response.status_code == 201
    interview = create_response.json()
    assert interview["status"] == "preparing"
    assert interview["current_question_index"] == 1
    assert len(interview["messages"]) == 0

    start_response = await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)
    assert start_response.status_code == 200
    interview = start_response.json()
    assert interview["status"] == "active"
    assert len(interview["messages"]) == 1
    assert interview["messages"][0]["role"] == "assistant"

    async with test_session_factory() as db:
        stored_session = await db.scalar(text("SELECT interview_plan_json FROM interview_sessions WHERE id = :id"), {"id": interview["id"]})
    assert stored_session and "项目经验" in stored_session

    answer_response = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "我负责登录鉴权模块，使用 bcrypt 哈希密码，并用 JWT 做接口鉴权。", "request_id": str(uuid4())},
    )
    assert answer_response.status_code == 200
    answered = answer_response.json()
    assert answered["current_question_index"] == 2
    assert len(answered["messages"]) == 3

    scores_response = await client.get(f"/api/interviews/{interview['id']}/scores", headers=headers)
    assert scores_response.status_code == 200
    scores = scores_response.json()
    assert len(scores) == 1
    assert scores[0]["score"] == 82
    assert scores[0]["dimension"] == "项目经验"

    report_response = await client.get(f"/api/interviews/{interview['id']}/report", headers=headers)
    assert report_response.status_code == 200
    report = report_response.json()
    assert report["total_score"] == 82
    assert report["summary"] == "整体表现良好。"
    assert report["dimension_scores"]
    assert report["dimension_scores"][0]["dimension"] == "项目经验"
    assert report["dimension_scores"][0]["score"] == 82
    assert report["dimension_scores"][0]["question_indexes"] == [1]


@pytest.mark.anyio
async def test_concurrent_start_and_stream_generate_only_one_first_question(
    client: AsyncClient,
    monkeypatch,
) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    entered = asyncio.Event()
    release = asyncio.Event()
    response_json = (
        '{"question":"请介绍一个并发项目。","plan":['
        '{"dimension":"项目经验","question_count":2,"weight":0.25,"focus":"项目"},'
        '{"dimension":"专业基础","question_count":2,"weight":0.25,"focus":"基础"},'
        '{"dimension":"系统设计","question_count":2,"weight":0.25,"focus":"设计"},'
        '{"dimension":"问题排查与协作","question_count":2,"weight":0.25,"focus":"协作"}]}'
    )

    class BlockingPlannerLLM:
        calls = 0

        async def ainvoke(self, messages):
            self.calls += 1
            entered.set()
            await release.wait()
            return SimpleNamespace(content=response_json)

        async def astream(self, messages):
            self.calls += 1
            entered.set()
            await release.wait()
            yield SimpleNamespace(content=response_json)

    blocking_llm = BlockingPlannerLLM()
    monkeypatch.setattr(interview_planner, "llm", blocking_llm)

    first = asyncio.create_task(
        client.post(f"/api/interviews/{interview['id']}/start", headers=headers)
    )
    await asyncio.wait_for(entered.wait(), timeout=2)
    competing = await client.post(
        f"/api/interviews/{interview['id']}/start/stream",
        headers={**headers, "Accept": "text/event-stream"},
    )
    release.set()
    started = await first

    assert competing.status_code == 409
    assert started.status_code == 200
    assert blocking_llm.calls == 1
    stored = (await client.get(f"/api/interviews/{interview['id']}", headers=headers)).json()
    assert len(stored["messages"]) == 1
    assert stored["messages"][0]["content"] == "请介绍一个并发项目。"


@pytest.mark.anyio
async def test_start_that_loses_lease_cannot_commit_first_question(
    client: AsyncClient,
    monkeypatch,
) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()

    class LostHeartbeat:
        lost = True

        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        async def stop(self):
            pass

    monkeypatch.setattr(interviews_api, "SessionLeaseHeartbeat", LostHeartbeat)

    response = await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    assert response.status_code == 409
    stored = (await client.get(f"/api/interviews/{interview['id']}", headers=headers)).json()
    assert stored["status"] == "preparing"
    assert stored["messages"] == []

@pytest.mark.anyio
async def test_failed_start_releases_session_lease(
    client: AsyncClient,
    test_session_factory,
    monkeypatch,
) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()

    async def fail_start(state):
        raise RuntimeError("forced start failure")

    monkeypatch.setattr(interviews_api.interview_graph, "ainvoke", fail_start)
    with pytest.raises(RuntimeError, match="forced start failure"):
        await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    async with test_session_factory() as db:
        row = (
            await db.execute(
                text(
                    "SELECT processing_request_id, processing_started_at "
                    "FROM interview_sessions WHERE id = :id"
                ),
                {"id": interview["id"]},
            )
        ).one()
    assert row.processing_request_id is None
    assert row.processing_started_at is None

@pytest.mark.anyio
async def test_start_stream_emits_first_question_deltas(client: AsyncClient) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()

    response = await client.post(
        f"/api/interviews/{interview['id']}/start/stream",
        headers={**headers, "Accept": "text/event-stream"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = [frame for frame in response.text.split("\n\n") if frame and not frame.startswith(":")]
    events = [frame.splitlines()[0].removeprefix("event: ") for frame in frames]
    assert events[0] == "status"
    assert "delta" in events
    assert events[-1] == "complete"
    assert events.index("text_done") > max(index for index, event in enumerate(events) if event == "delta")
    assert events.index("text_done") < events.index("complete")

    delta_text = "".join(
        __import__("json").loads(frame.split("data: ", 1)[1])["content"]
        for frame in frames
        if frame.startswith("event: delta")
    )
    completed = __import__("json").loads(frames[-1].split("data: ", 1)[1])
    assert delta_text == completed["messages"][-1]["content"]
    assert completed["status"] == "active"
    assert interview_planner.format_knowledge_context.calls == 1
    assert interview_planner.llm.calls == 1
    assert question_generator.llm.calls == 0

@pytest.mark.anyio
async def test_answer_stream_emits_deltas_before_complete(client: AsyncClient) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    response = await client.post(
        f"/api/interviews/{interview['id']}/answer/stream",
        headers={**headers, "Accept": "text/event-stream"},
        json={"answer": "I used JWT and bcrypt.", "request_id": str(uuid4())},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream"), response.text
    frames = [frame for frame in response.text.split("\n\n") if frame and not frame.startswith(":")]
    events = [frame.splitlines()[0].removeprefix("event: ") for frame in frames]
    assert events[0] == "status"
    assert "delta" in events
    assert events[-1] == "complete"
    assert events.index("text_done") > max(index for index, event in enumerate(events) if event == "delta")
    assert events.index("text_done") < events.index("complete")

    delta_text = "".join(
        __import__("json").loads(frame.split("data: ", 1)[1])["content"]
        for frame in frames
        if frame.startswith("event: delta")
    )
    completed = __import__("json").loads(frames[-1].split("data: ", 1)[1])
    assert delta_text == completed["messages"][-1]["content"]
    assert completed["messages"][-1]["role"] == "assistant"

@pytest.mark.anyio
async def test_duplicate_answer_request_is_idempotent(client: AsyncClient) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    request_id = str(uuid4())
    payload = {"answer": "I used JWT and bcrypt for authentication.", "request_id": request_id}
    first = await client.post(f"/api/interviews/{interview['id']}/answer", headers=headers, json=payload)
    second = await client.post(f"/api/interviews/{interview['id']}/answer", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    user_messages = [message for message in second.json()["messages"] if message["role"] == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["request_id"] == request_id

    scores = (await client.get(f"/api/interviews/{interview['id']}/scores", headers=headers)).json()
    assert len(scores) == 1


@pytest.mark.anyio
async def test_answer_returns_conflict_when_session_lease_is_occupied(client: AsyncClient, monkeypatch) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    async def reject_lease(*args, **kwargs):
        return False

    monkeypatch.setattr(interviews_api, "acquire_answer_lease", reject_lease)
    response = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "Concurrent answer", "request_id": str(uuid4())},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Another answer is being processed for this interview"

@pytest.mark.anyio
async def test_answer_can_create_followup_without_advancing_question(client: AsyncClient, monkeypatch) -> None:
    headers = await register_and_fill_profile(client)
    monkeypatch.setattr(followup_decider, "llm", FakeLLM([
        '{"needs_followup":true,"reason":"回答缺少实现细节，需要继续追问。","followup_question":"你能具体说说 token 过期后如何处理吗？"}'
    ]))
    monkeypatch.setattr(answer_pipeline, "llm", FakeLLM([
        '{"needs_followup":true,"decision_reason":"回答缺少实现细节，需要继续追问。","question":"你能具体说说 token 过期后如何处理吗？","score":70,"sub_scores":{"专业准确性":70,"表达清晰度":65,"项目真实性":60,"岗位匹配度":70},"reason":"回答过于简略。","weaknesses":["缺少细节"],"suggestions":["补充实现细节"]}'
    ]))

    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python 后端工程师", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    answer_response = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "我用 JWT 做登录。", "request_id": str(uuid4())},
    )

    assert answer_response.status_code == 200
    data = answer_response.json()
    assert data["current_question_index"] == 1
    assert len(data["messages"]) == 3
    assert data["messages"][-1]["role"] == "assistant"
    assert data["messages"][-1]["is_followup"] == 1
    assert data["messages"][-1]["followup_index"] == 1


@pytest.mark.anyio
async def test_finish_interview_marks_session_finished(client: AsyncClient) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python 后端工程师", "difficulty": "medium"},
    )).json()

    finish_response = await client.post(f"/api/interviews/{interview['id']}/finish", headers=headers)

    assert finish_response.status_code == 200
    data = finish_response.json()
    assert data["status"] == "finished"
    assert data["messages"][-1]["role"] == "assistant"
    assert "已完成" in data["messages"][-1]["content"]


@pytest.mark.anyio
async def test_answer_compacts_medium_memory_when_context_exceeds_threshold(
    client: AsyncClient,
    test_session_factory,
    monkeypatch,
) -> None:
    headers = await register_and_fill_profile(client)
    monkeypatch.setattr(interview_memory_service, "llm", FakeLLM([
        '{"summary":"已压缩：候选人介绍了登录鉴权项目。","covered_topics":["登录鉴权"],"strengths":["了解 JWT"],"weaknesses":["缺少指标"],"next_focus":["追问安全细节"]}'
    ]))

    async def compact_with_low_threshold(db, session):
        await interview_memory_service.maybe_compact_medium_term_memory(
            db,
            session,
            trigger_chars=20,
            keep_recent_messages=1,
            min_compact_chars=10,
        )

    monkeypatch.setattr(interviews_api, "maybe_compact_medium_term_memory", compact_with_low_threshold)

    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python 后端工程师", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)

    first_response = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "我负责登录鉴权模块，使用 bcrypt 哈希密码，并用 JWT 做接口鉴权，包含 token 过期处理。", "request_id": str(uuid4())},
    )
    assert first_response.status_code == 200

    second_response = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "我会校验 Bearer Token，解析用户 ID，并处理过期后的重新登录。", "request_id": str(uuid4())},
    )
    assert second_response.status_code == 200

    async with test_session_factory() as db:
        memory = await db.scalar(
            select(InterviewMemory).where(
                InterviewMemory.session_id == interview["id"],
                InterviewMemory.memory_type == "session_summary",
            )
        )

    assert memory is not None
    assert memory.summary == "已压缩：候选人介绍了登录鉴权项目。"
    assert "compressed_until_message_id" in memory.metadata_json


def _stream_frames(response) -> tuple[list[str], list[str], str, dict]:
    frames = [frame for frame in response.text.split("\n\n") if frame and not frame.startswith(":")]
    events = [frame.splitlines()[0].removeprefix("event: ") for frame in frames]
    delta_text = "".join(
        __import__("json").loads(frame.split("data: ", 1)[1])["content"]
        for frame in frames
        if frame.startswith("event: delta")
    )
    text_done_frame = next(frame for frame in frames if frame.startswith("event: text_done"))
    text_done_content = __import__("json").loads(text_done_frame.split("data: ", 1)[1])["content"]
    complete = __import__("json").loads(frames[-1].split("data: ", 1)[1])
    return frames, events, delta_text, {"text_done": text_done_content, "complete": complete}


@pytest.mark.anyio
async def test_invalid_json_stream_publishes_only_committed_fallback(client: AsyncClient, monkeypatch) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)
    monkeypatch.setattr(answer_pipeline, "llm", FakeLLM(["not valid json"]))

    response = await client.post(
        f"/api/interviews/{interview['id']}/answer/stream",
        headers={**headers, "Accept": "text/event-stream"},
        json={"answer": "I used JWT.", "request_id": str(uuid4())},
    )

    assert response.status_code == 200
    _frames, events, delta_text, payloads = _stream_frames(response)
    complete = payloads["complete"]
    committed_text = complete["messages"][-1]["content"]
    assert events[-3:] == ["delta", "text_done", "complete"] or events[-2:] == ["text_done", "complete"]
    assert delta_text == payloads["text_done"] == committed_text
    assert "not valid json" not in delta_text
    stored = (await client.get(f"/api/interviews/{interview['id']}", headers=headers)).json()
    assert stored["messages"][-1]["content"] == committed_text


@pytest.mark.anyio
async def test_duplicate_question_stream_publishes_replacement_saved_to_database(client: AsyncClient, monkeypatch) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    started = (await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)).json()
    repeated = started["messages"][-1]["content"]
    monkeypatch.setattr(answer_pipeline, "llm", FakeLLM([
        '{"needs_followup":false,"decision_reason":"continue","question":"'
        + repeated
        + '","score":80,"sub_scores":{"专业准确性":80,"表达清晰度":80,"项目真实性":80,"岗位匹配度":80},'
        '"reason":"ok","weaknesses":[],"suggestions":[]}'
    ]))

    response = await client.post(
        f"/api/interviews/{interview['id']}/answer/stream",
        headers={**headers, "Accept": "text/event-stream"},
        json={"answer": "A complete answer.", "request_id": str(uuid4())},
    )

    assert response.status_code == 200
    _frames, _events, delta_text, payloads = _stream_frames(response)
    committed_text = payloads["complete"]["messages"][-1]["content"]
    assert committed_text != repeated
    assert delta_text == payloads["text_done"] == committed_text
    stored = (await client.get(f"/api/interviews/{interview['id']}", headers=headers)).json()
    assert stored["messages"][-1]["content"] == committed_text


@pytest.mark.anyio
async def test_followup_limit_stream_discards_model_followup_and_saves_next_main_question(
    client: AsyncClient,
    monkeypatch,
) -> None:
    headers = await register_and_fill_profile(client)
    interview = (await client.post(
        "/api/interviews",
        headers=headers,
        json={"target_position": "Python backend engineer", "difficulty": "medium"},
    )).json()
    await client.post(f"/api/interviews/{interview['id']}/start", headers=headers)
    responses = []
    for index in range(1, 4):
        responses.append(
            '{"needs_followup":true,"decision_reason":"need detail","question":"followup '
            + str(index)
            + '?","score":70,"sub_scores":{"专业准确性":70,"表达清晰度":70,"项目真实性":70,"岗位匹配度":70},'
            '"reason":"brief","weaknesses":[],"suggestions":[]}'
        )
    monkeypatch.setattr(answer_pipeline, "llm", FakeLLM(responses))

    first = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "first", "request_id": str(uuid4())},
    )
    second = await client.post(
        f"/api/interviews/{interview['id']}/answer",
        headers=headers,
        json={"answer": "second", "request_id": str(uuid4())},
    )
    assert first.status_code == second.status_code == 200
    assert second.json()["messages"][-1]["followup_index"] == 2

    response = await client.post(
        f"/api/interviews/{interview['id']}/answer/stream",
        headers={**headers, "Accept": "text/event-stream"},
        json={"answer": "third", "request_id": str(uuid4())},
    )

    assert response.status_code == 200
    _frames, _events, delta_text, payloads = _stream_frames(response)
    complete = payloads["complete"]
    committed = complete["messages"][-1]
    assert complete["current_question_index"] == 2
    assert committed["is_followup"] == 0
    assert committed["content"] != "followup 3?"
    assert delta_text == payloads["text_done"] == committed["content"]
    stored = (await client.get(f"/api/interviews/{interview['id']}", headers=headers)).json()
    assert stored["messages"][-1]["content"] == committed["content"]

async def register_and_fill_profile(client: AsyncClient) -> dict[str, str]:
    email = f"flow_{uuid4().hex}@example.com"
    register_response = await client.post(
        "/api/auth/register",
        json={"email": email, "username": "flow-user", "password": "Password123"},
    )
    assert register_response.status_code == 201
    token = register_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    profile_response = await client.put(
        "/api/profile/me",
        headers=headers,
        json={
            "age": 24,
            "education": "本科",
            "major": "计算机科学与技术",
            "experience_years": 1,
            "target_position": "Python 后端工程师",
            "target_city": "上海",
            "expected_salary": "15k-20k",
            "skills": "Python, FastAPI, MySQL",
            "projects": "AI 模拟面试系统",
            "self_evaluation": "学习能力强",
        },
    )
    assert profile_response.status_code == 200
    return headers
