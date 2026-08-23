import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from app.agents.graph import interview_graph
from app.agents.nodes.interview_planner import get_interview_question_count, get_plan_item_for_question
from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.interview import InterviewMessage, InterviewSession
from app.models.job_description import JobDescription
from app.models.profile import UserProfile
from app.models.report import InterviewReport
from app.models.resume import Resume
from app.models.user import User
from app.schemas.interview import InterviewAnswer, InterviewCreate, InterviewListItem, InterviewSessionRead, InterviewWarmup
from app.schemas.llm_outputs import VisibleQuestionOutput
from app.schemas.question_review import QuestionReviewRead
from app.schemas.report import InterviewReportRead
from app.schemas.score import InterviewScoreRead
from app.services.interview_answer_service import (
    SessionLeaseHeartbeat,
    SessionLeaseLostError,
    acquire_answer_lease,
    find_completed_answer_request,
    release_answer_lease,
)
from app.services.interview_state_service import build_state_from_session
from app.services.job_description_service import build_job_description_snapshot
from app.services.resume_service import build_resume_snapshot
from app.services.interview_memory_service import maybe_compact_medium_term_memory
from app.services.report_service import get_or_create_report
from app.services.llm_stream import (
    reset_stream_delta_callback,
    reset_stream_text_done_callback,
    publish_committed_text,
    set_stream_delta_callback,
    set_stream_text_done_callback,
)
from app.services.question_review_service import ensure_question_reviews
from app.services.score_service import list_scores, save_latest_score
from app.rag.embeddings import embed_text
from app.rag.retriever import ensure_collection


router = APIRouter(prefix="/interviews", tags=["interviews"])
logger = logging.getLogger(__name__)


def _planner_knowledge_query(target_position: str) -> str:
    return f"{target_position.strip()} 面试计划 岗位能力模型 评分标准"


async def _warmup_interview_dependencies(target_position: str) -> None:
    try:
        await asyncio.gather(
            ensure_collection(),
            embed_text(_planner_knowledge_query(target_position)),
        )
    except Exception as exc:
        logger.warning("interview.warmup.failed target_position=%r error=%r", target_position, exc)


@router.post("/warmup", status_code=status.HTTP_204_NO_CONTENT)
async def warmup_interview(
    payload: InterviewWarmup,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
) -> None:
    """Schedule target-specific warmup while the user is filling the form."""
    del current_user
    background_tasks.add_task(_warmup_interview_dependencies, payload.target_position)


@router.post("", response_model=InterviewSessionRead, status_code=status.HTTP_201_CREATED)
async def create_interview(
    payload: InterviewCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewSession:
    """创建待启动会话，并冻结用户选中的简历与 JD 快照。"""
    resume = None
    if payload.resume_id is not None:
        resume = await db.scalar(
            select(Resume).where(Resume.id == payload.resume_id, Resume.user_id == current_user.id)
        )
        if resume is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resume not found")
        if resume.status != "parsed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only a parsed resume can be used for an interview",
            )

    job_description = None
    if payload.job_description_id is not None:
        job_description = await db.scalar(
            select(JobDescription).where(
                JobDescription.id == payload.job_description_id,
                JobDescription.user_id == current_user.id,
            )
        )
        if job_description is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job description not found")
        if job_description.status != "parsed":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only a parsed job description can be used for an interview",
            )

    if payload.parent_session_id is not None:
        parent_exists = await db.scalar(
            select(InterviewSession.id).where(
                InterviewSession.id == payload.parent_session_id,
                InterviewSession.user_id == current_user.id,
            )
        )
        if parent_exists is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Parent interview not found")

    if payload.source_report_id is not None:
        source_report_exists = await db.scalar(
            select(InterviewReport.id)
            .join(InterviewSession, InterviewSession.id == InterviewReport.session_id)
            .where(
                InterviewReport.id == payload.source_report_id,
                InterviewSession.user_id == current_user.id,
            )
        )
        if source_report_exists is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Source report not found")

    session = InterviewSession(
        user_id=current_user.id,
        target_position=payload.target_position,
        difficulty=payload.difficulty,
        mode=payload.mode,
        interview_type=payload.interview_type,
        status="preparing",
        resume_id=resume.id if resume else None,
        resume_snapshot_json=build_resume_snapshot(resume) if resume else None,
        job_description_id=job_description.id if job_description else None,
        job_description_snapshot_json=(
            build_job_description_snapshot(job_description) if job_description else None
        ),
        parent_session_id=payload.parent_session_id,
        source_report_id=payload.source_report_id,
        source_weakness_key=payload.source_weakness_key,
        session_purpose=payload.session_purpose,
        comparison_group_id=payload.comparison_group_id,
    )
    db.add(session)
    await db.commit()
    return await _load_session(db, current_user.id, session.id)


@router.post("/{session_id}/start", response_model=InterviewSessionRead)
async def start_interview(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewSession:
    """启动面试并持久化计划和首题；已有消息时直接返回，避免重复生成首题。"""
    session = await _load_session(db, current_user.id, session_id)
    if session.status == "finished":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview is already finished")
    if session.messages:
        if session.status == "preparing":
            session.status = "active"
            await db.commit()
        return await _load_session(db, current_user.id, session.id)

    start_request_id = str(uuid4())
    acquired = await acquire_answer_lease(
        db,
        session_id=session_id,
        user_id=current_user.id,
        request_id=start_request_id,
        allowed_statuses=("preparing",),
    )
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interview start is already being processed",
        )

    session_factory = async_sessionmaker(bind=db.bind, expire_on_commit=False)
    heartbeat = SessionLeaseHeartbeat(
        session_factory,
        session_id=session_id,
        request_id=start_request_id,
    )
    try:
        return await _process_start(
            db,
            current_user,
            session_id,
            start_request_id,
            heartbeat,
        )
    except SessionLeaseLostError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/{session_id}/start/stream")
async def start_interview_stream(
    session_id: int,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """以 SSE 启动面试，边生成首题边推送文本，完成后再统一提交会话结果。

    流式生成使用独立数据库会话，避免 FastAPI 请求依赖在响应迭代期间被提前关闭；
    客户端断开时会取消后台任务并回滚尚未提交的计划和消息。
    """
    session = await _load_session(db, current_user.id, session_id)
    if session.status == "finished":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview is already finished")
    if session.messages:
        if session.status == "preparing":
            session.status = "active"
            await db.commit()
        loaded = await _load_session(db, current_user.id, session.id)
        return _sse_response(_completed_session_stream(loaded))

    start_request_id = str(uuid4())
    acquired = await acquire_answer_lease(
        db,
        session_id=session_id,
        user_id=current_user.id,
        request_id=start_request_id,
        allowed_statuses=("preparing",),
    )
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interview start is already being processed",
        )

    stream_session_factory = async_sessionmaker(bind=db.bind, expire_on_commit=False)
    await db.close()

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        async with stream_session_factory() as stream_db:
            async def emit_delta(delta: str) -> None:
                await queue.put(_sse_event("delta", {"content": delta}))

            async def emit_text_done(content: str) -> None:
                await queue.put(_sse_event("text_done", {"content": content}))

            async def process() -> InterviewSession:
                delta_token = set_stream_delta_callback(emit_delta)
                done_token = set_stream_text_done_callback(emit_text_done)
                heartbeat = SessionLeaseHeartbeat(
                    stream_session_factory,
                    session_id=session_id,
                    request_id=start_request_id,
                )
                try:
                    committed = await _process_start(
                        stream_db,
                        current_user,
                        session_id,
                        start_request_id,
                        heartbeat,
                    )
                    await publish_committed_text(committed.messages[-1].content)
                    return committed
                finally:
                    reset_stream_text_done_callback(done_token)
                    reset_stream_delta_callback(delta_token)

            task = asyncio.create_task(process())
            yield _sse_event("status", {"phase": "retrieving"})
            try:
                while not task.done():
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    try:
                        yield await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"

                while not queue.empty():
                    yield queue.get_nowait()
                result = await task
                yield _sse_event("complete", _serialize_session(result))
            except asyncio.CancelledError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                await stream_db.rollback()
                raise
            except Exception as exc:
                await stream_db.rollback()
                logger.exception("interview.start.stream.failed session_id=%s", session_id)
                yield _sse_event("error", {"message": str(exc) or "Streaming start failed"})

    return _sse_response(
        event_stream(),
        background=BackgroundTask(
            _release_stream_lease,
            stream_session_factory,
            session_id,
            start_request_id,
        ),
    )

@router.get("", response_model=list[InterviewListItem])
async def list_interviews(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[InterviewSession]:
    result = await db.scalars(
        select(InterviewSession)
        .where(InterviewSession.user_id == current_user.id)
        .order_by(InterviewSession.updated_at.desc())
    )
    return list(result)


# 获取指定面试的完整信息。
@router.get("/{session_id}", response_model=InterviewSessionRead)
async def get_interview(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewSession:
    return await _load_session(db, current_user.id, session_id)


# 获取指定面试的各项评分。
@router.get("/{session_id}/scores", response_model=list[InterviewScoreRead])
async def get_interview_scores(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[InterviewScoreRead]:
    await _load_session(db, current_user.id, session_id)
    return await list_scores(db, session_id)


@router.get("/{session_id}/question-reviews", response_model=list[QuestionReviewRead])
async def get_interview_question_reviews(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[QuestionReviewRead]:
    """Generate the report if needed, then return its stable per-score reviews."""
    await _load_session(db, current_user.id, session_id)
    report_read = await get_or_create_report(db, session_id)
    report = await db.get(InterviewReport, report_read.id)
    if report is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    reviews = await ensure_question_reviews(db, report)
    await db.commit()
    return reviews


# 获取或生成指定面试的报告。
@router.get("/{session_id}/report", response_model=InterviewReportRead)
async def get_interview_report(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewReportRead:
    await _load_session(db, current_user.id, session_id)
    report = await get_or_create_report(db, session_id)
    await db.commit()
    return report


# 提交回答并生成追问或下一道问题。
@router.post("/{session_id}/answer", response_model=InterviewSessionRead)
async def answer_interview(
    session_id: int,
    payload: InterviewAnswer,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewSession:
    """处理非流式回答；request_id 保证重试幂等，会话租约阻止并发回答串线。"""
    request_id = str(payload.request_id)
    session = await _load_session(db, current_user.id, session_id)
    completed_request = await find_completed_answer_request(
        db,
        session_id=session_id,
        request_id=request_id,
    )
    if completed_request:
        return session
    if session.status == "preparing":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview is still preparing")
    if session.status != "active":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview is not active")

    acquired = await acquire_answer_lease(
        db,
        session_id=session_id,
        user_id=current_user.id,
        request_id=request_id,
    )
    if not acquired:
        completed_request = await find_completed_answer_request(
            db,
            session_id=session_id,
            request_id=request_id,
        )
        if completed_request:
            return await _load_session(db, current_user.id, session_id)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another answer is being processed for this interview",
        )

    try:
        session = await _load_session(db, current_user.id, session_id)
        return await _process_answer(db, session, current_user, payload.answer, request_id)
    except Exception:
        await db.rollback()
        await release_answer_lease(db, session_id=session_id, request_id=request_id)
        raise


@router.post("/{session_id}/answer/stream")
async def answer_interview_stream(
    session_id: int,
    payload: InterviewAnswer,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """处理流式回答，在同一租约内完成评分、路由、消息保存并推送下一问题。

    `delta` 只代表可展示文本，`complete` 才携带已提交的权威会话状态。异常或断连
    会回滚事务并释放租约，使客户端可复用原 request_id 安全重试。
    """
    request_id = str(payload.request_id)
    session = await _load_session(db, current_user.id, session_id)
    completed_request = await find_completed_answer_request(
        db,
        session_id=session_id,
        request_id=request_id,
    )
    if completed_request:
        return _sse_response(_completed_session_stream(session))
    if session.status == "preparing":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview is still preparing")
    if session.status != "active":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Interview is not active")

    acquired = await acquire_answer_lease(
        db,
        session_id=session_id,
        user_id=current_user.id,
        request_id=request_id,
    )
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Another answer is being processed for this interview",
        )

    stream_session_factory = async_sessionmaker(bind=db.bind, expire_on_commit=False)
    await db.close()

    async def event_stream() -> AsyncIterator[str]:
        queue: asyncio.Queue[str] = asyncio.Queue()
        async with stream_session_factory() as stream_db:
            async def emit_delta(delta: str) -> None:
                await queue.put(_sse_event("delta", {"content": delta}))

            async def emit_text_done(content: str) -> None:
                await queue.put(_sse_event("text_done", {"content": content}))

            async def process() -> InterviewSession:
                active_session = await _load_session(stream_db, current_user.id, session_id)
                delta_token = set_stream_delta_callback(emit_delta)
                done_token = set_stream_text_done_callback(emit_text_done)
                try:
                    committed = await _process_answer(
                        stream_db,
                        active_session,
                        current_user,
                        payload.answer,
                        request_id,
                    )
                    await publish_committed_text(committed.messages[-1].content)
                    return committed
                finally:
                    reset_stream_text_done_callback(done_token)
                    reset_stream_delta_callback(delta_token)

            task = asyncio.create_task(process())
            yield _sse_event("status", {"phase": "evaluating"})
            try:
                while not task.done():
                    if await request.is_disconnected():
                        task.cancel()
                        break
                    try:
                        yield await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"

                while not queue.empty():
                    yield queue.get_nowait()

                result = await task
                yield _sse_event("complete", _serialize_session(result))
            except asyncio.CancelledError:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                await stream_db.rollback()
                await release_answer_lease(stream_db, session_id=session_id, request_id=request_id)
                raise
            except Exception as exc:
                await stream_db.rollback()
                await release_answer_lease(stream_db, session_id=session_id, request_id=request_id)
                logger.exception("interview.answer.stream.failed session_id=%s", session_id)
                yield _sse_event("error", {"message": str(exc) or "Streaming answer failed"})

    return _sse_response(event_stream())


async def _process_answer(
    db: AsyncSession,
    session: InterviewSession,
    current_user: User,
    answer: str,
    request_id: str,
) -> InterviewSession:
    """执行一轮回答的核心事务：保存回答、运行面试图、落分并推进会话状态。

    追问仍归属当前主问题；只有不追问时才推进主问题序号。租约释放与本轮消息、
    评分在同一次提交中完成，保证客户端看到的完成状态与数据库一致。
    """
    existing_followups = sum(
        1
        for message in session.messages
        if message.role == "assistant"
        and message.question_index == session.current_question_index
        and bool(message.is_followup)
    )

    current_plan_item = get_plan_item_for_question(
        parse_interview_plan(session.interview_plan_json),
        session.current_question_index,
    )
    current_dimension = next(
        (
            message.dimension
            for message in reversed(session.messages)
            if message.role == "assistant"
            and message.question_index == session.current_question_index
            and message.dimension
        ),
        current_plan_item["dimension"],
    )

    db.add(
        InterviewMessage(
            session_id=session.id,
            role="user",
            content=answer,
            request_id=request_id,
            question_index=session.current_question_index,
            dimension=current_dimension,
            is_followup=1 if existing_followups > 0 else 0,
            followup_index=existing_followups,
        )
    )
    await db.flush()

    session = await _load_session(db, current_user.id, session.id)
    # 历史上下文超阈值时，先把较早消息压缩进中期记忆，再构造图状态。
    await maybe_compact_medium_term_memory(db, session)
    session = await _load_session(db, current_user.id, session.id)
    profile = await _get_profile(db, current_user.id)
    state = build_state_from_session(session, profile)
    state["action"] = "answer"
    result = await interview_graph.ainvoke(state)

    latest_score = await save_latest_score(db, session.id, result["scores"])

    is_followup = 0
    followup_index = 0
    message_question_index = session.current_question_index

    if result["followup_decision"] and result["followup_decision"]["needs_followup"]:
        is_followup = 1
        followup_index = result["follow_up_count"]
    else:
        if result["status"] == "finished":
            session.status = "finished"
        else:
            session.current_question_index += 1
            message_question_index = session.current_question_index

    question = _validated_visible_question(result["current_question"])
    if _is_duplicate_question(question, session.messages):
        total_question_count = get_interview_question_count(parse_interview_plan(session.interview_plan_json))
        if is_followup and session.current_question_index >= total_question_count:
            session.status = "finished"
            question = _validated_visible_question("本次模拟面试已完成。可以查看评分与复盘报告。")
        else:
            if is_followup:
                session.current_question_index += 1
                message_question_index = session.current_question_index
            next_plan_item = get_plan_item_for_question(
                parse_interview_plan(session.interview_plan_json),
                message_question_index,
            )
            question = _validated_visible_question(
                f"请围绕{next_plan_item['dimension']}，结合一个尚未讨论的具体经历说明你的做法、取舍和结果。"
            )
        is_followup = 0
        followup_index = 0

    db.add(
        InterviewMessage(
            session_id=session.id,
            role="assistant",
            content=question,
            question_index=message_question_index,
            dimension=result.get("current_dimension") or current_dimension,
            is_followup=is_followup,
            followup_index=followup_index,
        )
    )
    await release_answer_lease(
        db,
        session_id=session.id,
        request_id=request_id,
        commit=False,
        require_held=True,
    )
    await db.commit()
    return await _load_session(db, current_user.id, session.id)


@router.post("/{session_id}/finish", response_model=InterviewSessionRead)
async def finish_interview(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewSession:
    """主动结束面试，并复用回答租约与提交回答互斥，避免终止和评分并发写入。"""
    session = await _load_session(db, current_user.id, session_id)
    if session.status == "finished":
        return session

    finish_request_id = str(uuid4())
    acquired = await acquire_answer_lease(
        db,
        session_id=session_id,
        user_id=current_user.id,
        request_id=finish_request_id,
        allowed_statuses=("preparing", "active"),
    )
    if not acquired:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An answer is being processed; try finishing again shortly",
        )

    try:
        session = await _load_session(db, current_user.id, session_id)
        profile = await _get_profile(db, current_user.id)
        state = build_state_from_session(session, profile)
        state["action"] = "finish"
        result = await interview_graph.ainvoke(state)

        session.status = result["status"]
        question = _validated_visible_question(result["current_question"])
        db.add(
            InterviewMessage(
                session_id=session.id,
                role="assistant",
                content=question,
                question_index=session.current_question_index,
                dimension=result.get("current_dimension"),
                is_followup=0,
                followup_index=0,
            )
        )
        await release_answer_lease(
            db,
            session_id=session.id,
            request_id=finish_request_id,
            commit=False,
            require_held=True,
        )
        await db.commit()
        return await _load_session(db, current_user.id, session.id)
    except Exception:
        await db.rollback()
        await release_answer_lease(db, session_id=session_id, request_id=finish_request_id)
        raise

async def _process_start(
    db: AsyncSession,
    current_user: User,
    session_id: int,
    request_id: str,
    heartbeat: SessionLeaseHeartbeat,
) -> InterviewSession:
    """Generate and commit the first question behind a renewable, fenced session lease."""
    heartbeat.start()
    try:
        session = await _load_session(db, current_user.id, session_id)
        profile = await _get_profile(db, current_user.id)
        state = build_state_from_session(session, profile, messages=[])
        state["action"] = "start"
        result = await interview_graph.ainvoke(state)
        session.interview_plan_json = json.dumps(result.get("interview_plan") or [], ensure_ascii=False)
        session.status = "active"
        question = _validated_visible_question(result["current_question"])
        db.add(
            InterviewMessage(
                session_id=session.id,
                role="assistant",
                content=question,
                question_index=session.current_question_index,
                dimension=result.get("current_dimension"),
                is_followup=0,
                followup_index=0,
            )
        )

        await heartbeat.stop()
        if heartbeat.lost:
            raise SessionLeaseLostError("Interview session lease ownership was lost")
        await release_answer_lease(
            db,
            session_id=session_id,
            request_id=request_id,
            commit=False,
            require_held=True,
        )
        await db.commit()
        return await _load_session(db, current_user.id, session_id)
    except BaseException:
        await heartbeat.stop()
        await db.rollback()
        await release_answer_lease(db, session_id=session_id, request_id=request_id)
        raise


async def _load_session(db: AsyncSession, user_id: int, session_id: int) -> InterviewSession:
    """加载当前用户拥有的会话及消息、记忆，并挂载响应所需的当前计划字段。"""
    session = await db.scalar(
        select(InterviewSession)
        .options(selectinload(InterviewSession.messages), selectinload(InterviewSession.memories))
        .where(InterviewSession.id == session_id, InterviewSession.user_id == user_id)
        .execution_options(populate_existing=True)
    )
    if not session:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Interview not found")
    attach_current_plan_fields(session)
    return session


# 查询用户的求职画像。
async def _get_profile(db: AsyncSession, user_id: int) -> UserProfile | None:
    return await db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))


# 将面试计划 JSON 解析为列表。
def parse_interview_plan(plan_json: str | None) -> list[dict]:
    try:
        plan = json.loads(plan_json or "[]")
    except json.JSONDecodeError:
        return []
    return plan if isinstance(plan, list) else []


# 补充当前问题对应的计划字段。
def attach_current_plan_fields(session: InterviewSession) -> None:
    """根据持久化计划计算只用于接口响应的当前维度、重点和总题数。"""
    try:
        plan = json.loads(session.interview_plan_json or "[]")
    except json.JSONDecodeError:
        plan = []
    plan_item = get_plan_item_for_question(plan if isinstance(plan, list) else [], session.current_question_index)
    setattr(session, "current_dimension", plan_item.get("dimension"))
    setattr(session, "current_plan_focus", plan_item.get("focus"))
    setattr(
        session,
        "total_question_count",
        get_interview_question_count(plan if isinstance(plan, list) else []),
    )


def _validated_visible_question(value: object) -> str:
    return VisibleQuestionOutput.model_validate({"question": value}).question


def _is_duplicate_question(question: str, messages: list[InterviewMessage]) -> bool:
    normalized = "".join(question.split()).casefold()
    return any(
        message.role == "assistant" and "".join(message.content.split()).casefold() == normalized
        for message in messages
    )


def _sse_response(
    events: AsyncIterator[str],
    *,
    background: BackgroundTask | None = None,
) -> StreamingResponse:
    """创建禁用代理缓冲的 SSE 响应，确保模型增量能及时到达浏览器。"""
    return StreamingResponse(
        events,
        media_type="text/event-stream",
        background=background,
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _release_stream_lease(
    session_factory: async_sessionmaker[AsyncSession],
    session_id: int,
    request_id: str,
) -> None:
    async with session_factory() as db:
        await release_answer_lease(db, session_id=session_id, request_id=request_id)


def _sse_event(event: str, data: object) -> str:
    """按 SSE 帧格式序列化事件，JSON 保留中文并使用紧凑编码。"""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


def _serialize_session(session: InterviewSession) -> dict:
    return InterviewSessionRead.model_validate(session).model_dump(mode="json")


async def _completed_session_stream(session: InterviewSession) -> AsyncIterator[str]:
    yield _sse_event("complete", _serialize_session(session))
