import json
import re
from typing import Any
from uuid import uuid4

from fastapi import HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.interview import InterviewSession
from app.models.practice import PracticeSession
from app.models.question_review import QuestionReview
from app.models.report import InterviewReport
from app.models.score import InterviewScore
from app.schemas.practice import (
    PracticeComparisonRead,
    PracticeFromQuestionReviewCreate,
    PracticeFromReportCreate,
)
from app.services.question_review_service import ensure_question_reviews


async def create_practice_from_report(
    db: AsyncSession,
    user_id: int,
    payload: PracticeFromReportCreate,
) -> PracticeSession:
    """Create a practice source snapshot from a report weakness and its closest review."""
    report, source_session = await _get_owned_report(db, user_id, payload.report_id)
    report_weaknesses = _string_list(report.weaknesses)
    weakness_index = _find_exact_index(report_weaknesses, payload.weakness_title)
    expected_key = f"report_weakness_{weakness_index + 1}" if weakness_index >= 0 else ""
    if weakness_index < 0 or payload.weakness_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The selected weakness does not belong to this report",
        )

    reviews = await _get_report_reviews(db, report)
    if not reviews:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The report has no scored question available for practice",
        )
    review = _select_review_for_weakness(reviews, payload.weakness_title)
    snapshot = _build_source_snapshot(
        report=report,
        source_session=source_session,
        review=review,
        weakness_key=expected_key,
        weakness_title=report_weaknesses[weakness_index],
        practice_mode="report_weakness",
        report_weaknesses=report_weaknesses,
    )
    return await _create_practice(db, user_id, source_session, report, review, snapshot)


async def create_practice_from_question_review(
    db: AsyncSession,
    user_id: int,
    payload: PracticeFromQuestionReviewCreate,
) -> PracticeSession:
    """Create a practice source snapshot after validating every client-supplied source id."""
    report, source_session = await _get_owned_report(db, user_id, payload.report_id)
    await ensure_question_reviews(db, report)
    review = await db.scalar(
        select(QuestionReview).where(
            QuestionReview.id == payload.question_review_id,
            QuestionReview.report_id == report.id,
        )
    )
    if review is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Question review not found")
    if review.score_id != payload.score_id or review.question_index != payload.question_index:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Question review source fields do not match the saved review",
        )
    if payload.weakness_key != review.weakness_key:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The selected weakness key does not match the saved review",
        )
    allowed_titles = _unique_text(
        _string_list(review.weaknesses_json) + _string_list(review.deduction_reasons_json)
    )
    if allowed_titles and payload.weakness_title not in allowed_titles:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The selected weakness does not belong to this question review",
        )

    report_weaknesses = _string_list(report.weaknesses)
    snapshot = _build_source_snapshot(
        report=report,
        source_session=source_session,
        review=review,
        weakness_key=review.weakness_key,
        weakness_title=payload.weakness_title,
        practice_mode=payload.practice_mode,
        report_weaknesses=report_weaknesses,
    )
    return await _create_practice(db, user_id, source_session, report, review, snapshot)


async def list_owned_practices(db: AsyncSession, user_id: int) -> list[PracticeSession]:
    practices = list(await db.scalars(
        select(PracticeSession)
        .where(PracticeSession.user_id == user_id)
        .order_by(PracticeSession.updated_at.desc(), PracticeSession.id.desc())
    ))
    for practice in practices:
        await refresh_practice_status(db, practice)
    return practices


async def get_owned_practice(
    db: AsyncSession,
    user_id: int,
    practice_id: int,
    *,
    lock: bool = False,
) -> PracticeSession:
    statement = select(PracticeSession).where(
        PracticeSession.id == practice_id,
        PracticeSession.user_id == user_id,
    )
    if lock:
        statement = statement.with_for_update()
    practice = await db.scalar(statement)
    if practice is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Practice not found")
    return practice


async def start_owned_practice(
    db: AsyncSession,
    user_id: int,
    practice_id: int,
) -> InterviewSession:
    practice = await get_owned_practice(db, user_id, practice_id, lock=True)
    await refresh_practice_status(db, practice)
    session = await _load_owned_interview(db, user_id, practice.practice_session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Practice session is unavailable")
    if session.status != "finished":
        practice.status = "practicing"
    attach_practice_ids(session, practice.id)
    return session


async def start_owned_retest(
    db: AsyncSession,
    user_id: int,
    practice_id: int,
) -> InterviewSession:
    practice = await get_owned_practice(db, user_id, practice_id, lock=True)
    await refresh_practice_status(db, practice)
    practice_session = await db.get(InterviewSession, practice.practice_session_id)
    if practice_session is None or practice_session.status != "finished":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Finish the weakness practice before starting the retest",
        )

    if practice.retest_session_id is not None:
        existing = await _load_owned_interview(db, user_id, practice.retest_session_id)
        if existing is None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Retest session is unavailable")
        attach_practice_ids(existing, practice.id)
        return existing

    source_session = await db.get(InterviewSession, practice.source_session_id)
    if source_session is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Source interview is unavailable")
    context = _json_dict(practice.source_snapshot_json)
    retest = _new_interview_session(
        source_session=source_session,
        parent_session_id=practice_session.id,
        source_report_id=practice.source_report_id,
        weakness_key=practice.weakness_key,
        comparison_group_id=practice.comparison_group_id,
        purpose="retest",
        practice_context=context,
    )
    db.add(retest)
    await db.flush()
    practice.retest_session_id = retest.id
    practice.status = "ready_for_retest"
    loaded_retest = await _load_owned_interview(db, user_id, retest.id)
    if loaded_retest is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Retest session is unavailable")
    attach_practice_ids(loaded_retest, practice.id)
    return loaded_retest


async def get_practice_comparison(
    db: AsyncSession,
    user_id: int,
    practice_id: int,
) -> PracticeComparisonRead:
    practice = await get_owned_practice(db, user_id, practice_id, lock=True)
    await refresh_practice_status(db, practice)
    if practice.comparison_json:
        frozen = _json_dict(practice.comparison_json)
        if frozen:
            return PracticeComparisonRead.model_validate(frozen)
    return PracticeComparisonRead.model_validate(_pending_comparison(practice))


async def sync_practice_for_interview(
    db: AsyncSession,
    session: InterviewSession,
) -> None:
    """Advance a linked practice lifecycle in the same transaction as the interview state."""
    practice = await db.scalar(
        select(PracticeSession)
        .where(
            PracticeSession.user_id == session.user_id,
            or_(
                PracticeSession.practice_session_id == session.id,
                PracticeSession.retest_session_id == session.id,
            ),
        )
        .with_for_update()
    )
    if practice is None:
        return

    if practice.practice_session_id == session.id:
        if session.status == "finished" and practice.status != "completed":
            practice.status = "ready_for_retest"
        elif session.status == "active" and practice.status == "not_started":
            practice.status = "practicing"

    if practice.retest_session_id == session.id and session.status == "finished":
        await _freeze_comparison(db, practice, session)


async def refresh_practice_status(db: AsyncSession, practice: PracticeSession) -> None:
    if practice.comparison_json and practice.status == "completed":
        return
    if practice.retest_session_id is not None:
        retest = await db.get(InterviewSession, practice.retest_session_id)
        if retest is not None and retest.status == "finished":
            await _freeze_comparison(db, practice, retest)
            return
    practice_session = await db.get(InterviewSession, practice.practice_session_id)
    if practice_session is None:
        return
    if practice_session.status == "finished":
        practice.status = "ready_for_retest"
    elif practice_session.status == "active":
        practice.status = "practicing"


async def attach_practice_fields(db: AsyncSession, session: InterviewSession) -> None:
    if session.session_purpose not in {"weakness_practice", "retest"}:
        return
    practice_id = await db.scalar(
        select(PracticeSession.id).where(
            or_(
                PracticeSession.practice_session_id == session.id,
                PracticeSession.retest_session_id == session.id,
            )
        )
    )
    if practice_id is not None:
        attach_practice_ids(session, practice_id)


def attach_practice_ids(session: InterviewSession, practice_id: int) -> None:
    setattr(session, "practice_id", practice_id)
    setattr(session, "source_practice_id", practice_id)


async def _get_owned_report(
    db: AsyncSession,
    user_id: int,
    report_id: int,
) -> tuple[InterviewReport, InterviewSession]:
    result = (
        await db.execute(
            select(InterviewReport, InterviewSession)
            .join(InterviewSession, InterviewReport.session_id == InterviewSession.id)
            .where(
                InterviewReport.id == report_id,
                InterviewReport.is_final.is_(True),
                InterviewSession.user_id == user_id,
                InterviewSession.status == "finished",
                InterviewSession.session_purpose == "full_interview",
            )
        )
    ).first()
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")
    return result[0], result[1]


async def _load_owned_interview(
    db: AsyncSession,
    user_id: int,
    session_id: int | None,
) -> InterviewSession | None:
    if session_id is None:
        return None
    return await db.scalar(
        select(InterviewSession)
        .options(selectinload(InterviewSession.messages))
        .where(InterviewSession.id == session_id, InterviewSession.user_id == user_id)
        .execution_options(populate_existing=True)
    )


async def _get_report_reviews(
    db: AsyncSession,
    report: InterviewReport,
) -> list[QuestionReview]:
    await ensure_question_reviews(db, report)
    return list(await db.scalars(
        select(QuestionReview)
        .where(QuestionReview.report_id == report.id)
        .order_by(QuestionReview.question_index.asc(), QuestionReview.id.asc())
    ))


def _select_review_for_weakness(
    reviews: list[QuestionReview],
    weakness_title: str,
) -> QuestionReview:
    for review in reviews:
        candidates = _string_list(review.weaknesses_json) + _string_list(
            review.deduction_reasons_json
        )
        if any(_same_weakness(candidate, weakness_title) for candidate in candidates):
            return review
    return min(reviews, key=lambda item: (item.score, item.question_index, item.id))


async def _create_practice(
    db: AsyncSession,
    user_id: int,
    source_session: InterviewSession,
    report: InterviewReport,
    review: QuestionReview,
    snapshot: dict[str, Any],
) -> PracticeSession:
    comparison_group_id = uuid4().hex
    practice_interview = _new_interview_session(
        source_session=source_session,
        parent_session_id=source_session.id,
        source_report_id=report.id,
        weakness_key=str(snapshot["weakness_key"]),
        comparison_group_id=comparison_group_id,
        purpose="weakness_practice",
        practice_context=snapshot,
    )
    db.add(practice_interview)
    await db.flush()

    practice = PracticeSession(
        user_id=user_id,
        source_report_id=report.id,
        source_session_id=source_session.id,
        source_question_review_id=review.id,
        source_score_id=review.score_id,
        weakness_key=str(snapshot["weakness_key"]),
        weakness_title=str(snapshot["weakness_title"]),
        target_dimension=str(snapshot["target_dimension"]),
        practice_mode=str(snapshot["practice_mode"]),
        comparison_group_id=comparison_group_id,
        practice_session_id=practice_interview.id,
        status="not_started",
        before_score=int(snapshot["before"]["score"]),
        source_snapshot_json=json.dumps(snapshot, ensure_ascii=False),
    )
    db.add(practice)
    await db.flush()
    attach_practice_ids(practice_interview, practice.id)
    return practice


def _new_interview_session(
    *,
    source_session: InterviewSession,
    parent_session_id: int,
    source_report_id: int,
    weakness_key: str,
    comparison_group_id: str,
    purpose: str,
    practice_context: dict[str, Any],
) -> InterviewSession:
    return InterviewSession(
        user_id=source_session.user_id,
        target_position=source_session.target_position,
        difficulty=source_session.difficulty,
        mode="training" if purpose == "weakness_practice" else "mock",
        interview_type=source_session.interview_type,
        status="preparing",
        resume_id=source_session.resume_id,
        resume_snapshot_json=source_session.resume_snapshot_json,
        job_description_id=source_session.job_description_id,
        job_description_snapshot_json=source_session.job_description_snapshot_json,
        practice_context_json=json.dumps(practice_context, ensure_ascii=False),
        parent_session_id=parent_session_id,
        source_report_id=source_report_id,
        source_weakness_key=weakness_key,
        session_purpose=purpose,
        comparison_group_id=comparison_group_id,
    )


def _build_source_snapshot(
    *,
    report: InterviewReport,
    source_session: InterviewSession,
    review: QuestionReview,
    weakness_key: str,
    weakness_title: str,
    practice_mode: str,
    report_weaknesses: list[str],
) -> dict[str, Any]:
    review_weaknesses = _string_list(review.weaknesses_json)
    before_weaknesses = _unique_text([weakness_title] + review_weaknesses)
    answer = review.answer.strip()
    return {
        "weakness_key": weakness_key,
        "weakness_title": weakness_title,
        "target_dimension": review.dimension,
        "practice_mode": practice_mode,
        "target_position": source_session.target_position,
        "source_report_id": report.id,
        "source_report_total_score": report.total_score,
        "source_session_id": source_session.id,
        "source_question_review_id": review.id,
        "source_score_id": review.score_id,
        "source_question_index": review.question_index,
        "source_question": review.question,
        "source_answer": answer,
        "deduction_reasons": _string_list(review.deduction_reasons_json),
        "suggested_structure": _string_list(review.suggested_structure_json),
        "source_weaknesses": before_weaknesses,
        "report_weaknesses": report_weaknesses,
        "before": {
            "session_id": source_session.id,
            "score": review.score,
            "weaknesses": before_weaknesses,
            "answer": answer,
            "answer_structure": _detect_answer_structure(answer),
            "sub_scores": _int_dict(review.sub_scores_json),
        },
    }


async def _freeze_comparison(
    db: AsyncSession,
    practice: PracticeSession,
    retest_session: InterviewSession,
) -> None:
    if practice.comparison_json:
        practice.status = "completed"
        return
    scores = list(await db.scalars(
        select(InterviewScore)
        .where(InterviewScore.session_id == retest_session.id)
        .order_by(InterviewScore.question_index.asc(), InterviewScore.id.asc())
    ))
    source = _json_dict(practice.source_snapshot_json)
    before = source.get("before") if isinstance(source.get("before"), dict) else {}
    after = _score_snapshot(retest_session.id, scores)
    before_weaknesses = _string_list(before.get("weaknesses"))
    after_weaknesses = _string_list(after.get("weaknesses"))
    score_delta = int(after["score"]) - int(before.get("score") or 0)
    target_score = _target_dimension_score(scores, practice.target_dimension)
    target_improved = target_score is not None and target_score > int(before.get("score") or 0)
    resolved, remaining = _compare_weaknesses(
        before_weaknesses,
        after_weaknesses,
        target_title=practice.weakness_title,
        target_improved=target_improved,
    )
    before_sub_scores = _int_dict(before.get("sub_scores"))
    after_sub_scores = _int_dict(after.get("sub_scores"))
    sub_score_delta = {
        key: after_sub_scores[key] - before_sub_scores[key]
        for key in before_sub_scores.keys() & after_sub_scores.keys()
    }
    comparison = {
        "practice_id": practice.id,
        "status": "completed",
        "source_report_id": practice.source_report_id,
        "practice_session_id": practice.practice_session_id,
        "retest_session_id": practice.retest_session_id,
        "weakness_title": practice.weakness_title,
        "before": before,
        "after": after,
        "delta": {
            "score": score_delta,
            "resolved_weaknesses": resolved,
            "remaining_weaknesses": remaining,
            "sub_scores": sub_score_delta,
        },
        "summary": _comparison_summary(score_delta, resolved, remaining, bool(scores)),
        "next_weakness": _next_weakness(source, practice.weakness_title),
    }
    practice.after_score = int(after["score"])
    practice.comparison_json = json.dumps(comparison, ensure_ascii=False)
    practice.status = "completed"


def _pending_comparison(practice: PracticeSession) -> dict[str, Any]:
    source = _json_dict(practice.source_snapshot_json)
    before = source.get("before") if isinstance(source.get("before"), dict) else {
        "session_id": practice.source_session_id,
        "score": practice.before_score,
        "weaknesses": [practice.weakness_title],
        "answer": "",
        "answer_structure": [],
        "sub_scores": {},
    }
    return {
        "practice_id": practice.id,
        "status": practice.status,
        "source_report_id": practice.source_report_id,
        "practice_session_id": practice.practice_session_id,
        "retest_session_id": practice.retest_session_id,
        "weakness_title": practice.weakness_title,
        "before": before,
        "after": None,
        "delta": {
            "score": 0,
            "resolved_weaknesses": [],
            "remaining_weaknesses": _string_list(before.get("weaknesses")),
            "sub_scores": {},
        },
        "summary": "专项练习已创建；完成练习和再测后将冻结前后对比结果。",
        "next_weakness": _next_weakness(source, practice.weakness_title),
    }


def _score_snapshot(session_id: int, scores: list[InterviewScore]) -> dict[str, Any]:
    if not scores:
        return {
            "session_id": session_id,
            "score": 0,
            "weaknesses": ["本次再测未生成有效评分"],
            "answer": "",
            "answer_structure": [],
            "sub_scores": {},
        }
    answers = [
        f"第 {score.question_index} 题：{score.answer.strip()}"
        for score in scores
        if score.answer.strip()
    ]
    sub_score_buckets: dict[str, list[int]] = {}
    for score in scores:
        for key, value in _int_dict(score.sub_scores).items():
            sub_score_buckets.setdefault(key, []).append(value)
    sub_scores = {
        key: round(sum(values) / len(values))
        for key, values in sub_score_buckets.items()
        if values
    }
    answer = "\n".join(answers)
    return {
        "session_id": session_id,
        "score": round(sum(score.score for score in scores) / len(scores)),
        "weaknesses": _unique_text(
            [item for score in scores for item in _string_list(score.weaknesses)]
        ),
        "answer": answer,
        "answer_structure": _detect_answer_structure(answer),
        "sub_scores": sub_scores,
    }


def _comparison_summary(
    score_delta: int,
    resolved: list[str],
    remaining: list[str],
    has_scores: bool,
) -> str:
    if not has_scores:
        return "再测会话已结束，但没有可用评分，暂时无法判断短板变化。"
    if score_delta > 0:
        score_text = f"再测得分提高 {score_delta} 分"
    elif score_delta < 0:
        score_text = f"再测得分下降 {abs(score_delta)} 分"
    else:
        score_text = "再测得分与原始得分持平"
    change_text = (
        f"，有 {len(resolved)} 项原短板未在本次评分中再次出现"
        if resolved
        else "，原短板仍需继续结合回答明细复盘"
    )
    if remaining:
        change_text += f"，当前仍记录 {len(remaining)} 项待改进内容"
    return f"{score_text}{change_text}。"


def _next_weakness(source: dict[str, Any], current_title: str) -> dict[str, str] | None:
    weaknesses = _string_list(source.get("report_weaknesses"))
    for index, title in enumerate(weaknesses):
        if title != current_title:
            return {"key": f"report_weakness_{index + 1}", "title": title}
    return None


def _detect_answer_structure(answer: str) -> list[str]:
    text = answer.strip()
    if not text:
        return []
    detected = ["已给出核心回答"]
    marker_groups = [
        ("包含背景或约束", ("背景", "当时", "场景", "目标", "约束", "需求")),
        ("说明行动或方案", ("我负责", "我采用", "我实现", "步骤", "方案", "行动")),
        ("解释取舍或依据", ("取舍", "权衡", "因为", "选择", "原因")),
        ("给出结果或验证", ("结果", "最终", "提升", "降低", "验证", "%", "指标")),
        ("包含复盘或改进", ("复盘", "改进", "下一次", "经验", "不足")),
    ]
    for label, markers in marker_groups:
        if any(marker in text for marker in markers):
            detected.append(label)
    return detected


def _find_exact_index(values: list[str], target: str) -> int:
    try:
        return values.index(target.strip())
    except ValueError:
        return -1


def _same_weakness(left: str, right: str) -> bool:
    left_key = _canonical_weakness_key(left)
    right_key = _canonical_weakness_key(right)
    if left_key and left_key == right_key:
        return True
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return min(len(left_normalized), len(right_normalized)) >= 4 and (
        left_normalized in right_normalized or right_normalized in left_normalized
    )


def _canonical_weakness_key(value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        return ""
    concept_markers = (
        ("quantified_result", ("量化", "数据结果", "结果指标", "数值结果", "百分比")),
        ("tradeoff", ("取舍", "权衡", "方案对比", "选择依据", "选型依据")),
        ("personal_contribution", ("个人贡献", "本人贡献", "个人职责", "自己负责", "职责边界")),
        ("validation", ("验证", "压测", "测试依据", "验证依据")),
        ("answer_structure", ("回答结构", "表达结构", "逻辑结构", "条理")),
        ("technical_detail", ("技术细节", "实现细节", "具体实现", "深度不足")),
        ("communication_clarity", ("表达不清", "表达模糊", "不够清晰", "过于冗长")),
        ("job_match", ("岗位匹配", "职位匹配", "岗位要求", "jd匹配")),
    )
    for key, markers in concept_markers:
        if any(marker in normalized for marker in markers):
            return key
    return normalized


def _compare_weaknesses(
    before: list[str],
    after: list[str],
    *,
    target_title: str,
    target_improved: bool,
) -> tuple[list[str], list[str]]:
    resolved: list[str] = []
    remaining = list(after)
    for item in before:
        if any(_same_weakness(item, current) for current in after):
            continue
        if target_improved and _same_weakness(item, target_title):
            resolved.append(item)
        else:
            remaining.append(item)
    resolved = _unique_text(resolved)
    remaining = [
        item
        for item in _unique_text(remaining)
        if not any(_same_weakness(item, resolved_item) for resolved_item in resolved)
    ]
    return resolved, remaining


def _target_dimension_score(scores: list[InterviewScore], target_dimension: str) -> int | None:
    target = _normalize_text(target_dimension)
    matching = [
        score.score
        for score in scores
        if target and _normalize_text(score.dimension) == target
    ]
    if not matching:
        return None
    return round(sum(matching) / len(matching))


def _normalize_text(value: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "").casefold())


def _unique_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = str(value or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def _json_value(value: Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def _json_dict(value: Any) -> dict[str, Any]:
    parsed = _json_value(value, {})
    return parsed if isinstance(parsed, dict) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            value = [value]
    if not isinstance(value, list):
        return []
    return _unique_text([str(item) for item in value])


def _int_dict(value: Any) -> dict[str, int]:
    parsed = _json_value(value, {})
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, int] = {}
    for key, item in parsed.items():
        try:
            result[str(key)] = max(0, min(100, int(item)))
        except (TypeError, ValueError):
            continue
    return result
