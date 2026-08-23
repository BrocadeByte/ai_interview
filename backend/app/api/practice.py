from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.interview import InterviewSessionRead
from app.schemas.practice import (
    PracticeComparisonRead,
    PracticeCreationRead,
    PracticeFromQuestionReviewCreate,
    PracticeFromReportCreate,
    PracticeListItem,
)
from app.services.practice_service import (
    create_practice_from_question_review,
    create_practice_from_report,
    get_practice_comparison,
    list_owned_practices,
    start_owned_practice,
    start_owned_retest,
)


router = APIRouter(prefix="/practice", tags=["practice"])


@router.post(
    "/from-report",
    response_model=PracticeCreationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_from_report(
    payload: PracticeFromReportCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    practice = await create_practice_from_report(db, current_user.id, payload)
    await db.commit()
    return practice


@router.post(
    "/from-question-review",
    response_model=PracticeCreationRead,
    status_code=status.HTTP_201_CREATED,
)
async def create_from_question_review(
    payload: PracticeFromQuestionReviewCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    practice = await create_practice_from_question_review(db, current_user.id, payload)
    await db.commit()
    return practice


@router.get("", response_model=list[PracticeListItem])
async def list_practices(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    practices = await list_owned_practices(db, current_user.id)
    await db.commit()
    return practices


@router.post("/{practice_id}/start", response_model=InterviewSessionRead)
async def start_practice(
    practice_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await start_owned_practice(db, current_user.id, practice_id)
    await db.commit()
    return session


@router.post("/{practice_id}/start-retest", response_model=InterviewSessionRead)
async def start_retest(
    practice_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = await start_owned_retest(db, current_user.id, practice_id)
    await db.commit()
    return session


@router.get("/{practice_id}/comparison", response_model=PracticeComparisonRead)
async def get_comparison(
    practice_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    comparison = await get_practice_comparison(db, current_user.id, practice_id)
    await db.commit()
    return comparison
