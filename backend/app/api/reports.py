from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.interview import InterviewSession
from app.models.report import InterviewReport
from app.models.user import User
from app.schemas.report import InterviewReportListItem, InterviewReportRead
from app.services.report_service import repair_report_if_incomplete, report_to_read


router = APIRouter(prefix="/reports", tags=["reports"])


# 获取当前登录用户的所有面试报告列表。
@router.get("", response_model=list[InterviewReportListItem])
async def list_reports(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[InterviewReportListItem]:
    rows = await db.execute(
        select(InterviewReport, InterviewSession)
        .join(InterviewSession, InterviewReport.session_id == InterviewSession.id)
        .where(InterviewSession.user_id == current_user.id)
        .order_by(InterviewReport.created_at.desc())
    )

    return [
        InterviewReportListItem(
            id=report.id,
            session_id=report.session_id,
            target_position=session.target_position,
            difficulty=session.difficulty,
            total_score=report.total_score,
            created_at=report.created_at,
            updated_at=report.updated_at,
        )
        for report, session in rows.all()
    ]


# 获取指定报告详情，并校验报告归属当前用户。
@router.get("/{report_id}", response_model=InterviewReportRead)
async def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> InterviewReportRead:
    row = await db.execute(
        select(InterviewReport, InterviewSession)
        .join(InterviewSession, InterviewReport.session_id == InterviewSession.id)
        .where(InterviewReport.id == report_id, InterviewSession.user_id == current_user.id)
    )
    result = row.first()
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Report not found")

    report, session = result
    if await repair_report_if_incomplete(db, report, session):
        await db.commit()
    return report_to_read(report)
