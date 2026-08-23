from datetime import date, datetime, timedelta

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin_user, get_current_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.analytics import AnalyticsMetricsRead, ClientAnalyticsEventCreate
from app.services.analytics_service import build_analytics_metrics, record_client_event


router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.post("/events", status_code=status.HTTP_204_NO_CONTENT)
async def create_client_event(
    payload: ClientAnalyticsEventCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    """Accept only the two UI interactions that the server cannot infer reliably."""
    await record_client_event(db, user_id=current_user.id, payload=payload)
    await db.commit()


@router.get("/metrics", response_model=AnalyticsMetricsRead)
async def get_metrics(
    period_start: date | None = Query(default=None),
    period_end: date | None = Query(default=None),
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> AnalyticsMetricsRead:
    del current_admin
    today = datetime.utcnow().date()
    resolved_end = period_end or today
    resolved_start = period_start or (resolved_end - timedelta(days=29))
    return await build_analytics_metrics(
        db,
        period_start=resolved_start,
        period_end=resolved_end,
    )
