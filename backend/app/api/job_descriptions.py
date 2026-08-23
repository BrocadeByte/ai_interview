from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.job_description import JobDescription
from app.models.user import User
from app.schemas.job_description import JobDescriptionParse, JobDescriptionRead
from app.services.analytics_service import record_analytics_event_safely
from app.services.job_description_service import (
    parse_job_description_text,
    safe_job_description_parse_error,
    serialize_parsed_job_description,
    to_job_description_read,
    validate_job_description_text,
)


router = APIRouter(prefix="/job-descriptions", tags=["job-descriptions"])


@router.post("/parse", response_model=JobDescriptionRead, status_code=status.HTTP_201_CREATED)
async def parse_job_description(
    payload: JobDescriptionParse,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JobDescriptionRead:
    raw_text = validate_job_description_text(payload.raw_text)
    job_description = JobDescription(
        user_id=current_user.id,
        title=payload.title,
        company_name=payload.company_name,
        raw_text=raw_text,
        target_position=payload.title[:160],
        status="pending",
    )
    db.add(job_description)
    await db.commit()
    await db.refresh(job_description)

    try:
        parsed = await parse_job_description_text(raw_text)
        job_description.parsed_json = serialize_parsed_job_description(parsed)
        job_description.target_position = parsed.target_position
        job_description.status = "parsed"
        job_description.error_message = None
        await record_analytics_event_safely(
            db,
            event_name="jd_pasted",
            user_id=current_user.id,
            job_description_id=job_description.id,
            deduplication_key=f"jd_pasted:{job_description.id}",
        )
        await db.commit()
        await db.refresh(job_description)
    except Exception as exc:
        await db.rollback()
        job_description.status = "failed"
        job_description.error_message = safe_job_description_parse_error(exc)
        job_description.parsed_json = None
        db.add(job_description)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Job description parsing failed",
        ) from exc
    return to_job_description_read(job_description)


@router.get("", response_model=list[JobDescriptionRead])
async def list_job_descriptions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[JobDescriptionRead]:
    rows = (
        await db.scalars(
            select(JobDescription)
            .where(JobDescription.user_id == current_user.id)
            .order_by(JobDescription.created_at.desc(), JobDescription.id.desc())
        )
    ).all()
    return [to_job_description_read(row) for row in rows]


@router.get("/{job_description_id}", response_model=JobDescriptionRead)
async def get_job_description(
    job_description_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JobDescriptionRead:
    return to_job_description_read(
        await get_owned_job_description(db, job_description_id, current_user.id)
    )


@router.post("/{job_description_id}/activate", response_model=JobDescriptionRead)
async def activate_job_description(
    job_description_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> JobDescriptionRead:
    job_description = await get_owned_job_description(db, job_description_id, current_user.id)
    if job_description.status != "parsed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Only a parsed job description can be activated",
        )
    await db.execute(
        update(JobDescription)
        .where(JobDescription.user_id == current_user.id, JobDescription.id != job_description.id)
        .values(is_active=False)
    )
    job_description.is_active = True
    await db.commit()
    await db.refresh(job_description)
    return to_job_description_read(job_description)


async def get_owned_job_description(
    db: AsyncSession,
    job_description_id: int,
    user_id: int,
) -> JobDescription:
    job_description = await db.scalar(
        select(JobDescription).where(
            JobDescription.id == job_description_id,
            JobDescription.user_id == user_id,
        )
    )
    if job_description is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job description not found")
    return job_description
