from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.job_description import JobDescription
from app.models.profile import UserProfile
from app.models.resume import Resume
from app.models.user import User
from app.schemas.job_description import ParsedJobDescription
from app.schemas.profile import AutoProfileDraft, AutoProfileGenerate, ProfileRead, ProfileUpdate
from app.schemas.resume import ParsedResume, ResumeProfilePatch
from app.services.analytics_service import record_analytics_event_safely
from app.services.job_description_service import load_parsed_job_description
from app.services.profile_service import build_auto_profile_draft
from app.services.resume_service import load_json_object


router = APIRouter(prefix="/profile", tags=["profile"])


@router.post("/auto-generate", response_model=AutoProfileDraft)
async def auto_generate_profile(
    payload: AutoProfileGenerate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> AutoProfileDraft:
    """Generate a reviewable draft without writing any profile fields."""
    profile = await _get_user_profile(db, current_user.id)
    existing_profile = ProfileRead.model_validate(profile).model_dump(exclude={"id", "user_id"}) if profile else None

    resume_patch: ResumeProfilePatch | None = None
    parsed_resume: ParsedResume | None = None
    if payload.resume_id is not None:
        resume = await _get_owned_resume(db, payload.resume_id, current_user.id)
        if resume.status != "parsed" or not resume.profile_patch_json:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Resume has no parsed profile draft",
            )
        try:
            resume_patch = ResumeProfilePatch.model_validate(load_json_object(resume.profile_patch_json))
            parsed_data = load_json_object(resume.parsed_json)
            parsed_resume = ParsedResume.model_validate(parsed_data) if parsed_data is not None else None
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Stored resume parsing result is invalid",
            ) from exc

    parsed_job_description: ParsedJobDescription | None = None
    if payload.job_description_id is not None:
        job_description = await _get_owned_job_description(
            db, payload.job_description_id, current_user.id
        )
        if job_description.status != "parsed" or not job_description.parsed_json:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Job description has no parsed result",
            )
        try:
            parsed_job_description = load_parsed_job_description(job_description.parsed_json)
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Stored job description parsing result is invalid",
            ) from exc

    draft = build_auto_profile_draft(
        existing_profile=existing_profile,
        resume_patch=resume_patch,
        parsed_resume=parsed_resume,
        parsed_job_description=parsed_job_description,
        requested_target_position=payload.target_position,
    )
    await record_analytics_event_safely(
        db,
        event_name="profile_auto_generated",
        user_id=current_user.id,
        resume_id=payload.resume_id,
        job_description_id=payload.job_description_id,
        properties={"completeness": draft.completeness},
    )
    await db.commit()
    return draft


# 获取当前登录用户的求职画像；如果不存在则自动创建空画像。
@router.get("/me", response_model=ProfileRead)
async def get_profile(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)) -> UserProfile:
    profile = await _get_user_profile(db, current_user.id)
    if not profile:
        profile = UserProfile(user_id=current_user.id)
        db.add(profile)
        await db.commit()
        await db.refresh(profile)
    return profile


# 更新当前登录用户的求职画像信息。
@router.put("/me", response_model=ProfileRead)
async def update_profile(
    payload: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    profile = await _get_user_profile(db, current_user.id) or UserProfile(user_id=current_user.id)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(profile, field, value)
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return profile


# 按用户 ID 查询求职画像。
async def _get_user_profile(db: AsyncSession, user_id: int) -> UserProfile | None:
    return await db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))


async def _get_owned_resume(db: AsyncSession, resume_id: int, user_id: int) -> Resume:
    resume = await db.scalar(select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id))
    if resume is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resume not found")
    return resume


async def _get_owned_job_description(
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
