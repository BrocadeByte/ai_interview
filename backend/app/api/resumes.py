from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.profile import UserProfile
from app.models.resume import Resume
from app.models.user import User
from app.schemas.profile import ProfileRead
from app.schemas.resume import ParsedResume, ResumePaste, ResumeProfilePatch, ResumeRead
from app.services.analytics_service import record_analytics_event_safely
from app.services.resume_service import (
    load_json_object,
    parse_resume_text,
    parse_resume_upload,
    safe_parse_error,
    serialize_resume_parse_output,
    validate_resume_text,
)


router = APIRouter(prefix="/resumes", tags=["resumes"])


@router.post("/paste", response_model=ResumeRead, status_code=status.HTTP_201_CREATED)
async def paste_resume(
    payload: ResumePaste,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResumeRead:
    raw_text = validate_resume_text(payload.content)
    resume = Resume(
        user_id=current_user.id,
        title=payload.title,
        source_type="paste",
        raw_text=raw_text,
        status="pending",
    )
    return await _persist_and_parse_resume(db, resume)


@router.post("/upload", response_model=ResumeRead, status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResumeRead:
    upload = await parse_resume_upload(file, title)
    resume = Resume(
        user_id=current_user.id,
        title=upload.title,
        source_type="upload",
        file_name=upload.file_name,
        file_type=upload.file_type,
        raw_text=upload.raw_text,
        status="pending",
    )
    return await _persist_and_parse_resume(db, resume)


@router.get("", response_model=list[ResumeRead])
async def list_resumes(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[ResumeRead]:
    rows = (
        await db.scalars(
            select(Resume)
            .where(Resume.user_id == current_user.id)
            .order_by(Resume.created_at.desc(), Resume.id.desc())
        )
    ).all()
    return [_to_resume_read(resume) for resume in rows]


@router.get("/{resume_id}", response_model=ResumeRead)
async def get_resume(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResumeRead:
    return _to_resume_read(await _get_owned_resume(db, resume_id, current_user.id))


@router.post("/{resume_id}/activate", response_model=ResumeRead)
async def activate_resume(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResumeRead:
    resume = await _get_owned_resume(db, resume_id, current_user.id)
    if resume.status != "parsed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Only a parsed resume can be activated")

    await db.execute(
        update(Resume)
        .where(Resume.user_id == current_user.id, Resume.id != resume.id)
        .values(is_active=False)
    )
    resume.is_active = True
    await db.commit()
    await db.refresh(resume)
    return _to_resume_read(resume)


@router.post("/{resume_id}/apply-profile", response_model=ProfileRead)
async def apply_resume_to_profile(
    resume_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> UserProfile:
    resume = await _get_owned_resume(db, resume_id, current_user.id)
    if resume.status != "parsed" or not resume.profile_patch_json:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Resume has no parsed profile draft to apply",
        )

    try:
        patch = ResumeProfilePatch.model_validate(load_json_object(resume.profile_patch_json))
    except (ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Stored resume profile draft is invalid",
        ) from exc

    profile = await db.scalar(select(UserProfile).where(UserProfile.user_id == current_user.id))
    if profile is None:
        profile = UserProfile(user_id=current_user.id)
    for field, value in patch.model_dump(exclude_none=True).items():
        setattr(profile, field, value)
    db.add(profile)
    await record_analytics_event_safely(
        db,
        event_name="profile_applied",
        user_id=current_user.id,
        resume_id=resume.id,
        deduplication_key=f"profile_applied:resume:{current_user.id}:{resume.id}",
        properties={"source": "resume_profile_patch"},
    )
    await db.commit()
    await db.refresh(profile)
    return profile


async def _persist_and_parse_resume(db: AsyncSession, resume: Resume) -> ResumeRead:
    db.add(resume)
    await db.commit()
    await db.refresh(resume)

    try:
        output = await parse_resume_text(resume.raw_text)
        resume.parsed_json, resume.profile_patch_json = serialize_resume_parse_output(output)
        resume.status = "parsed"
        resume.error_message = None
        await record_analytics_event_safely(
            db,
            event_name="resume_uploaded" if resume.source_type == "upload" else "resume_pasted",
            user_id=resume.user_id,
            resume_id=resume.id,
            deduplication_key=f"resume_{resume.source_type}:{resume.id}",
            properties={"file_type": resume.file_type} if resume.file_type else None,
        )
        await db.commit()
        await db.refresh(resume)
    except Exception as exc:
        await db.rollback()
        resume.status = "failed"
        resume.error_message = safe_parse_error(exc)
        resume.parsed_json = None
        resume.profile_patch_json = None
        db.add(resume)
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Resume parsing failed; the existing profile was not changed",
        ) from exc
    return _to_resume_read(resume)


async def _get_owned_resume(db: AsyncSession, resume_id: int, user_id: int) -> Resume:
    resume = await db.scalar(select(Resume).where(Resume.id == resume_id, Resume.user_id == user_id))
    if resume is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resume not found")
    return resume


def _to_resume_read(resume: Resume) -> ResumeRead:
    parsed_data = load_json_object(resume.parsed_json)
    profile_patch_data = load_json_object(resume.profile_patch_json)
    return ResumeRead(
        id=resume.id,
        title=resume.title,
        source_type=resume.source_type,
        file_name=resume.file_name,
        file_type=resume.file_type,
        raw_text=resume.raw_text,
        status=resume.status,
        error_message=resume.error_message,
        is_active=resume.is_active,
        parsed=ParsedResume.model_validate(parsed_data) if parsed_data is not None else None,
        profile_patch=ResumeProfilePatch.model_validate(profile_patch_data) if profile_patch_data is not None else None,
        created_at=resume.created_at,
        updated_at=resume.updated_at,
    )
