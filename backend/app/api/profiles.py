from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.database import get_db
from app.models.profile import UserProfile
from app.models.user import User
from app.schemas.profile import ProfileRead, ProfileUpdate


router = APIRouter(prefix="/profile", tags=["profile"])


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
    for field, value in payload.model_dump().items():
        setattr(profile, field, value)
    db.add(profile)
    await db.commit()
    await db.refresh(profile)
    return profile


# 按用户 ID 查询求职画像。
async def _get_user_profile(db: AsyncSession, user_id: int) -> UserProfile | None:
    return await db.scalar(select(UserProfile).where(UserProfile.user_id == user_id))
