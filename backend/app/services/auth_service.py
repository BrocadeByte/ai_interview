from datetime import datetime, timedelta, timezone
from secrets import compare_digest
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_refresh_token, decode_refresh_token, hash_refresh_token
from app.models.auth_session import AuthSession
from app.models.user import User


class RefreshTokenError(Exception):
    pass


class RefreshTokenReuseError(RefreshTokenError):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def create_auth_session(
    db: AsyncSession,
    user: User,
    *,
    user_agent: str | None,
    ip_address: str | None,
) -> tuple[AuthSession, str]:
    session_id = str(uuid4())
    expires_at = utcnow() + timedelta(days=settings.refresh_token_expire_days)
    refresh_token_jti = str(uuid4())
    refresh_token_issued_at = utcnow().replace(microsecond=0)
    refresh_token = create_refresh_token(
        str(user.id),
        session_id,
        expires_at,
        token_id=refresh_token_jti,
        issued_at=refresh_token_issued_at,
    )
    session = AuthSession(
        id=session_id,
        user_id=user.id,
        refresh_token_hash=hash_refresh_token(refresh_token),
        refresh_token_jti=refresh_token_jti,
        refresh_token_issued_at=refresh_token_issued_at,
        expires_at=expires_at,
        user_agent=(user_agent or "")[:512] or None,
        ip_address=(ip_address or "")[:45] or None,
    )
    db.add(session)
    await db.flush()
    return session, refresh_token


async def rotate_refresh_token(
    db: AsyncSession,
    refresh_token: str,
    *,
    user_agent: str | None,
    ip_address: str | None,
) -> tuple[User, AuthSession, str]:
    claims = decode_refresh_token(refresh_token)
    if claims is None:
        raise RefreshTokenError("Invalid or expired refresh token")

    session = await db.scalar(
        select(AuthSession)
        .where(AuthSession.id == claims.session_id)
        .with_for_update()
    )
    now = utcnow()
    if session is None or str(session.user_id) != claims.subject:
        raise RefreshTokenError("Invalid refresh token session")
    if session.revoked_at is not None or session.expires_at <= now:
        raise RefreshTokenError("Refresh token session is no longer active")
    incoming_hash = hash_refresh_token(refresh_token)
    is_current_token = compare_digest(session.refresh_token_hash, incoming_hash)
    is_recent_previous_token = bool(
        session.previous_refresh_token_hash
        and session.previous_token_valid_until
        and session.previous_token_valid_until >= now
        and compare_digest(session.previous_refresh_token_hash, incoming_hash)
    )
    if not is_current_token and not is_recent_previous_token:
        session.revoked_at = now
        await db.flush()
        raise RefreshTokenReuseError("Refresh token reuse detected")

    user = await db.get(User, session.user_id)
    if user is None:
        session.revoked_at = now
        await db.flush()
        raise RefreshTokenError("Refresh token user no longer exists")

    if is_current_token:
        session.previous_refresh_token_hash = session.refresh_token_hash
        session.previous_token_valid_until = now + timedelta(
            seconds=settings.refresh_token_reuse_grace_seconds
        )
        session.refresh_token_jti = str(uuid4())
        session.refresh_token_issued_at = now.replace(microsecond=0)

    # 并发请求携带刚轮换的旧 token 时，重建并返回同一个新 token，避免误判重放。
    rotated_token = create_refresh_token(
        str(user.id),
        session.id,
        session.expires_at,
        token_id=session.refresh_token_jti,
        issued_at=session.refresh_token_issued_at,
    )
    session.refresh_token_hash = hash_refresh_token(rotated_token)
    session.last_used_at = now
    session.user_agent = (user_agent or session.user_agent or "")[:512] or None
    session.ip_address = (ip_address or session.ip_address or "")[:45] or None
    await db.flush()
    return user, session, rotated_token


async def revoke_session_from_refresh_token(db: AsyncSession, refresh_token: str) -> None:
    claims = decode_refresh_token(refresh_token)
    if claims is None:
        return
    session = await db.scalar(
        select(AuthSession)
        .where(AuthSession.id == claims.session_id)
        .with_for_update()
    )
    if session is not None and str(session.user_id) == claims.subject and session.revoked_at is None:
        session.revoked_at = utcnow()
        await db.flush()


async def revoke_all_user_sessions(db: AsyncSession, user_id: int) -> None:
    await db.execute(
        update(AuthSession)
        .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
