from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.security import create_access_token, hash_password, verify_password
from app.models.profile import UserProfile
from app.models.user import User
from app.schemas.user import SessionStatus, Token, UserCreate, UserLogin, UserRead
from app.services.auth_service import (
    RefreshTokenError,
    create_auth_session,
    revoke_all_user_sessions,
    revoke_session_from_refresh_token,
    rotate_refresh_token,
)


router = APIRouter(prefix="/auth", tags=["auth"])


# 注册新用户，创建默认求职画像，并返回登录 token。
@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(
    payload: UserCreate,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> Token:
    exists = await db.scalar(select(User).where(User.email == payload.email))
    if exists:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    user = User(email=payload.email, username=payload.username, password_hash=hash_password(payload.password))
    db.add(user)
    await db.flush()
    db.add(UserProfile(user_id=user.id))
    session, refresh_token = await create_auth_session(db, user, **_request_metadata(request))
    await db.commit()
    await db.refresh(user)
    _set_refresh_cookie(response, refresh_token, session.expires_at)
    return _token_response(user, session.id)


# 用户登录，校验邮箱和密码，成功后返回登录 token。
@router.post("/login", response_model=Token)
async def login(
    payload: UserLogin,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> Token:
    user = await db.scalar(select(User).where(User.email == payload.email))
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    session, refresh_token = await create_auth_session(db, user, **_request_metadata(request))
    await db.commit()
    await db.refresh(user)
    _set_refresh_cookie(response, refresh_token, session.expires_at)
    return _token_response(user, session.id)


# 使用 HttpOnly Cookie 中的 refresh token 轮换登录会话并签发短效 access token。
@router.post("/refresh", response_model=Token)
async def refresh(request: Request, db: AsyncSession = Depends(get_db)) -> Token | Response:
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if not refresh_token:
        return _unauthorized_refresh_response()
    try:
        user, session, rotated_token = await rotate_refresh_token(
            db,
            refresh_token,
            **_request_metadata(request),
        )
        await db.commit()
        await db.refresh(user)
    except RefreshTokenError:
        # Token 重放时服务层会撤销会话；其他失败提交只读事务也不会产生副作用。
        await db.commit()
        return _unauthorized_refresh_response()

    response = JSONResponse(content=_token_response(user, session.id).model_dump(mode="json"))
    _set_refresh_cookie(response, rotated_token, session.expires_at)
    return response


# 匿名安全的会话探测始终返回 200；存在有效 Refresh Token 时同时完成轮换续期。
@router.get("/session", response_model=SessionStatus)
async def session_status(request: Request, db: AsyncSession = Depends(get_db)) -> Response:
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if not refresh_token:
        return _session_status_response(SessionStatus(authenticated=False))
    try:
        user, session, rotated_token = await rotate_refresh_token(
            db,
            refresh_token,
            **_request_metadata(request),
        )
        await db.commit()
        await db.refresh(user)
    except RefreshTokenError:
        await db.commit()
        response = _session_status_response(SessionStatus(authenticated=False))
        _clear_refresh_cookie(response)
        return response

    token = _token_response(user, session.id)
    response = _session_status_response(
        SessionStatus(
            authenticated=True,
            access_token=token.access_token,
            token_type=token.token_type,
            expires_in=token.expires_in,
            user=token.user,
        )
    )
    _set_refresh_cookie(response, rotated_token, session.expires_at)
    return response


# 退出当前设备。即使 access token 已过期，也可以用 refresh cookie 撤销会话。
@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> None:
    refresh_token = request.cookies.get(settings.refresh_cookie_name)
    if refresh_token:
        await revoke_session_from_refresh_token(db, refresh_token)
        await db.commit()
    _clear_refresh_cookie(response)


# 撤销当前用户的全部登录会话。
@router.post("/logout-all", status_code=status.HTTP_204_NO_CONTENT)
async def logout_all(
    response: Response,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    await revoke_all_user_sessions(db, current_user.id)
    await db.commit()
    _clear_refresh_cookie(response)


# 获取当前登录用户的基础信息。
@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user


def _token_response(user: User, session_id: str) -> Token:
    return Token(
        access_token=create_access_token(str(user.id), session_id),
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserRead.model_validate(user),
    )


def _request_metadata(request: Request) -> dict[str, str | None]:
    return {
        "user_agent": request.headers.get("user-agent"),
        "ip_address": request.client.host if request.client else None,
    }


def _set_refresh_cookie(response: Response, token: str, expires_at: datetime) -> None:
    now = datetime.now(timezone.utc)
    aware_expires_at = expires_at.replace(tzinfo=timezone.utc) if expires_at.tzinfo is None else expires_at
    response.set_cookie(
        key=settings.refresh_cookie_name,
        value=token,
        max_age=max(0, int((aware_expires_at - now).total_seconds())),
        expires=aware_expires_at,
        path=settings.auth_cookie_path,
        domain=settings.auth_cookie_domain,
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite=settings.auth_cookie_samesite,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.refresh_cookie_name,
        path=settings.auth_cookie_path,
        domain=settings.auth_cookie_domain,
        secure=settings.auth_cookie_secure,
        httponly=True,
        samesite=settings.auth_cookie_samesite,
    )


def _unauthorized_refresh_response() -> JSONResponse:
    response = JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": "Invalid or expired refresh token"},
    )
    _clear_refresh_cookie(response)
    return response


def _session_status_response(payload: SessionStatus) -> JSONResponse:
    return JSONResponse(
        content=payload.model_dump(mode="json"),
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )
