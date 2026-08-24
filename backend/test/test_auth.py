import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401, E402
from app.api import auth  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.database import Base, get_db  # noqa: E402
from app.core.security import (  # noqa: E402
    create_access_token,
    create_refresh_token,
    decode_access_token,
    decode_refresh_token,
)


auth_app = FastAPI()
auth_app.include_router(auth.router, prefix="/api")


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
async def test_session_factory():
    source_url = make_url(settings.database_url)
    test_database = f"ai_interview_auth_test_{uuid4().hex[:10]}"
    admin_url = source_url.set(database=None)
    test_url = source_url.set(database=test_database)

    admin_engine = create_async_engine(admin_url)
    async with admin_engine.begin() as conn:
        await conn.execute(
            text(
                f"CREATE DATABASE `{test_database}` "
                "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
        )
    await admin_engine.dispose()

    test_engine = create_async_engine(test_url)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    try:
        yield session_factory
    finally:
        await test_engine.dispose()
        cleanup_engine = create_async_engine(admin_url)
        async with cleanup_engine.begin() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS `{test_database}`"))
        await cleanup_engine.dispose()


@pytest.fixture
async def client(test_session_factory):
    async def override_get_db():
        async with test_session_factory() as db:
            yield db

    auth_app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        yield async_client
    auth_app.dependency_overrides.clear()


def test_token_types_are_not_interchangeable() -> None:
    expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(days=1)
    access_token = create_access_token("1", "session-1")
    refresh_token = create_refresh_token("1", "session-1", expires_at)

    assert decode_access_token(access_token) is not None
    assert decode_refresh_token(refresh_token) is not None
    assert decode_access_token(refresh_token) is None
    assert decode_refresh_token(access_token) is None
    assert decode_access_token(
        create_access_token("1", "session-1", expires_delta=timedelta(seconds=-1))
    ) is None


@pytest.mark.anyio
async def test_session_probe_is_anonymous_safe_and_renews_valid_session(
    client: AsyncClient,
) -> None:
    anonymous = await client.get("/api/auth/session")
    assert anonymous.status_code == 200
    assert anonymous.json()["authenticated"] is False
    assert anonymous.headers["cache-control"] == "no-store"

    registered = await _register(client)
    previous_refresh = client.cookies.get(settings.refresh_cookie_name)
    assert previous_refresh

    active = await client.get("/api/auth/session")
    assert active.status_code == 200
    assert active.headers["cache-control"] == "no-store"
    payload = active.json()
    assert payload["authenticated"] is True
    assert payload["user"]["id"] == registered.json()["user"]["id"]
    assert payload["access_token"]
    assert client.cookies.get(settings.refresh_cookie_name) != previous_refresh
    assert (
        await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
        )
    ).status_code == 200


@pytest.mark.anyio
async def test_session_probe_clears_expired_or_invalid_refresh_without_401(
    client: AsyncClient,
) -> None:
    client.cookies.set(
        settings.refresh_cookie_name,
        "invalid-refresh-token",
        domain="testserver.local",
        path=settings.auth_cookie_path,
    )
    response = await client.get("/api/auth/session")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["authenticated"] is False
    assert client.cookies.get(settings.refresh_cookie_name) is None


@pytest.mark.anyio
async def test_register_sets_refresh_cookie_and_expired_access_can_refresh(client: AsyncClient) -> None:
    response = await _register(client)
    payload = response.json()
    cookie_header = response.headers["set-cookie"].lower()

    assert payload["expires_in"] == settings.access_token_expire_minutes * 60
    assert "httponly" in cookie_header
    assert "samesite=lax" in cookie_header
    assert client.cookies.get(settings.refresh_cookie_name)

    claims = decode_access_token(payload["access_token"])
    assert claims is not None
    expired_access = create_access_token(
        claims.subject,
        claims.session_id,
        expires_delta=timedelta(seconds=-1),
    )
    expired_response = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {expired_access}"},
    )
    assert expired_response.status_code == 401

    refresh_response = await client.post("/api/auth/refresh")
    assert refresh_response.status_code == 200
    refreshed_access = refresh_response.json()["access_token"]
    me_response = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {refreshed_access}"},
    )
    assert me_response.status_code == 200

    refresh_as_access = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {client.cookies.get(settings.refresh_cookie_name)}"},
    )
    assert refresh_as_access.status_code == 401


@pytest.mark.anyio
async def test_concurrent_refresh_returns_one_rotated_token(client: AsyncClient) -> None:
    await _register(client)
    old_refresh = client.cookies.get(settings.refresh_cookie_name)
    assert old_refresh

    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as other_client:
        other_client.cookies.set(
            settings.refresh_cookie_name,
            old_refresh,
            domain="testserver.local",
            path=settings.auth_cookie_path,
        )
        first, second = await asyncio.gather(
            client.post("/api/auth/refresh"),
            other_client.post("/api/auth/refresh"),
        )

        assert first.status_code == second.status_code == 200
        assert client.cookies.get(settings.refresh_cookie_name) == other_client.cookies.get(
            settings.refresh_cookie_name
        )
        assert first.json()["access_token"] != second.json()["access_token"]


@pytest.mark.anyio
async def test_refresh_replay_outside_grace_revokes_session(
    client: AsyncClient,
    test_session_factory,
) -> None:
    response = await _register(client)
    old_refresh = client.cookies.get(settings.refresh_cookie_name)
    assert old_refresh
    first_refresh = await client.post("/api/auth/refresh")
    assert first_refresh.status_code == 200
    current_access = first_refresh.json()["access_token"]

    claims = decode_refresh_token(old_refresh)
    assert claims is not None
    async with test_session_factory() as db:
        await db.execute(
            text(
                "UPDATE auth_sessions "
                "SET previous_token_valid_until = :expired_at WHERE id = :session_id"
            ),
            {
                "expired_at": datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=1),
                "session_id": claims.session_id,
            },
        )
        await db.commit()

    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as replay_client:
        replay_client.cookies.set(
            settings.refresh_cookie_name,
            old_refresh,
            domain="testserver.local",
            path=settings.auth_cookie_path,
        )
        replay_response = await replay_client.post("/api/auth/refresh")
    assert replay_response.status_code == 401

    revoked_access = await client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {current_access}"},
    )
    assert revoked_access.status_code == 401
    assert (await client.post("/api/auth/refresh")).status_code == 401


@pytest.mark.anyio
async def test_logout_revokes_current_session(client: AsyncClient) -> None:
    response = await _register(client)
    access_token = response.json()["access_token"]

    logout_response = await client.post("/api/auth/logout")
    assert logout_response.status_code == 204
    assert client.cookies.get(settings.refresh_cookie_name) is None
    assert (
        await client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    ).status_code == 401


@pytest.mark.anyio
async def test_logout_all_revokes_every_device(client: AsyncClient) -> None:
    registered = await _register(client)
    email = registered.json()["user"]["email"]
    first_access = registered.json()["access_token"]

    transport = ASGITransport(app=auth_app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as second_client:
        login_response = await second_client.post(
            "/api/auth/login",
            json={"email": email, "password": "Password123"},
        )
        assert login_response.status_code == 200
        second_access = login_response.json()["access_token"]

        logout_all_response = await client.post(
            "/api/auth/logout-all",
            headers={"Authorization": f"Bearer {first_access}"},
        )
        assert logout_all_response.status_code == 204
        assert (
            await second_client.get(
                "/api/auth/me",
                headers={"Authorization": f"Bearer {second_access}"},
            )
        ).status_code == 401
        assert (await second_client.post("/api/auth/refresh")).status_code == 401


async def _register(client: AsyncClient):
    response = await client.post(
        "/api/auth/register",
        json={
            "email": f"auth_{uuid4().hex}@example.com",
            "username": "auth-user",
            "password": "Password123",
        },
    )
    assert response.status_code == 201, response.text
    return response
