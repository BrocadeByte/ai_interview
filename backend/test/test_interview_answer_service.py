import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.services.interview_answer_service as answer_service

from app.services.interview_answer_service import (
    SessionLeaseHeartbeat,
    SessionLeaseLostError,
    acquire_answer_lease,
    release_answer_lease,
    renew_answer_lease,
)


class FakeSession:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount
        self.statements = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return SimpleNamespace(rowcount=self.rowcount)

    async def commit(self) -> None:
        self.commits += 1


@pytest.mark.anyio
async def test_acquire_answer_lease_returns_database_claim_result() -> None:
    acquired_db = FakeSession(rowcount=1)
    rejected_db = FakeSession(rowcount=0)

    assert await acquire_answer_lease(
        acquired_db,  # type: ignore[arg-type]
        session_id=1,
        user_id=2,
        request_id="d6ffbea6-82a6-45df-88ef-c24f5380ba44",
    ) is True
    assert await acquire_answer_lease(
        rejected_db,  # type: ignore[arg-type]
        session_id=1,
        user_id=2,
        request_id="6c48de67-6435-463f-b030-fc3ae3b044db",
    ) is False
    assert acquired_db.commits == 1
    assert rejected_db.commits == 1


@pytest.mark.anyio
async def test_release_answer_lease_can_share_caller_transaction() -> None:
    db = FakeSession(rowcount=1)

    await release_answer_lease(
        db,  # type: ignore[arg-type]
        session_id=1,
        request_id="d6ffbea6-82a6-45df-88ef-c24f5380ba44",
        commit=False,
    )

    assert len(db.statements) == 1
    assert db.commits == 0


@pytest.mark.anyio
async def test_renew_answer_lease_returns_database_ownership_result() -> None:
    owned_db = FakeSession(rowcount=1)
    stale_db = FakeSession(rowcount=0)

    assert await renew_answer_lease(
        owned_db,  # type: ignore[arg-type]
        session_id=1,
        request_id="d6ffbea6-82a6-45df-88ef-c24f5380ba44",
    ) is True
    assert await renew_answer_lease(
        stale_db,  # type: ignore[arg-type]
        session_id=1,
        request_id="6c48de67-6435-463f-b030-fc3ae3b044db",
    ) is False
    assert owned_db.commits == stale_db.commits == 1


@pytest.mark.anyio
async def test_release_answer_lease_fences_stale_owner_before_commit() -> None:
    db = FakeSession(rowcount=0)

    with pytest.raises(SessionLeaseLostError, match="ownership was lost"):
        await release_answer_lease(
            db,  # type: ignore[arg-type]
            session_id=1,
            request_id="d6ffbea6-82a6-45df-88ef-c24f5380ba44",
            commit=False,
            require_held=True,
        )

    assert db.commits == 0
@pytest.mark.anyio
async def test_session_lease_heartbeat_renews_until_stopped(monkeypatch) -> None:
    renewed = asyncio.Event()
    calls = 0

    class FakeContext:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    def fake_factory():
        return FakeContext()

    async def fake_renew(db, **kwargs):
        nonlocal calls
        calls += 1
        renewed.set()
        return True

    monkeypatch.setattr(answer_service, "renew_answer_lease", fake_renew)
    heartbeat = SessionLeaseHeartbeat(
        fake_factory,  # type: ignore[arg-type]
        session_id=1,
        request_id="d6ffbea6-82a6-45df-88ef-c24f5380ba44",
        interval_seconds=0.01,
    )

    heartbeat.start()
    await asyncio.wait_for(renewed.wait(), timeout=1)
    await heartbeat.stop()

    assert calls >= 1
    assert heartbeat.lost is False
