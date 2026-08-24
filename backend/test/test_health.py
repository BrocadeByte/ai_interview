import sys
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.main import app  # noqa: E402
from app.services import health_service  # noqa: E402


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


async def _ok() -> None:
    return None


@pytest.mark.anyio
async def test_health_liveness_does_not_probe_dependencies(monkeypatch) -> None:
    async def unexpected_probe() -> None:
        raise AssertionError("liveness must not probe dependencies")

    monkeypatch.setattr(health_service, "probe_database", unexpected_probe)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_health_readiness_reports_optional_dependencies_as_degraded(monkeypatch) -> None:
    async def qdrant_down() -> None:
        raise ConnectionError("secret connection string must not be exposed")

    async def rabbitmq_down() -> None:
        raise TimeoutError("secret credentials must not be exposed")

    monkeypatch.setattr(health_service, "probe_database", _ok)
    monkeypatch.setattr(health_service, "probe_qdrant", qdrant_down)
    monkeypatch.setattr(health_service, "probe_rabbitmq", rabbitmq_down)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get("/api/health/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "degraded"
    assert payload["dependencies"]["database"]["status"] == "ok"
    assert payload["dependencies"]["qdrant"]["status"] == "degraded"
    assert payload["dependencies"]["qdrant"]["error"] == "ConnectionError"
    assert payload["dependencies"]["rabbitmq"]["status"] == "degraded"
    assert "secret" not in response.text


@pytest.mark.anyio
async def test_health_readiness_returns_503_when_database_is_down(monkeypatch) -> None:
    async def database_down() -> None:
        raise ConnectionError("database unavailable")

    monkeypatch.setattr(health_service, "probe_database", database_down)
    monkeypatch.setattr(health_service, "probe_qdrant", _ok)
    monkeypatch.setattr(health_service, "probe_rabbitmq", _ok)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        ready = await client.get("/api/health/ready")
        compatibility = await client.get("/api/health")

    assert ready.status_code == compatibility.status_code == 503
    assert ready.json()["status"] == compatibility.json()["status"] == "down"
    assert ready.json()["dependencies"]["database"]["status"] == "down"
