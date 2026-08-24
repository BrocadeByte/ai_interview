import asyncio
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any

import aio_pika
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine
from app.rag import retriever


DependencyProbe = Callable[[], Awaitable[None]]


async def probe_database() -> None:
    async with engine.connect() as connection:
        await connection.execute(text("SELECT 1"))


async def probe_qdrant() -> None:
    await retriever.client.get_collections()


async def probe_rabbitmq() -> None:
    connection = await aio_pika.connect_robust(
        settings.rabbitmq_url,
        timeout=settings.healthcheck_timeout_seconds,
    )
    await connection.close()


async def readiness_status() -> tuple[dict[str, Any], int]:
    database, qdrant, rabbitmq = await asyncio.gather(
        _run_probe(probe_database, required=True),
        _run_probe(probe_qdrant, required=False),
        _run_probe(probe_rabbitmq, required=False),
    )
    dependencies = {
        "database": database,
        "qdrant": qdrant,
        "rabbitmq": rabbitmq,
    }
    if database["status"] == "down":
        overall_status = "down"
        status_code = 503
    elif any(item["status"] != "ok" for item in dependencies.values()):
        overall_status = "degraded"
        status_code = 200
    else:
        overall_status = "ok"
        status_code = 200
    return {"status": overall_status, "dependencies": dependencies}, status_code


async def _run_probe(probe: DependencyProbe, *, required: bool) -> dict[str, Any]:
    started_at = perf_counter()
    try:
        await asyncio.wait_for(probe(), timeout=settings.healthcheck_timeout_seconds)
    except Exception as exc:
        return {
            "status": "down" if required else "degraded",
            "latency_ms": round((perf_counter() - started_at) * 1000),
            "error": type(exc).__name__,
        }
    return {
        "status": "ok",
        "latency_ms": round((perf_counter() - started_at) * 1000),
    }
