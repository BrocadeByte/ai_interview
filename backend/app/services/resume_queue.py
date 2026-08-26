import json
from dataclasses import dataclass
from typing import Any

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message

from app.core.config import settings


RESUME_PARSE_ROUTING_KEY = "parse"
RESUME_DEAD_ROUTING_KEY = "dead"
RESUME_RETRY_DELAY_MS = 5_000


@dataclass(frozen=True)
class ResumeParseMessage:
    resume_id: int
    user_id: int
    attempt: int


async def declare_resume_topology(channel: aio_pika.abc.AbstractChannel) -> dict[str, Any]:
    """Declare the durable resume work, delayed retry, and dead-letter routes."""
    dead_exchange = await channel.declare_exchange(
        settings.rabbitmq_resume_dlx,
        ExchangeType.DIRECT,
        durable=True,
    )
    main_exchange = await channel.declare_exchange(
        settings.rabbitmq_resume_exchange,
        ExchangeType.DIRECT,
        durable=True,
    )
    dead_queue = await channel.declare_queue(
        f"{settings.rabbitmq_resume_queue}.dead",
        durable=True,
    )
    await dead_queue.bind(dead_exchange, routing_key=RESUME_DEAD_ROUTING_KEY)
    retry_queue = await channel.declare_queue(
        f"{settings.rabbitmq_resume_queue}.retry",
        durable=True,
        arguments={
            "x-message-ttl": RESUME_RETRY_DELAY_MS,
            "x-dead-letter-exchange": settings.rabbitmq_resume_exchange,
            "x-dead-letter-routing-key": RESUME_PARSE_ROUTING_KEY,
        },
    )
    work_queue = await channel.declare_queue(
        settings.rabbitmq_resume_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": settings.rabbitmq_resume_dlx,
            "x-dead-letter-routing-key": RESUME_DEAD_ROUTING_KEY,
        },
    )
    await work_queue.bind(main_exchange, routing_key=RESUME_PARSE_ROUTING_KEY)
    return {
        "exchange": main_exchange,
        "work_queue": work_queue,
        "retry_queue": retry_queue,
        "dead_queue": dead_queue,
    }


def _resume_message(resume_id: int, user_id: int, attempt: int) -> Message:
    payload = {"resume_id": resume_id, "user_id": user_id, "attempt": attempt}
    return Message(
        json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        content_type="application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
        message_id=f"resume:{resume_id}:attempt:{attempt}",
    )


def parse_resume_message(body: bytes) -> ResumeParseMessage:
    payload = json.loads(body.decode("utf-8"))
    expected_keys = {"resume_id", "user_id", "attempt"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Resume task message has an invalid shape")
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], int) for key in expected_keys):
        raise ValueError("Resume task message fields must be integers")
    if payload["resume_id"] <= 0 or payload["user_id"] <= 0 or payload["attempt"] <= 0:
        raise ValueError("Resume task message fields must be positive")
    return ResumeParseMessage(**payload)


async def _publish(resume_id: int, user_id: int, attempt: int, route: str) -> None:
    connection = await aio_pika.connect_robust(
        settings.rabbitmq_url,
        timeout=settings.rabbitmq_connect_timeout_seconds,
    )
    try:
        channel = await connection.channel(publisher_confirms=True)
        topology = await declare_resume_topology(channel)
        message = _resume_message(resume_id, user_id, attempt)
        if route == "retry":
            await topology["retry_queue"].channel.default_exchange.publish(
                message,
                routing_key=topology["retry_queue"].name,
            )
        elif route == "dead":
            await topology["dead_queue"].channel.default_exchange.publish(
                message,
                routing_key=topology["dead_queue"].name,
            )
        else:
            await topology["exchange"].publish(message, routing_key=RESUME_PARSE_ROUTING_KEY)
    finally:
        await connection.close()


async def publish_resume_parse_task(resume_id: int, user_id: int, attempt: int = 1) -> None:
    await _publish(resume_id, user_id, attempt, "parse")


async def publish_resume_parse_retry(resume_id: int, user_id: int, attempt: int) -> None:
    await _publish(resume_id, user_id, attempt, "retry")


async def publish_resume_parse_dead_letter(resume_id: int, user_id: int, attempt: int) -> None:
    await _publish(resume_id, user_id, attempt, "dead")
