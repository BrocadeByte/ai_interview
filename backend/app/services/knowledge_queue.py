import json
import logging

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, Message

from app.core.config import settings


logger = logging.getLogger(__name__)


async def declare_ingestion_topology(channel: aio_pika.abc.AbstractChannel) -> dict[str, aio_pika.abc.AbstractQueue]:
    """Declare durable work, retry and dead-letter routes."""
    dead_exchange = await channel.declare_exchange(settings.rabbitmq_ingestion_dlx, ExchangeType.DIRECT, durable=True)
    main_exchange = await channel.declare_exchange(settings.rabbitmq_ingestion_exchange, ExchangeType.DIRECT, durable=True)
    dead_queue = await channel.declare_queue(f"{settings.rabbitmq_ingestion_queue}.dead", durable=True)
    await dead_queue.bind(dead_exchange, routing_key="dead")
    retry_queue = await channel.declare_queue(
        f"{settings.rabbitmq_ingestion_queue}.retry",
        durable=True,
        arguments={
            "x-message-ttl": 5000,
            "x-dead-letter-exchange": settings.rabbitmq_ingestion_exchange,
            "x-dead-letter-routing-key": "ingest",
        },
    )
    work_queue = await channel.declare_queue(
        settings.rabbitmq_ingestion_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": settings.rabbitmq_ingestion_dlx,
            "x-dead-letter-routing-key": "dead",
        },
    )
    await work_queue.bind(main_exchange, routing_key="ingest")
    return {"exchange": main_exchange, "work_queue": work_queue, "retry_queue": retry_queue, "dead_queue": dead_queue}


def _task_message(task_id: int) -> Message:
    return Message(
        json.dumps({"task_id": task_id}).encode("utf-8"),
        content_type="application/json",
        delivery_mode=DeliveryMode.PERSISTENT,
        message_id=str(task_id),
    )


async def _publish(task_id: int, route: str) -> None:
    connection = await aio_pika.connect_robust(
        settings.rabbitmq_url,
        timeout=settings.rabbitmq_connect_timeout_seconds,
    )
    try:
        channel = await connection.channel(publisher_confirms=True)
        topology = await declare_ingestion_topology(channel)
        if route == "retry":
            await topology["retry_queue"].channel.default_exchange.publish(
                _task_message(task_id), routing_key=topology["retry_queue"].name
            )
        elif route == "dead":
            await topology["dead_queue"].channel.default_exchange.publish(
                _task_message(task_id), routing_key=topology["dead_queue"].name
            )
        else:
            await topology["exchange"].publish(_task_message(task_id), routing_key="ingest")
    finally:
        await connection.close()


async def publish_ingestion_task(task_id: int) -> None:
    await _publish(task_id, "ingest")


async def publish_ingestion_retry(task_id: int) -> None:
    await _publish(task_id, "retry")


async def publish_ingestion_dead_letter(task_id: int) -> None:
    await _publish(task_id, "dead")
