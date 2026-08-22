import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.services.knowledge_queue as knowledge_queue
from app.services.knowledge_ingestion_service import process_ingestion_task


class FakeExchange:
    def __init__(self, name: str) -> None:
        self.name = name
        self.published: list[tuple[object, str]] = []

    async def publish(self, message, routing_key: str) -> None:
        self.published.append((message, routing_key))


class FakeQueue:
    def __init__(self, name: str) -> None:
        self.name = name
        self.bindings: list[tuple[object, str]] = []
        self.channel = SimpleNamespace(default_exchange=FakeExchange(""))

    async def bind(self, exchange, routing_key: str) -> None:
        self.bindings.append((exchange, routing_key))


class FakeChannel:
    def __init__(self) -> None:
        self.exchanges: dict[str, FakeExchange] = {}
        self.queues: dict[str, FakeQueue] = {}

    async def declare_exchange(self, name: str, exchange_type, durable: bool):
        assert durable is True
        exchange = self.exchanges.setdefault(name, FakeExchange(name))
        return exchange

    async def declare_queue(self, name: str, durable: bool, arguments=None):
        assert durable is True
        queue = self.queues.setdefault(name, FakeQueue(name))
        queue.arguments = arguments or {}
        return queue


@pytest.mark.asyncio
async def test_ingestion_topology_is_durable_and_has_retry_and_dead_routes() -> None:
    topology = await knowledge_queue.declare_ingestion_topology(FakeChannel())

    retry = topology["retry_queue"]
    work = topology["work_queue"]
    assert retry.arguments["x-message-ttl"] == 5000
    assert retry.arguments["x-dead-letter-routing-key"] == "ingest"
    assert work.arguments["x-dead-letter-routing-key"] == "dead"
    assert work.bindings[0][1] == "ingest"


def test_task_message_contains_only_task_id() -> None:
    message = knowledge_queue._task_message(42)

    assert json.loads(message.body) == {"task_id": 42}
    assert message.delivery_mode.value == 2
