import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.services import resume_queue  # noqa: E402


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
        self.arguments: dict[str, object] = {}

    async def bind(self, exchange, routing_key: str) -> None:
        self.bindings.append((exchange, routing_key))


class FakeChannel:
    def __init__(self) -> None:
        self.exchanges: dict[str, FakeExchange] = {}
        self.queues: dict[str, FakeQueue] = {}

    async def declare_exchange(self, name: str, _exchange_type, durable: bool):
        assert durable is True
        exchange = self.exchanges.setdefault(name, FakeExchange(name))
        return exchange

    async def declare_queue(self, name: str, durable: bool, arguments=None):
        assert durable is True
        queue = self.queues.setdefault(name, FakeQueue(name))
        queue.arguments = arguments or {}
        return queue


@pytest.mark.asyncio
async def test_resume_topology_is_durable_with_one_retry_and_dead_routes() -> None:
    topology = await resume_queue.declare_resume_topology(FakeChannel())

    retry = topology["retry_queue"]
    work = topology["work_queue"]
    dead = topology["dead_queue"]
    assert work.name == settings.rabbitmq_resume_queue
    assert retry.name == f"{settings.rabbitmq_resume_queue}.retry"
    assert dead.name == f"{settings.rabbitmq_resume_queue}.dead"
    assert retry.arguments["x-message-ttl"] == 5_000
    assert retry.arguments["x-dead-letter-exchange"] == settings.rabbitmq_resume_exchange
    assert retry.arguments["x-dead-letter-routing-key"] == "parse"
    assert work.arguments["x-dead-letter-exchange"] == settings.rabbitmq_resume_dlx
    assert work.arguments["x-dead-letter-routing-key"] == "dead"
    assert work.bindings[0][1] == "parse"
    assert dead.bindings[0][1] == "dead"


def test_resume_task_message_contains_only_ids_and_attempt() -> None:
    message = resume_queue._resume_message(123, 456, 1)
    payload = json.loads(message.body)

    assert payload == {"resume_id": 123, "user_id": 456, "attempt": 1}
    assert set(payload) == {"resume_id", "user_id", "attempt"}
    assert not {"raw_text", "file", "content", "parsed_json", "profile_patch_json"}.intersection(payload)
    assert message.delivery_mode.value == 2
    assert resume_queue.parse_resume_message(message.body) == resume_queue.ResumeParseMessage(123, 456, 1)


@pytest.mark.parametrize(
    "payload",
    [
        {"resume_id": 1, "user_id": 2},
        {"resume_id": 1, "user_id": 2, "attempt": 1, "raw_text": "forbidden"},
        {"resume_id": "1", "user_id": 2, "attempt": 1},
        {"resume_id": 1, "user_id": 2, "attempt": 0},
    ],
)
def test_invalid_resume_messages_are_rejected(payload: dict) -> None:
    with pytest.raises(ValueError):
        resume_queue.parse_resume_message(json.dumps(payload).encode("utf-8"))
