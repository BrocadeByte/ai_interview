import asyncio

import pytest

import app.rag.embeddings as embeddings


class FakeResponse:
    def __init__(self, payload: dict, *, delay: float = 0) -> None:
        self.payload = payload
        self.delay = delay
        self.status = 200

    async def __aenter__(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        return None

    async def text(self) -> str:
        return "ok"

    async def json(self) -> dict:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class FakeSession:
    def __init__(self, payload: dict, *, delay: float = 0) -> None:
        self.payload = payload
        self.delay = delay

    def post(self, url: str, *, json: dict, headers: dict) -> FakeResponse:
        return FakeResponse(self.payload, delay=self.delay)


@pytest.mark.anyio
async def test_dashscope_async_batch_restores_input_order(monkeypatch) -> None:
    monkeypatch.setattr(embeddings.settings, "embedding_dim", 3)
    session = FakeSession(
        {
            "output": {
                "embeddings": [
                    {"text_index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"text_index": 0, "embedding": [1.0, 0.0, 0.0]},
                ]
            }
        }
    )

    vectors = await embeddings.dashscope_embed_texts(["first", "second"], session=session)  # type: ignore[arg-type]

    assert vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]


@pytest.mark.anyio
async def test_dashscope_wait_does_not_block_event_loop(monkeypatch) -> None:
    monkeypatch.setattr(embeddings.settings, "embedding_dim", 2)
    session = FakeSession(
        {"output": {"embeddings": [{"text_index": 0, "embedding": [1.0, 0.0]}]}},
        delay=0.03,
    )
    heartbeat_ran = False

    async def heartbeat() -> None:
        nonlocal heartbeat_ran
        await asyncio.sleep(0.005)
        heartbeat_ran = True

    await asyncio.gather(
        embeddings.dashscope_embed_texts(["text"], session=session),  # type: ignore[arg-type]
        heartbeat(),
    )

    assert heartbeat_ran is True

@pytest.mark.anyio
async def test_embed_text_coalesces_concurrent_requests(monkeypatch) -> None:
    calls = 0
    embeddings._embedding_cache.clear()
    embeddings._embedding_inflight.clear()

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        return [[1.0, 0.0] for _text in texts]

    monkeypatch.setattr(embeddings, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(embeddings, "embedding_signature", lambda: "test:model:2")

    results = await asyncio.gather(
        embeddings.embed_text("same query"),
        embeddings.embed_text("same query"),
        embeddings.embed_text("same query"),
    )

    assert calls == 1
    assert results == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    await asyncio.sleep(0)
    assert embeddings._embedding_inflight == {}
