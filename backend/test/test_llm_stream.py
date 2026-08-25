import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.llm_stream import (
    extract_json_string_field,
    invoke_json_with_streaming_field,
    publish_committed_text,
    reset_stream_delta_callback,
    reset_stream_text_done_callback,
    set_stream_delta_callback,
    set_stream_text_done_callback,
)


class ChunkedLLM:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks

    async def astream(self, messages):
        for chunk in self.chunks:
            yield SimpleNamespace(content=chunk)


@pytest.mark.anyio
async def test_model_question_is_streamed_as_draft_then_authoritative_text_is_published() -> None:
    chunks = [
        '{"question":"Unvalidated ',
        'question?","dimension":"backend"}',
    ]
    deltas: list[str] = []
    completed: list[str] = []

    async def capture(delta: str) -> None:
        deltas.append(delta)

    async def capture_text_done(content: str) -> None:
        completed.append(content)

    delta_token = set_stream_delta_callback(capture)
    done_token = set_stream_text_done_callback(capture_text_done)
    try:
        response = await invoke_json_with_streaming_field(
            ChunkedLLM(chunks),
            [],
            field="question",
        )
        assert "".join(deltas) == "Unvalidated question?"
        assert completed == []

        await publish_committed_text("Validated committed question?")
    finally:
        reset_stream_text_done_callback(done_token)
        reset_stream_delta_callback(delta_token)

    assert response.content == "".join(chunks)
    assert "".join(deltas) == "Unvalidated question?"
    assert completed == ["Validated committed question?"]


@pytest.mark.anyio
async def test_draft_delta_arrives_before_model_stream_finishes() -> None:
    release_stream = asyncio.Event()
    delta_received = asyncio.Event()
    deltas: list[str] = []

    class PausingLLM:
        async def astream(self, messages):
            yield SimpleNamespace(content='{"question":"Live')
            await release_stream.wait()
            yield SimpleNamespace(content=' question?","score":80}')

    async def capture(delta: str) -> None:
        deltas.append(delta)
        delta_received.set()

    token = set_stream_delta_callback(capture)
    try:
        task = asyncio.create_task(
            invoke_json_with_streaming_field(PausingLLM(), [], field="question")
        )
        await asyncio.wait_for(delta_received.wait(), timeout=1)
        assert not task.done()
        assert "".join(deltas) == "Live"

        release_stream.set()
        response = await asyncio.wait_for(task, timeout=1)
    finally:
        release_stream.set()
        reset_stream_delta_callback(token)

    assert response.content == '{"question":"Live question?","score":80}'
    assert "".join(deltas) == "Live question?"


@pytest.mark.anyio
async def test_stream_field_can_be_suppressed_for_known_replacement_path() -> None:
    deltas: list[str] = []

    async def capture(delta: str) -> None:
        deltas.append(delta)

    token = set_stream_delta_callback(capture)
    try:
        response = await invoke_json_with_streaming_field(
            ChunkedLLM(['{"question":"discard me"}']),
            [],
            field="question",
            stream_field=False,
        )
    finally:
        reset_stream_delta_callback(token)

    assert response.content == '{"question":"discard me"}'
    assert deltas == []


@pytest.mark.anyio
async def test_empty_committed_text_is_rejected() -> None:
    async def capture(delta: str) -> None:
        pass

    token = set_stream_delta_callback(capture)
    try:
        with pytest.raises(ValueError, match="non-empty"):
            await publish_committed_text("   ")
    finally:
        reset_stream_delta_callback(token)


def test_extract_json_string_field_waits_for_complete_unicode_escape() -> None:
    partial = '{"question":"hello \\u4f'
    complete = partial + '60 world"}'

    assert extract_json_string_field(partial, "question") == "hello "
    assert extract_json_string_field(complete, "question") == "hello 你 world"
