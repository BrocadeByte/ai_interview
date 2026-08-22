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
async def test_model_json_is_buffered_until_authoritative_text_is_published() -> None:
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
        assert deltas == []
        assert completed == []

        await publish_committed_text("Validated committed question?", chunk_chars=10)
    finally:
        reset_stream_text_done_callback(done_token)
        reset_stream_delta_callback(delta_token)

    assert response.content == "".join(chunks)
    assert "".join(deltas) == "Validated committed question?"
    assert completed == ["Validated committed question?"]


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
