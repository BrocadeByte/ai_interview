import asyncio

import pytest
from starlette.middleware.gzip import GZipMiddleware

from app.api.interviews import _sse_response
from app.main import app


def test_application_does_not_enable_gzip_middleware() -> None:
    assert all(middleware.cls is not GZipMiddleware for middleware in app.user_middleware)


@pytest.mark.anyio
async def test_sse_response_emits_each_event_without_compression_buffering() -> None:
    expected_chunks = [
        b'event: delta\ndata: {"content":"first"}\n\n',
        b'event: delta\ndata: {"content":"second"}\n\n',
    ]

    async def events():
        for chunk in expected_chunks:
            yield chunk
            await asyncio.sleep(0)

    response = _sse_response(events())
    messages: list[dict] = []

    async def send(message: dict) -> None:
        messages.append(message)

    async def receive() -> dict:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    await response(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/interviews/1/start/stream",
            "raw_path": b"/api/interviews/1/start/stream",
            "query_string": b"",
            "headers": [(b"accept-encoding", b"gzip")],
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "root_path": "",
        },
        receive,
        send,
    )

    body_chunks = [
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body" and message.get("more_body")
    ]
    assert body_chunks == expected_chunks

    response_start = next(message for message in messages if message["type"] == "http.response.start")
    headers = dict(response_start["headers"])
    assert b"content-encoding" not in headers
