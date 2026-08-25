import json
import re
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar, Token
from types import SimpleNamespace
from typing import Any


StreamDeltaCallback = Callable[[str], Awaitable[None]]
StreamTextDoneCallback = Callable[[str], Awaitable[None]]
MAX_STREAMED_FIELD_CHARS = 2000

_stream_delta_callback: ContextVar[StreamDeltaCallback | None] = ContextVar(
    "stream_delta_callback",
    default=None,
)
_stream_text_done_callback: ContextVar[StreamTextDoneCallback | None] = ContextVar(
    "stream_text_done_callback",
    default=None,
)


def set_stream_delta_callback(callback: StreamDeltaCallback) -> Token:
    return _stream_delta_callback.set(callback)


def reset_stream_delta_callback(token: Token) -> None:
    _stream_delta_callback.reset(token)


def set_stream_text_done_callback(callback: StreamTextDoneCallback) -> Token:
    return _stream_text_done_callback.set(callback)


def reset_stream_text_done_callback(token: Token) -> None:
    _stream_text_done_callback.reset(token)


async def invoke_json_with_streaming_field(
    llm: Any,
    messages: Sequence[Any],
    *,
    field: str,
    stream_field: bool = True,
) -> Any:
    """Collect a model stream while publishing one provisional JSON string field.

    Only the requested string value is exposed; the surrounding JSON and private scoring
    fields remain buffered. The emitted deltas are provisional and must be reconciled with
    publish_committed_text after Schema/business validation and the database commit.
    """
    callback = _stream_delta_callback.get()
    if callback is None:
        return await llm.ainvoke(messages)

    chunks: list[str] = []
    published = ""
    async for chunk in llm.astream(messages):
        text = _content_to_text(chunk.content)
        if text:
            chunks.append(text)
        if not stream_field:
            continue

        extracted = extract_json_string_field("".join(chunks), field)[:MAX_STREAMED_FIELD_CHARS]
        if not extracted.startswith(published):
            # 正常追加式模型输出应让字段单调增长；供应商行为异常时停止草稿推送，
            # 最终由提交后的 text_done 安全校正。
            stream_field = False
            continue
        delta = extracted[len(published) :]
        if delta:
            await callback(delta)
            published = extracted
    return SimpleNamespace(content="".join(chunks))


async def publish_committed_text(text: str) -> None:
    """Confirm one authoritative committed text after any provisional field deltas."""
    callback = _stream_delta_callback.get()
    done_callback = _stream_text_done_callback.get()
    if callback is None and done_callback is None:
        return
    authoritative = str(text or "").strip()
    if not authoritative:
        raise ValueError("Committed stream text must be non-empty")
    if done_callback is not None:
        await done_callback(authoritative)
    elif callback is not None:
        # 兼容只注册单一回调的独立调用方。
        await callback(authoritative)


def extract_json_string_field(raw: str, field: str) -> str:
    """Extract a possibly partial JSON string field for diagnostics and tests."""
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', raw)
    if not match:
        return ""

    result: list[str] = []
    index = match.end()
    while index < len(raw):
        char = raw[index]
        if char == '"':
            break
        if char != "\\":
            result.append(char)
            index += 1
            continue
        if index + 1 >= len(raw):
            break
        escape = raw[index + 1]
        if escape == "u":
            digits = raw[index + 2 : index + 6]
            if len(digits) < 4 or not all(value in "0123456789abcdefABCDEF" for value in digits):
                break
            result.append(chr(int(digits, 16)))
            index += 6
            continue
        result.append(json.loads(f'"\\{escape}"'))
        index += 2
    return "".join(result)


def is_json_string_field_complete(raw: str, field: str) -> bool:
    match = re.search(rf'"{re.escape(field)}"\s*:\s*"', raw)
    if not match:
        return False
    escaped = False
    for char in raw[match.end() :]:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return True
    return False


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    return str(content or "")
