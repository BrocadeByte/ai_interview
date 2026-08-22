from __future__ import annotations

import asyncio

import aiohttp

from app.core.config import settings


class RerankUnavailableError(RuntimeError):
    """The configured rerank model cannot produce a usable ranking."""


async def rerank_documents(query: str, documents: list[str], *, top_n: int) -> list[tuple[int, float]]:
    """Return DashScope rerank results as original document indexes and scores."""
    if not settings.rerank_enabled:
        raise RerankUnavailableError("rerank is disabled")
    if not settings.dashscope_api_key:
        raise RerankUnavailableError("DASHSCOPE_API_KEY is required for reranking")
    if not query or not documents:
        return []

    payload = {
        "model": settings.rerank_model,
        "input": {"query": query, "documents": documents},
        "parameters": {"top_n": min(top_n, len(documents)), "return_documents": False},
    }
    headers = {"Authorization": f"Bearer {settings.dashscope_api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=settings.rerank_timeout_seconds)
    try:
        async with asyncio.timeout(settings.rerank_timeout_seconds):
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(settings.dashscope_rerank_url, json=payload, headers=headers) as response:
                    body = await response.text()
                    if response.status >= 400:
                        raise RerankUnavailableError(f"DashScope rerank HTTP {response.status}: {body[:300]}")
                    data = await response.json()
    except RerankUnavailableError:
        raise
    except Exception as exc:
        raise RerankUnavailableError(f"DashScope rerank request failed: {type(exc).__name__}: {exc}") from exc

    results = data.get("output", {}).get("results") or []
    ranked: list[tuple[int, float]] = []
    seen_indexes: set[int] = set()
    for item in results:
        try:
            index = int(item["index"])
            score = float(item["relevance_score"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RerankUnavailableError("DashScope rerank returned an invalid result item") from exc
        if 0 <= index < len(documents) and index not in seen_indexes:
            ranked.append((index, score))
            seen_indexes.add(index)
    if not ranked:
        raise RerankUnavailableError("DashScope rerank returned no usable results")
    return ranked
