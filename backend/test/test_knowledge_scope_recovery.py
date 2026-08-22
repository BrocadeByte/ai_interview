import sys
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.rag.retriever as retriever
import app.services.knowledge_service as knowledge_service
from app.core.config import settings


def _chunk(
    document_id: int,
    *,
    title: str,
    position: str,
    category: str = "scoring",
    text: str,
    version: int = 1,
    status: str = "ready",
) -> dict:
    return {
        "document_id": document_id,
        "point_id": f"{document_id}-{version}-0",
        "chunk_index": 0,
        "title": title,
        "target_position": position,
        "category": category,
        "text": text,
        "metadata": {},
        "index_version": version,
        "document_status": status,
    }


async def _install_local_client(monkeypatch, client: AsyncQdrantClient, collection_name: str) -> None:
    async def ready() -> None:
        return None

    async def active() -> str:
        return collection_name

    async def fake_embed_texts(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def fake_embed_text(text: str) -> list[float]:
        return [1.0, 0.0]

    monkeypatch.setattr(retriever, "client", client)
    monkeypatch.setattr(retriever, "ensure_collection", ready)
    monkeypatch.setattr(retriever, "get_active_collection_name", active)
    monkeypatch.setattr(retriever, "validate_embedding_settings", lambda: None)
    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever, "embed_text", fake_embed_text)
    monkeypatch.setattr(retriever, "embedding_signature", lambda: "test:embedding:2")


@pytest.mark.anyio
async def test_java_scope_excludes_python_and_explicitly_falls_back_to_general(monkeypatch) -> None:
    client = AsyncQdrantClient(location=":memory:")
    collection = "scope_test"
    await client.create_collection(
        collection,
        vectors_config=qmodels.VectorParams(size=2, distance=qmodels.Distance.COSINE),
    )
    await _install_local_client(monkeypatch, client, collection)
    await retriever.upsert_chunks(
        [
            _chunk(1, title="Java scoring", position="Java Engineer", text="JVM GC concurrency scoring"),
            _chunk(2, title="Python scoring", position="Python Engineer", text="Python GIL asyncio scoring"),
            _chunk(3, title="General scoring", position="general", text="General communication scoring"),
            _chunk(4, title="Java questions", position="Java Engineer", category="question", text="JVM question"),
            _chunk(5, title="Old Java", position="Java Engineer", text="old JVM scoring", version=2, status="failed"),
        ],
        collection_name=collection,
    )

    scoped = await retriever._persistent_recall_candidates(
        limit=20,
        target_position="Java Engineer",
        categories=["scoring"],
        statuses=["ready"],
        versions=[1],
        include_general=True,
    )
    strict = await retriever._persistent_recall_candidates(
        limit=20,
        target_position="Java Engineer",
        categories=["scoring"],
        statuses=["ready"],
        versions=[1],
        include_general=False,
    )

    assert {item["title"] for item in scoped} == {"Java scoring", "General scoring"}
    assert {item["title"] for item in strict} == {"Java scoring"}
    assert all(item["target_position"] != "Python Engineer" for item in scoped)
    await client.close()


@pytest.mark.anyio
async def test_agent_purposes_use_distinct_category_sets(monkeypatch) -> None:
    calls: list[dict] = []

    async def capture_search(query: str, limit: int = 5, **kwargs) -> list[dict]:
        calls.append(kwargs)
        return []

    monkeypatch.setattr(knowledge_service, "search_knowledge", capture_search)

    for purpose in ("planning", "scoring", "question", "report"):
        await knowledge_service.format_knowledge_context(
            "query",
            target_position="Java Engineer",
            purpose=purpose,
        )

    primary_calls = calls[::2]
    fallback_calls = calls[1::2]
    category_sets = [tuple(call["categories"]) for call in primary_calls]
    assert len(set(category_sets)) == 4
    assert all(call["target_position"] == "Java Engineer" for call in calls)
    assert all(call["include_general"] is True for call in calls)
    assert all(call["categories"] == knowledge_service.LEGACY_FILE_CATEGORIES for call in fallback_calls)
    assert all(call["legacy_title_scope"] == "Java Engineer" for call in fallback_calls)


@pytest.mark.anyio
async def test_agent_category_fallback_only_runs_after_business_categories_miss(monkeypatch) -> None:
    calls: list[list[str] | None] = []

    async def return_primary_result(query: str, limit: int = 5, **kwargs) -> list[dict]:
        calls.append(kwargs.get("categories"))
        return [_chunk(9, title="Java rubric", position="general", text="Java scoring")]

    monkeypatch.setattr(knowledge_service, "search_knowledge", return_primary_result)

    result = await knowledge_service.format_knowledge_context(
        "Java scoring",
        target_position="Java Engineer",
        purpose="scoring",
    )

    assert "Java rubric" in result
    assert calls == [knowledge_service.KNOWLEDGE_CATEGORIES["scoring"]]


@pytest.mark.anyio
async def test_explicit_categories_do_not_enable_legacy_fallback(monkeypatch) -> None:
    calls: list[list[str] | None] = []

    async def no_results(query: str, limit: int = 5, **kwargs) -> list[dict]:
        calls.append(kwargs.get("categories"))
        return []

    monkeypatch.setattr(knowledge_service, "search_knowledge", no_results)

    result = await knowledge_service.format_knowledge_context(
        "Java scoring",
        target_position="Java Engineer",
        categories=["scoring"],
    )

    assert result == "No relevant knowledge base content."
    assert calls == [["scoring"]]


def test_legacy_title_scope_matches_target_position_without_crossing_jobs() -> None:
    java = _chunk(1, title="Java后端工程师面试评分标准", position="general", category="md", text="JVM")
    go = _chunk(2, title="Go开发工程师面试评分标准", position="general", category="md", text="goroutine")

    assert retriever._legacy_title_matches(java, "java开发") is True
    assert retriever._legacy_title_matches(java, "Java Engineer") is True
    assert retriever._legacy_title_matches(go, "java开发") is False


@pytest.mark.anyio
async def test_legacy_title_scope_filters_candidates_before_rerank(monkeypatch) -> None:
    java = _chunk(1, title="Java后端工程师面试评分标准", position="general", category="md", text="JVM GC")
    go = _chunk(2, title="Go开发工程师面试评分标准", position="general", category="md", text="Go GC")

    async def vector_recall(*args, **kwargs) -> list[dict]:
        return [go, java]

    async def persistent_recall(*args, **kwargs) -> list[dict]:
        return [go, java]

    async def rerank(query: str, chunks: list[dict], *, top_n: int) -> list[dict]:
        assert {item["title"] for item in chunks} == {"Java后端工程师面试评分标准"}
        return chunks

    async def no_expansion(chunks: list[dict]) -> list[dict]:
        return chunks

    monkeypatch.setattr(retriever, "_vector_recall", vector_recall)
    monkeypatch.setattr(retriever, "_persistent_recall_candidates", persistent_recall)
    monkeypatch.setattr(retriever, "_rerank_chunks", rerank)
    monkeypatch.setattr(retriever, "_expand_adjacent_chunks", no_expansion)

    results = await retriever.search_chunks_advanced(
        "java JVM",
        categories=knowledge_service.LEGACY_FILE_CATEGORIES,
        target_position="java开发",
        legacy_title_scope="java开发",
    )

    assert [item["title"] for item in results] == ["Java后端工程师面试评分标准"]


@pytest.mark.anyio
async def test_hybrid_results_survive_client_restart(tmp_path, monkeypatch) -> None:
    collection = settings.qdrant_collection_name
    first_client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    await first_client.create_collection(
        collection,
        vectors_config=qmodels.VectorParams(size=2, distance=qmodels.Distance.COSINE),
    )
    await _install_local_client(monkeypatch, first_client, collection)
    await retriever.upsert_chunks(
        [
            _chunk(11, title="JWT rubric", position="Python Engineer", text="JWT refresh expiration token"),
            _chunk(12, title="General auth", position="general", text="authentication token security"),
            _chunk(13, title="Java auth", position="Java Engineer", text="Spring Security token"),
        ],
        collection_name=collection,
    )

    before = await retriever.search_chunks_advanced(
        "JWT token expiration",
        limit=3,
        target_position="Python Engineer",
        categories=["scoring"],
    )
    before_signature = [
        (item["point_id"], tuple(item.get("_retrieval_routes") or []))
        for item in before
    ]
    await first_client.close()

    second_client = AsyncQdrantClient(path=str(tmp_path / "qdrant"))
    await _install_local_client(monkeypatch, second_client, collection)
    after = await retriever.search_chunks_advanced(
        "JWT token expiration",
        limit=3,
        target_position="Python Engineer",
        categories=["scoring"],
    )
    after_signature = [
        (item["point_id"], tuple(item.get("_retrieval_routes") or []))
        for item in after
    ]

    assert before_signature == after_signature
    assert before_signature
    assert all(item["target_position"] != "Java Engineer" for item in after)
    await second_client.close()
