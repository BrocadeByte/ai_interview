import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.rag.retriever as retriever


def _chunk(document_id: int, index: int, text: str) -> dict:
    return {
        "document_id": document_id,
        "index_version": 1,
        "document_status": "ready",
        "point_id": f"{document_id}-1-{index}",
        "chunk_index": index,
        "title": "Auth rubric",
        "category": "scoring",
        "target_position": "Python Engineer",
        "text": text,
        "metadata": {},
    }


def test_bm25_recall_prefers_specific_term_density() -> None:
    results = retriever._bm25_recall(
        "JWT refresh",
        [
            _chunk(1, 0, "JWT is used for authentication."),
            _chunk(2, 0, "JWT refresh token rotation and refresh expiration are required."),
        ],
        limit=30,
    )

    assert [item["document_id"] for item in results] == [2, 1]
    assert all(item["_retrieval"]["route"] == "bm25" for item in results)


@pytest.mark.anyio
async def test_rerank_maps_model_indexes_and_marks_normal_result(monkeypatch) -> None:
    async def fake_rerank(query: str, documents: list[str], *, top_n: int):
        assert query == "JWT refresh"
        assert len(documents) == 2
        assert top_n == 8
        return [(1, 0.91), (0, 0.42)]

    monkeypatch.setattr(retriever, "rerank_documents", fake_rerank)
    results = await retriever._rerank_chunks(
        "JWT refresh",
        [_chunk(1, 0, "weak"), _chunk(2, 0, "strong")],
        top_n=8,
    )

    assert [item["document_id"] for item in results] == [2, 1]
    assert results[0]["_rerank"]["is_fallback"] is False
    assert results[0]["_rerank"]["model"] == retriever.settings.rerank_model


@pytest.mark.anyio
async def test_rerank_failure_is_explicitly_marked(monkeypatch) -> None:
    async def failing_rerank(*args, **kwargs):
        raise retriever.RerankUnavailableError("timeout")

    monkeypatch.setattr(retriever, "rerank_documents", failing_rerank)
    results = await retriever._rerank_chunks("JWT", [_chunk(1, 0, "JWT")], top_n=8)

    assert results[0]["_rerank"]["is_fallback"] is True
    assert results[0]["_rerank"]["fallback_reason"] == "timeout"


def test_context_budget_truncates_and_tracks_tokens(monkeypatch) -> None:
    monkeypatch.setattr(retriever.settings, "rag_context_token_budget", 8)
    results = retriever._apply_context_token_budget(
        [_chunk(1, 0, "JWT refresh token rotation and expiration details")],
        8,
    )

    assert len(results) == 1
    assert results[0]["_context_truncated"] is True
    assert results[0]["_context_token_count"] <= 8


@pytest.mark.anyio
async def test_adjacent_expansion_is_same_document_and_version(monkeypatch) -> None:
    selected = [_chunk(1, 1, "middle")]
    records = [
        type("Record", (), {"payload": _chunk(1, 0, "before")})(),
        type("Record", (), {"payload": _chunk(1, 1, "middle")})(),
        type("Record", (), {"payload": _chunk(1, 2, "after")})(),
        type("Record", (), {"payload": _chunk(2, 1, "other document")})(),
        type("Record", (), {"payload": {**_chunk(1, 2, "old version"), "index_version": 2}})(),
    ]

    class FakeClient:
        async def scroll(self, **kwargs):
            return records, None

    async def ready():
        return None

    async def active():
        return "knowledge_documents"

    monkeypatch.setattr(retriever, "client", FakeClient())
    monkeypatch.setattr(retriever, "ensure_collection", ready)
    monkeypatch.setattr(retriever, "get_active_collection_name", active)
    monkeypatch.setattr(retriever.settings, "rag_adjacent_chunk_window", 1)

    results = await retriever._expand_adjacent_chunks(selected)

    assert results[0]["text"] == "before\n\nmiddle\n\nafter"
    assert results[0]["_adjacent_chunk_ids"] == ["1-1-0", "1-1-2"]
    assert [item["chunk_id"] for item in results[0]["_context_segments"]] == [
        "1-1-0",
        "1-1-1",
        "1-1-2",
    ]
