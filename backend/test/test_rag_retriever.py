import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.rag.retriever as retriever


@pytest.mark.anyio
async def test_advanced_search_uses_persistent_keyword_metadata_and_compression(monkeypatch) -> None:
    long_text = "Auth section intro. " + "JWT token expiration refresh strategy. " * 80 + "Audit logging."
    candidates = [
        {
            "document_id": 1,
            "point_id": "1-1-0",
            "chunk_index": 0,
            "text": "Redis cache avalanche and consistency.",
            "title": "Cache rubric",
            "category": "scoring",
            "target_position": "Python Backend Engineer",
            "metadata": {"section_title": "Cache", "heading_path": ["Backend", "Cache"]},
        },
        {
            "document_id": 2,
            "point_id": "2-1-0",
            "chunk_index": 0,
            "text": long_text,
            "title": "Auth rubric",
            "category": "scoring",
            "target_position": "Python Backend Engineer",
            "metadata": {"section_title": "Auth", "heading_path": ["Backend", "Auth"], "source_page": 3},
        },
    ]

    async def no_vector_results(query, limit, **kwargs):
        return []

    async def persistent_results(**kwargs):
        return candidates

    monkeypatch.setattr(retriever, "_vector_recall", no_vector_results)
    monkeypatch.setattr(retriever, "_persistent_recall_candidates", persistent_results)

    results = await retriever.search_chunks_advanced(
        "JWT token Auth",
        limit=2,
        target_position="Python Backend Engineer",
        categories=["scoring"],
    )

    assert results[0]["point_id"] == "2-1-0"
    assert results[0]["metadata"]["source_page"] == 3
    assert results[0].get("_compressed") is True
    assert len(results[0]["text"]) <= retriever.MAX_CONTEXT_CHARS_PER_CHUNK


def test_scope_filter_includes_target_general_category_status_and_version() -> None:
    query_filter = retriever._build_filter(
        target_position="Java Engineer",
        categories=["scoring"],
        statuses=["ready"],
        versions=[2],
        include_signature=False,
        include_general=True,
    )

    conditions = {
        condition.should[0].key: condition.should[0]
        for condition in query_filter.must
    }
    assert conditions["target_position_key"].match.any == ["java engineer", "general"]
    assert conditions["category_key"].match.any == ["scoring"]
    assert conditions["document_status"].match.any == ["ready"]
    assert conditions["index_version"].match.any == [2]
    assert query_filter.must[0].should[1].must[1].key == "target_position"
    assert query_filter.must[2].should[1].is_empty.key == "document_status"


def test_metadata_recall_matches_persistent_heading_metadata() -> None:
    candidates = [{
        "document_id": 1,
        "point_id": "1-1-0",
        "chunk_index": 0,
        "text": "Short text.",
        "title": "Backend guide",
        "category": "rubric",
        "target_position": "Python Backend Engineer",
        "metadata": {"section_title": "Followup", "heading_path": ["Interview", "Followup"]},
    }]

    results = retriever._metadata_recall("Followup", candidates, limit=5)

    assert len(results) == 1
    assert results[0]["point_id"] == "1-1-0"
    assert results[0]["_retrieval"]["route"] == "metadata"


def test_rrf_fuse_deduplicates_same_point_from_multiple_routes() -> None:
    fused = retriever._rrf_fuse(
        [
            [{"point_id": "1-1-0", "text": "vector", "_retrieval": {"route": "vector", "rank": 1}}],
            [{"point_id": "1-1-0", "text": "keyword", "_retrieval": {"route": "keyword", "rank": 1}}],
        ]
    )
    assert len(fused) == 1
    assert fused[0]["_retrieval_routes"] == ["keyword", "vector"]


@pytest.mark.anyio
async def test_upsert_failure_is_not_hidden(monkeypatch) -> None:
    chunks = [{
        "document_id": 7,
        "point_id": "7-1-0",
        "chunk_index": 0,
        "text": "JWT expiration strategy",
        "title": "Auth",
        "category": "rubric",
        "target_position": "Backend Engineer",
        "metadata": {},
    }]

    class FailingClient:
        async def upsert(self, **kwargs):
            assert kwargs["wait"] is True
            raise RuntimeError("qdrant unavailable")

    async def ready_collection():
        return None

    async def active_collection():
        return "knowledge_documents__active"

    async def fake_embed_texts(texts):
        return [[1.0, 0.0] for _text in texts]

    monkeypatch.setattr(retriever, "client", FailingClient())
    monkeypatch.setattr(retriever, "ensure_collection", ready_collection)
    monkeypatch.setattr(retriever, "get_active_collection_name", active_collection)
    monkeypatch.setattr(retriever, "validate_embedding_settings", lambda: None)
    monkeypatch.setattr(retriever, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(retriever, "embedding_signature", lambda: "test:embedding:2")

    with pytest.raises(RuntimeError, match="qdrant unavailable"):
        await retriever.upsert_chunks(chunks)


@pytest.mark.anyio
async def test_replace_document_deletes_only_stale_versions(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeClient:
        async def delete(self, **kwargs):
            calls.append((kwargs["collection_name"], kwargs["points_selector"]))

    async def ready_collection():
        return None

    async def active_collection():
        return "knowledge_documents__active"

    async def fake_upsert(chunks, *, collection_name=None):
        calls.append((collection_name, list(chunks)))

    monkeypatch.setattr(retriever, "client", FakeClient())
    monkeypatch.setattr(retriever, "ensure_collection", ready_collection)
    monkeypatch.setattr(retriever, "get_active_collection_name", active_collection)
    monkeypatch.setattr(retriever, "upsert_chunks", fake_upsert)

    await retriever.replace_document_chunks(9, 3, [{"point_id": "9-3-0"}])

    assert calls[0][0] == "knowledge_documents__active"
    selector = calls[1][1].filter
    assert selector.must[0].key == "document_id"
    assert selector.must[0].match.value == 9
    assert selector.must_not[0].key == "index_version"
    assert selector.must_not[0].match.value == 3


@pytest.mark.anyio
async def test_alias_switch_is_one_atomic_qdrant_operation(monkeypatch) -> None:
    recorded = []

    class FakeClient:
        async def get_aliases(self):
            return SimpleNamespace(aliases=[
                SimpleNamespace(alias_name=retriever.active_alias_name(), collection_name="old_collection")
            ])

        async def update_collection_aliases(self, **kwargs):
            recorded.append(kwargs["change_aliases_operations"])

    monkeypatch.setattr(retriever, "client", FakeClient())

    previous = await retriever.switch_active_collection("new_collection")

    assert previous == "old_collection"
    assert len(recorded) == 1
    assert recorded[0][0].delete_alias.alias_name == retriever.active_alias_name()
    assert recorded[0][1].create_alias.collection_name == "new_collection"
