import pytest

import app.rag.retriever as retriever


@pytest.mark.anyio
async def test_vector_recall_filters_by_embedding_signature_and_threshold(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        async def query_points(self, **kwargs):
            captured.update(kwargs)
            return type("Response", (), {"points": []})()

    monkeypatch.setattr(retriever, "client", FakeClient())
    async def ensure_collection():
        return None

    async def embed_query(query):
        return [1.0, 0.0]

    async def active_collection():
        return "knowledge_documents__active"

    monkeypatch.setattr(retriever, "ensure_collection", ensure_collection)
    monkeypatch.setattr(retriever, "get_active_collection_name", active_collection)
    monkeypatch.setattr(retriever, "embed_text", embed_query)
    monkeypatch.setattr(retriever, "embedding_signature", lambda: "dashscope:text-embedding-v4:1024")
    monkeypatch.setattr(retriever.settings, "embedding_score_threshold", 0.42)

    assert await retriever._vector_recall("token renewal", limit=3) == []
    assert captured["score_threshold"] == 0.42
    assert captured["query_filter"].must[0].key == "embedding_signature"
    assert captured["query_filter"].must[0].match.value == "dashscope:text-embedding-v4:1024"
