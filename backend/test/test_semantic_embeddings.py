import pytest

import app.rag.embeddings as embeddings


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.status = 200

    async def __aenter__(self):
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
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def post(self, url: str, *, json: dict, headers: dict) -> FakeResponse:
        return FakeResponse(self.payload)


@pytest.mark.anyio
async def test_dashscope_batch_restores_text_order_and_validates_dimension(monkeypatch) -> None:
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
async def test_dashscope_dimension_mismatch_fails_fast(monkeypatch) -> None:
    monkeypatch.setattr(embeddings.settings, "embedding_dim", 3)
    session = FakeSession(
        {"output": {"embeddings": [{"text_index": 0, "embedding": [1.0, 0.0]}]}}
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        await embeddings.dashscope_embed_texts(["text"], session=session)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_hash_embeddings_require_explicit_test_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(embeddings.settings, "embedding_provider", "hash")
    monkeypatch.setattr(embeddings.settings, "allow_hash_embeddings", False)

    with pytest.raises(ValueError, match="test-only"):
        await embeddings.embed_text("JWT refresh strategy")


def test_embedding_signature_identifies_vector_space(monkeypatch) -> None:
    monkeypatch.setattr(embeddings.settings, "embedding_provider", "dashscope")
    monkeypatch.setattr(embeddings.settings, "embedding_model", "text-embedding-v4")
    monkeypatch.setattr(embeddings.settings, "embedding_dim", 1024)

    assert embeddings.embedding_signature() == "dashscope:text-embedding-v4:1024"