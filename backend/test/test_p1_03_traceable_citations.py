import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.rag.retriever as retriever
import app.services.knowledge_service as knowledge_service
import app.services.report_service as report_service
from app.services.score_service import _score_to_read, save_latest_score


def _citation(
    *,
    purpose: str,
    title: str = "JWT scoring rubric",
    document_id: int = 17,
    index_version: int = 3,
    chunk_id: str = "17-3-4",
) -> dict:
    return {
        "reference": 1,
        "query": "Python JWT refresh token",
        "purpose": purpose,
        "document_id": document_id,
        "title": title,
        "category": "scoring",
        "target_position": "Python Backend Engineer",
        "index_version": index_version,
        "chunk_id": chunk_id,
        "chunk_index": 4,
        "source_page": 12,
        "retrieval_score": 0.93,
        "retrieval_routes": ["bm25", "dense"],
        "retrieval_scores": {"dense": 0.82, "bm25": 4.7},
        "retrieval_ranks": {"dense": 2, "bm25": 1},
        "rrf_score": 0.0325,
        "rerank_score": 0.93,
        "rerank_provider": "dashscope",
        "rerank_model": "gte-rerank-v2",
        "rerank_rank": 1,
        "context_chunks": [
            {
                "chunk_id": chunk_id,
                "chunk_index": 4,
                "source_page": 12,
                "is_primary": True,
                "context_token_count": 96,
                "context_truncated": False,
            },
            {
                "chunk_id": "17-3-5",
                "chunk_index": 5,
                "source_page": 13,
                "is_primary": False,
                "context_token_count": 51,
                "context_truncated": True,
            },
        ],
        "context_token_count": 147,
        "context_truncated": True,
    }


@pytest.mark.anyio
async def test_context_citation_captures_retrieval_and_actual_context_chunks(monkeypatch) -> None:
    async def fake_search(*args, **kwargs):
        return [{
            "document_id": 17,
            "title": "JWT scoring rubric",
            "category": "scoring",
            "target_position": "Python Backend Engineer",
            "index_version": 3,
            "point_id": "17-3-4",
            "chunk_index": 4,
            "source_page": 12,
            "section_title": "Refresh token rotation",
            "text": "Use refresh token rotation and revoke reused token families.",
            "metadata": {},
            "_retrieval_routes": ["bm25", "dense"],
            "_retrieval_scores": {"dense": 0.82, "bm25": 4.7},
            "_retrieval_ranks": {"dense": 2, "bm25": 1},
            "_rrf_score": 0.0325,
            "_rerank_score": 0.93,
            "_rerank": {
                "provider": "dashscope",
                "model": "gte-rerank-v2",
                "rank": 1,
                "is_fallback": False,
            },
            "_context_chunks": _citation(purpose="answer")["context_chunks"],
            "_context_token_count": 147,
            "_context_truncated": True,
        }]

    monkeypatch.setattr(knowledge_service, "search_knowledge", fake_search)
    result = await knowledge_service.format_knowledge_context(
        "Python JWT refresh token",
        target_position="Python Backend Engineer",
        purpose="answer",
        with_citations=True,
    )

    citation = result.citations[0]
    assert citation.document_id == 17
    assert citation.index_version == 3
    assert citation.chunk_id == "17-3-4"
    assert citation.source_page == 12
    assert citation.retrieval_scores == {"dense": 0.82, "bm25": 4.7}
    assert citation.retrieval_ranks == {"dense": 2, "bm25": 1}
    assert citation.rrf_score == pytest.approx(0.0325)
    assert citation.rerank_score == pytest.approx(0.93)
    assert [item.chunk_id for item in citation.context_chunks] == ["17-3-4", "17-3-5"]
    assert citation.context_chunks[1].context_truncated is True


def test_token_budget_prioritizes_primary_chunk_and_only_cites_used_segments() -> None:
    selected = retriever._apply_context_token_budget(
        [{
            "document_id": 17,
            "index_version": 3,
            "point_id": "17-3-4",
            "chunk_index": 4,
            "text": "unused combined text",
            "metadata": {},
            "_context_segments": [
                {
                    "chunk_id": "17-3-3",
                    "chunk_index": 3,
                    "source_page": 11,
                    "is_primary": False,
                    "text": "相邻内容",
                },
                {
                    "chunk_id": "17-3-4",
                    "chunk_index": 4,
                    "source_page": 12,
                    "is_primary": True,
                    "text": "主块内容",
                },
            ],
        }],
        budget=4,
    )

    assert selected[0]["text"] == "主块内容"
    assert [item["chunk_id"] for item in selected[0]["_context_chunks"]] == ["17-3-4"]
    assert selected[0]["_context_chunks"][0]["is_primary"] is True
    assert selected[0]["_context_truncated"] is True


@pytest.mark.anyio
async def test_empty_retrieval_returns_empty_citations(monkeypatch) -> None:
    async def no_results(*args, **kwargs):
        return []

    monkeypatch.setattr(knowledge_service, "search_knowledge", no_results)
    result = await knowledge_service.format_knowledge_context(
        "missing topic",
        purpose="scoring",
        with_citations=True,
    )

    assert result.text == "No relevant knowledge base content."
    assert result.citations == []


class _FakeScoreDb:
    def __init__(self) -> None:
        self.record = None

    def add(self, record) -> None:
        self.record = record

    async def flush(self) -> None:
        self.record.id = 1
        self.record.created_at = datetime(2026, 8, 11, 12, 0, 0)


@pytest.mark.anyio
async def test_score_citation_is_persisted_with_question_index() -> None:
    db = _FakeScoreDb()
    await save_latest_score(db, 9, [{
        "question_index": 2,
        "question": "How do you rotate refresh tokens?",
        "answer": "Rotate on every use and detect reuse.",
        "dimension": "Authentication",
        "score": 88,
        "sub_scores": {"accuracy": 90},
        "reason": "Covers rotation and reuse detection.",
        "weaknesses": [],
        "suggestions": [],
        "citations": [_citation(purpose="answer")],
    }])

    stored = json.loads(db.record.citations_json)
    assert stored[0]["question_index"] == 2
    assert stored[0]["document_id"] == 17
    assert stored[0]["context_chunks"][1]["chunk_id"] == "17-3-5"

    response = _score_to_read(db.record)
    assert response.citations[0].question_index == 2
    assert response.citations[0].retrieval_scores["bm25"] == pytest.approx(4.7)


class _StoredScore:
    def model_dump(self) -> dict:
        score_citation = _citation(purpose="answer")
        score_citation["question_index"] = 1
        return {
            "question_index": 1,
            "question": "JWT question",
            "answer": "JWT answer",
            "dimension": "Authentication",
            "score": 86,
            "sub_scores": {},
            "reason": "Good",
            "weaknesses": [],
            "suggestions": [],
            "citations": [score_citation],
        }


class _FakeReportDb:
    def __init__(self) -> None:
        self.report = None
        self.session = SimpleNamespace(
            target_position="Python Backend Engineer",
            interview_plan_json=None,
        )

    async def scalar(self, statement):
        return self.report

    async def get(self, model, record_id):
        return self.session

    def add(self, report) -> None:
        self.report = report

    async def flush(self) -> None:
        self.report.id = 33
        self.report.created_at = datetime(2026, 8, 11, 12, 0, 0)
        self.report.updated_at = datetime(2026, 8, 11, 12, 0, 0)


@pytest.mark.anyio
async def test_report_keeps_saved_snapshot_when_knowledge_source_changes(monkeypatch) -> None:
    generated_title = {"value": "Report rubric v3"}
    generation_calls = 0

    async def no_messages(db, session_id):
        return []

    async def stored_scores(db, session_id):
        return [_StoredScore()]

    async def no_question_reviews(db, report):
        return []

    async def generate(*args, **kwargs):
        nonlocal generation_calls
        generation_calls += 1
        return {
            "summary": "Structured report",
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
            "learning_path": [],
            "sample_answer": "Sample",
            "citations": [_citation(
                purpose="report",
                title=generated_title["value"],
                document_id=21,
                index_version=3,
                chunk_id="21-3-2",
            )],
        }

    monkeypatch.setattr(report_service, "_list_messages", no_messages)
    monkeypatch.setattr(report_service, "list_scores", stored_scores)
    monkeypatch.setattr(report_service, "generate_report", generate)
    monkeypatch.setattr(report_service, "ensure_question_reviews", no_question_reviews)
    db = _FakeReportDb()

    first = await report_service.get_or_create_report(db, 9)
    generated_title["value"] = "Report rubric v4"
    second = await report_service.get_or_create_report(db, 9)

    assert generation_calls == 1
    assert [citation.purpose for citation in first.citations] == ["answer", "report"]
    assert second.citations[1].title == "Report rubric v3"
    assert second.citations[1].index_version == 3
    assert json.loads(db.report.citations_json)[1]["title"] == "Report rubric v3"
