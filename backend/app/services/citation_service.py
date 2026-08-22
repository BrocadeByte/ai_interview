import json
from typing import Any

from pydantic import TypeAdapter

from app.schemas.citation import KnowledgeCitation


_CITATION_LIST_ADAPTER = TypeAdapter(list[KnowledgeCitation])


def normalize_citations(
    value: Any,
    *,
    question_index: int | None = None,
) -> list[KnowledgeCitation]:
    if value in (None, ""):
        return []
    citations = _CITATION_LIST_ADAPTER.validate_python(value)
    if question_index is None:
        return citations
    return [
        citation.model_copy(update={"question_index": question_index})
        for citation in citations
    ]


def citations_to_json(value: Any) -> str:
    citations = normalize_citations(value)
    return json.dumps(
        [citation.model_dump(mode="json") for citation in citations],
        ensure_ascii=False,
    )


def citations_from_json(value: str | None) -> list[KnowledgeCitation]:
    if not value:
        return []
    return normalize_citations(json.loads(value))


def merge_citations(*groups: Any) -> list[KnowledgeCitation]:
    merged: list[KnowledgeCitation] = []
    seen: set[tuple[Any, ...]] = set()
    for group in groups:
        for citation in normalize_citations(group):
            key = (
                citation.purpose,
                citation.question_index,
                citation.document_id,
                citation.index_version,
                citation.chunk_id,
                citation.reference,
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(citation)
    return merged
