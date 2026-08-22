import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.rag.document_loader import chunk_text, chunk_text_with_metadata, document_to_chunks
from app.schemas.knowledge import KnowledgeDocumentCreate


def test_chunk_text_prefers_headings_and_paragraphs_over_fixed_slices() -> None:
    text = """# Auth Module

JWT login flow should cover password hashing, token issuing, expiration checks and refresh strategy.

# Cache Module

Redis cache design should explain penetration, breakdown, avalanche and consistency handling.
"""

    chunks = chunk_text(text, max_chars=110, overlap_chars=0)

    assert len(chunks) >= 2
    assert chunks[0].startswith("# Auth Module")
    assert "JWT login flow" in chunks[0]
    assert any(chunk.startswith("# Cache Module") for chunk in chunks)


def test_chunk_text_splits_oversized_paragraph_by_sentence() -> None:
    sentence = "API design should describe validation, auth, error handling and trace logs."
    text = "# API Design\n\n" + " ".join([sentence] * 12)

    chunks = chunk_text(text, max_chars=180, overlap_chars=0)

    assert len(chunks) > 1
    assert all(len(chunk) <= 220 for chunk in chunks)
    assert all("# API Design" in chunk for chunk in chunks)



def test_markdown_heading_path_is_added_to_chunk_metadata() -> None:
    document = KnowledgeDocumentCreate(
        title="Backend rubric",
        category="rubric",
        target_position="Python Backend Engineer",
        content="# Backend\n\n## Auth\n\nJWT expiration and refresh strategy.\n\n## Cache\n\nRedis consistency strategy.",
        metadata={"source": "unit-test"},
    )

    chunks = document_to_chunks(document)

    assert chunks[0]["metadata"]["heading_path"] == ["Backend", "Auth"]
    assert chunks[0]["metadata"]["section_title"] == "Auth"
    assert chunks[0]["heading_path"] == ["Backend", "Auth"]
    assert any(chunk["metadata"].get("section_title") == "Cache" for chunk in chunks)


def test_page_markers_are_added_to_chunk_metadata() -> None:
    document = KnowledgeDocumentCreate(
        title="PDF rubric",
        category="rubric",
        target_position="Python Backend Engineer",
        content="Page 1\n# Auth\n\nToken validation.\n\nPage 2\n# Cache\n\nRedis consistency.",
        metadata={"source": "upload", "file_type": "pdf"},
    )

    chunks = document_to_chunks(document)

    assert chunks[0]["metadata"]["source_page"] == 1
    assert chunks[0]["source_page"] == 1
    assert any(chunk["metadata"].get("source_page") == 2 for chunk in chunks)


def test_chunk_text_with_metadata_exposes_section_and_page() -> None:
    chunks = chunk_text_with_metadata("Page 3\n# Interview\n\nQuestion generation rubric.")

    assert chunks[0].source_page == 3
    assert chunks[0].heading_path == ["Interview"]
    assert chunks[0].section_title == "Interview"
