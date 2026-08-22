import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.services.knowledge_file_service as knowledge_file_service
from app.services.knowledge_file_service import (
    MarkdownDocumentParser,
    PdfDocumentParser,
    TextDocumentParser,
    get_document_parser,
    normalize_extracted_text,
)


def test_get_document_parser_returns_type_specific_parsers() -> None:
    assert isinstance(get_document_parser(".txt"), TextDocumentParser)
    assert isinstance(get_document_parser(".md"), MarkdownDocumentParser)
    assert isinstance(get_document_parser(".pdf"), PdfDocumentParser)


def test_markdown_parser_preserves_markdown_structure() -> None:
    parser = get_document_parser(".md")

    text = parser.parse(b"# Auth\n\n- token expiration\n- password hashing", "auth.md")

    assert "# Auth" in text
    assert "- token expiration" in text


def test_normalize_extracted_text_removes_repeated_blank_lines() -> None:
    text = normalize_extracted_text("  line one  \r\n\r\n\r\n line two  ")

    assert text == "line one\n\nline two"


def test_pdf_parser_uses_ocr_fallback_when_selectable_text_is_empty(monkeypatch) -> None:
    calls = []

    def fake_extract_pdf_text(content: bytes) -> str:
        calls.append("text")
        return ""

    def fake_extract_pdf_text_with_ocr(content: bytes) -> str:
        calls.append("ocr")
        return "ocr text"

    monkeypatch.setattr(knowledge_file_service, "extract_pdf_text", fake_extract_pdf_text)
    monkeypatch.setattr(knowledge_file_service, "extract_pdf_text_with_ocr", fake_extract_pdf_text_with_ocr)

    text = PdfDocumentParser().parse(b"pdf", "scan.pdf")

    assert text == "ocr text"
    assert calls == ["text", "ocr"]


def test_pdf_ocr_fallback_returns_422_when_ocr_dependencies_are_missing(monkeypatch) -> None:
    def fake_import(name, *args, **kwargs):
        if name in {"pdf2image", "pytesseract"}:
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(HTTPException) as exc_info:
        knowledge_file_service.extract_pdf_text_with_ocr(b"pdf")

    assert exc_info.value.status_code == 422
    assert "scanned PDF" in exc_info.value.detail
