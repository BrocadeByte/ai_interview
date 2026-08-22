import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.knowledge import KnowledgeDocument, KnowledgeIngestionTask
from app.rag.document_loader import document_to_chunks
from app.rag.retriever import upsert_chunks
from app.schemas.knowledge import KnowledgeDocumentCreate
from app.services.knowledge_file_service import get_document_parser, normalize_extracted_text
from app.services.knowledge_quality import KnowledgeQualityError, validate_knowledge_text
from app.storage.oss_client import OssObjectStorage, build_knowledge_oss_key


logger = logging.getLogger(__name__)


async def create_upload_task(
    db: AsyncSession,
    *,
    title: str,
    category: str,
    target_position: str,
    filename: str,
    file_type: str,
    file_size: int,
    original_content: bytes,
) -> KnowledgeIngestionTask:
    source_key = build_knowledge_oss_key("original", filename)
    await OssObjectStorage().upload_bytes(original_content, source_key)
    task = KnowledgeIngestionTask(
        title=title,
        category=category,
        target_position=target_position,
        file_name=filename,
        file_type=file_type,
        file_size=file_size,
        source_oss_key=source_key,
        max_attempts=settings.knowledge_ingestion_max_attempts,
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)
    return task


async def process_ingestion_task(db: AsyncSession, task_id: int) -> None:
    task = await db.scalar(select(KnowledgeIngestionTask).where(KnowledgeIngestionTask.id == task_id))
    if not task or task.status in {"succeeded", "running"}:
        return

    task.status = "running"
    task.stage = "parsing"
    task.attempts += 1
    task.started_at = datetime.utcnow()
    task.error = None
    await db.commit()
    try:
        original = await OssObjectStorage().download_bytes(task.source_oss_key)
        parsed = normalize_extracted_text(get_document_parser(f".{task.file_type}").parse(original, task.file_name))
        quality = validate_knowledge_text(parsed)
        parsed_key = task.parsed_text_oss_key or build_knowledge_oss_key("parsed", f"{Path(task.file_name).stem}.txt")
        await OssObjectStorage().upload_bytes(parsed.encode("utf-8"), parsed_key)
        task.parsed_text_oss_key = parsed_key
        task.stage = "embedding"
        await db.commit()

        document = await db.scalar(select(KnowledgeDocument).where(KnowledgeDocument.id == task.document_id)) if task.document_id else None
        if not document:
            metadata = {
                "source": "async_upload",
                "filename": task.file_name,
                "file_type": task.file_type,
                "quality_report": quality.to_metadata(),
                "original_oss_key": task.source_oss_key,
                "parsed_text_oss_key": parsed_key,
                "file_size": task.file_size,
            }
            document = KnowledgeDocument(
                title=task.title,
                category=task.category,
                target_position=task.target_position,
                content=parsed[:8000],
                metadata_json=json.dumps(metadata, ensure_ascii=False),
                original_oss_key=task.source_oss_key,
                parsed_text_oss_key=parsed_key,
                file_name=task.file_name,
                file_type=task.file_type,
                file_size=task.file_size,
                parse_status="indexing",
                index_version=1,
            )
            db.add(document)
            await db.flush()
            task.document_id = document.id

        payload = KnowledgeDocumentCreate(
            title=task.title,
            category=task.category,
            target_position=task.target_position,
            content=parsed,
            metadata=json.loads(document.metadata_json or "{}"),
        )
        chunks = document_to_chunks(payload)
        for chunk in chunks:
            chunk.update(
                document_id=document.id,
                point_id=f"{document.id}-{document.index_version}-{chunk['chunk_index']}",
                document_status="ready",
                index_version=document.index_version,
                original_oss_key=task.source_oss_key,
                parsed_text_oss_key=parsed_key,
            )
        document.chunk_count = len(chunks)
        task.stage = "indexing"
        await db.commit()
        try:
            await upsert_chunks(chunks)
        except Exception as exc:
            document.parse_status = "failed"
            document.parse_error = f"{type(exc).__name__}: {exc}"[:2000]
            await db.commit()
            raise
        document.parse_status = "ready"
        document.parse_error = None
        task.status = "succeeded"
        task.stage = "completed"
        task.finished_at = datetime.utcnow()
        await db.commit()
    except KnowledgeQualityError as exc:
        task.status = "dead"
        task.stage = "quality_failed"
        task.error = f"KnowledgeQualityError: {exc}"[:2000]
        task.finished_at = datetime.utcnow()
        await db.commit()
        raise
    except Exception as exc:
        task.status = "failed" if task.attempts < task.max_attempts else "dead"
        task.stage = "failed"
        task.error = f"{type(exc).__name__}: {exc}"[:2000]
        task.finished_at = datetime.utcnow()
        await db.commit()
        logger.exception("knowledge.ingestion.failed task_id=%s", task_id)
        raise
