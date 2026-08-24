import asyncio
from contextlib import asynccontextmanager, suppress
from collections.abc import AsyncGenerator
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from sqlalchemy import text

import app.models  # noqa: F401
from app.api import auth, interviews, job_descriptions, knowledge, knowledge_async, profiles, reports, resumes
from app.core.config import settings
from app.core.database import Base, engine
from app.rag.embeddings import close_embedding_session
from app.rag.retriever import close_retriever_client, warm_retriever


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await ensure_user_columns(conn)
        await ensure_knowledge_document_columns(conn)
        await ensure_interview_message_columns(conn)
        await ensure_interview_session_columns(conn)
        await ensure_interview_report_columns(conn)
        await ensure_interview_score_columns(conn)
    qdrant_warmup_task = asyncio.create_task(warm_retriever())
    try:
        yield
    finally:
        if not qdrant_warmup_task.done():
            qdrant_warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await qdrant_warmup_task
        await close_retriever_client()
        await close_embedding_session()


async def ensure_user_columns(conn) -> None:
    result = await conn.execute(text("SHOW COLUMNS FROM users"))
    existing_columns = {row[0] for row in result.fetchall()}
    migrations = {"is_admin": "ALTER TABLE users ADD COLUMN is_admin BOOL NOT NULL DEFAULT FALSE"}
    for column, statement in migrations.items():
        if column not in existing_columns:
            await conn.execute(text(statement))


async def ensure_knowledge_document_columns(conn) -> None:
    column_type = await conn.execute(
        text(
            "SELECT DATA_TYPE FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'knowledge_documents' AND COLUMN_NAME = 'content'"
        )
    )
    current_type = column_type.scalar()
    if current_type and current_type.lower() != "longtext":
        await conn.execute(text("ALTER TABLE knowledge_documents MODIFY COLUMN content LONGTEXT NOT NULL"))
    result = await conn.execute(text("SHOW COLUMNS FROM knowledge_documents"))
    existing_columns = {row[0] for row in result.fetchall()}
    migrations = {
        "original_oss_key": "ALTER TABLE knowledge_documents ADD COLUMN original_oss_key VARCHAR(512) NULL",
        "parsed_text_oss_key": "ALTER TABLE knowledge_documents ADD COLUMN parsed_text_oss_key VARCHAR(512) NULL",
        "file_name": "ALTER TABLE knowledge_documents ADD COLUMN file_name VARCHAR(255) NULL",
        "file_type": "ALTER TABLE knowledge_documents ADD COLUMN file_type VARCHAR(32) NULL",
        "file_size": "ALTER TABLE knowledge_documents ADD COLUMN file_size INT NULL",
        "chunk_count": "ALTER TABLE knowledge_documents ADD COLUMN chunk_count INT NOT NULL DEFAULT 0",
        "index_version": "ALTER TABLE knowledge_documents ADD COLUMN index_version INT NOT NULL DEFAULT 1",
        "parse_status": "ALTER TABLE knowledge_documents ADD COLUMN parse_status VARCHAR(32) NOT NULL DEFAULT 'completed'",
        "parse_error": "ALTER TABLE knowledge_documents ADD COLUMN parse_error TEXT NULL",
    }
    for column, statement in migrations.items():
        if column not in existing_columns:
            await conn.execute(text(statement))


async def ensure_interview_message_columns(conn) -> None:
    result = await conn.execute(text("SHOW COLUMNS FROM interview_messages"))
    existing_columns = {row[0] for row in result.fetchall()}
    migrations = {
        "question_index": "ALTER TABLE interview_messages ADD COLUMN question_index INT NOT NULL DEFAULT 1",
        "dimension": "ALTER TABLE interview_messages ADD COLUMN dimension VARCHAR(160) NULL",
        "request_id": "ALTER TABLE interview_messages ADD COLUMN request_id VARCHAR(36) NULL, ADD UNIQUE KEY uq_interview_message_request (session_id, request_id)",
        "is_followup": "ALTER TABLE interview_messages ADD COLUMN is_followup INT NOT NULL DEFAULT 0",
        "followup_index": "ALTER TABLE interview_messages ADD COLUMN followup_index INT NOT NULL DEFAULT 0",
    }
    for column, statement in migrations.items():
        if column not in existing_columns:
            await conn.execute(text(statement))
    await ensure_interview_request_index(conn)


async def ensure_interview_request_index(conn) -> None:
    rows = (await conn.execute(text("SHOW INDEX FROM interview_messages"))).fetchall()
    indexes: dict[str, list[tuple[int, str, int]]] = {}
    for row in rows:
        indexes.setdefault(str(row[2]), []).append((int(row[3]), str(row[4]), int(row[1])))
    expected_columns = ["session_id", "request_id"]
    for parts in indexes.values():
        ordered = sorted(parts, key=lambda part: part[0])
        if [part[1] for part in ordered] == expected_columns and all(part[2] == 0 for part in ordered):
            return
    request_index = indexes.get("request_id")
    if request_index:
        ordered = sorted(request_index, key=lambda part: part[0])
        if [part[1] for part in ordered] == ["request_id"] and all(part[2] == 0 for part in ordered):
            await conn.execute(text("ALTER TABLE interview_messages DROP INDEX request_id"))
    await conn.execute(text("ALTER TABLE interview_messages ADD UNIQUE KEY uq_interview_message_request (session_id, request_id)"))


async def ensure_interview_session_columns(conn) -> None:
    result = await conn.execute(text("SHOW COLUMNS FROM interview_sessions"))
    existing_columns = {row[0] for row in result.fetchall()}
    migrations = {
        "interview_plan_json": "ALTER TABLE interview_sessions ADD COLUMN interview_plan_json TEXT NULL",
        "job_description_id": "ALTER TABLE interview_sessions ADD COLUMN job_description_id INT NULL",
        "job_description_snapshot_json": "ALTER TABLE interview_sessions ADD COLUMN job_description_snapshot_json LONGTEXT NULL",
        "processing_request_id": "ALTER TABLE interview_sessions ADD COLUMN processing_request_id VARCHAR(36) NULL",
        "processing_started_at": "ALTER TABLE interview_sessions ADD COLUMN processing_started_at DATETIME NULL",
    }
    for column, statement in migrations.items():
        if column not in existing_columns:
            await conn.execute(text(statement))
    indexes = (await conn.execute(text("SHOW INDEX FROM interview_sessions"))).fetchall()
    if not any(str(row[2]) == "ix_interview_sessions_job_description_id" for row in indexes):
        await conn.execute(
            text(
                "ALTER TABLE interview_sessions "
                "ADD INDEX ix_interview_sessions_job_description_id (job_description_id)"
            )
        )


async def ensure_interview_report_columns(conn) -> None:
    result = await conn.execute(text("SHOW COLUMNS FROM interview_reports"))
    existing_columns = {row[0] for row in result.fetchall()}
    migrations = {
        "dimension_scores_json": "ALTER TABLE interview_reports ADD COLUMN dimension_scores_json TEXT NULL",
        "citations_json": "ALTER TABLE interview_reports ADD COLUMN citations_json TEXT NULL",
    }
    for column, statement in migrations.items():
        if column not in existing_columns:
            await conn.execute(text(statement))


async def ensure_interview_score_columns(conn) -> None:
    result = await conn.execute(text("SHOW COLUMNS FROM interview_scores"))
    existing_columns = {row[0] for row in result.fetchall()}
    migrations = {
        "is_fallback": "ALTER TABLE interview_scores ADD COLUMN is_fallback BOOL NOT NULL DEFAULT FALSE",
        "fallback_reason": "ALTER TABLE interview_scores ADD COLUMN fallback_reason TEXT NULL",
        "citations_json": "ALTER TABLE interview_scores ADD COLUMN citations_json TEXT NULL",
    }
    for column, statement in migrations.items():
        if column not in existing_columns:
            await conn.execute(text(statement))


app = FastAPI(title=settings.app_name, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth.router, prefix="/api")
app.include_router(profiles.router, prefix="/api")
app.include_router(interviews.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(resumes.router, prefix="/api")
app.include_router(job_descriptions.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(knowledge_async.router, prefix="/api")
