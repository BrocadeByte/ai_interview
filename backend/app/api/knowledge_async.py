from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin_user
from app.core.database import get_db
from app.models.knowledge import KnowledgeIngestionTask
from app.models.user import User
from app.schemas.knowledge_task import KnowledgeIngestionTaskRead
from app.services.knowledge_file_service import ALLOWED_KNOWLEDGE_SUFFIXES, MAX_KNOWLEDGE_FILE_BYTES
from app.services.knowledge_ingestion_service import create_upload_task
from app.services.knowledge_queue import publish_ingestion_task


router = APIRouter(prefix="/knowledge", tags=["knowledge-ingestion"])


@router.post("/files/async", response_model=KnowledgeIngestionTaskRead, status_code=status.HTTP_202_ACCEPTED)
async def upload_document_file_async(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    category: str = Form(...),
    target_position: str | None = Form(default=None),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeIngestionTaskRead:
    filename = file.filename or "uploaded"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_KNOWLEDGE_SUFFIXES:
        raise HTTPException(status_code=400, detail="Only .txt, .md and .pdf files are supported")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > MAX_KNOWLEDGE_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")
    task = await create_upload_task(
        db,
        title=(title or Path(filename).stem).strip(),
        category=category.strip(),
        target_position=(target_position or "general").strip() or "general",
        filename=filename,
        file_type=suffix[1:],
        file_size=len(content),
        original_content=content,
    )
    try:
        await publish_ingestion_task(task.id)
    except Exception as exc:
        task.status = "publish_failed"
        task.stage = "publish_failed"
        task.error = f"{type(exc).__name__}: {exc}"[:2000]
        await db.commit()
        raise HTTPException(status_code=503, detail="Task saved but RabbitMQ publish failed") from exc
    return _to_read(task)


@router.get("/ingestion-tasks", response_model=list[KnowledgeIngestionTaskRead])
async def list_ingestion_tasks(
    task_status: str | None = None,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> list[KnowledgeIngestionTaskRead]:
    statement = select(KnowledgeIngestionTask).order_by(KnowledgeIngestionTask.created_at.desc())
    if task_status:
        statement = statement.where(KnowledgeIngestionTask.status == task_status)
    tasks = await db.scalars(statement)
    return [_to_read(task) for task in tasks]


@router.get("/ingestion-tasks/{task_id}", response_model=KnowledgeIngestionTaskRead)
async def get_ingestion_task(
    task_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeIngestionTaskRead:
    return _to_read(await _get_task(db, task_id))


@router.post("/ingestion-tasks/{task_id}/retry", response_model=KnowledgeIngestionTaskRead)
async def retry_ingestion_task(
    task_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeIngestionTaskRead:
    return await _reset_and_publish(db, task_id, reset_attempts=False)


@router.post("/ingestion-tasks/{task_id}/reexecute", response_model=KnowledgeIngestionTaskRead)
async def reexecute_ingestion_task(
    task_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeIngestionTaskRead:
    return await _reset_and_publish(db, task_id, reset_attempts=True)


async def _reset_and_publish(db: AsyncSession, task_id: int, *, reset_attempts: bool) -> KnowledgeIngestionTaskRead:
    task = await _get_task(db, task_id)
    allowed = {"failed", "dead", "publish_failed"}
    if reset_attempts:
        allowed.add("succeeded")
    if task.status not in allowed:
        raise HTTPException(status_code=409, detail="Task cannot be retried in its current state")
    task.status = "pending"
    task.stage = "queued"
    task.error = None
    task.finished_at = None
    if reset_attempts:
        task.attempts = 0
    await db.commit()
    try:
        await publish_ingestion_task(task.id)
    except Exception as exc:
        task.status = "publish_failed"
        task.stage = "publish_failed"
        task.error = f"{type(exc).__name__}: {exc}"[:2000]
        await db.commit()
        raise HTTPException(status_code=503, detail="Task reset but RabbitMQ publish failed") from exc
    return _to_read(task)


async def _get_task(db: AsyncSession, task_id: int) -> KnowledgeIngestionTask:
    task = await db.scalar(select(KnowledgeIngestionTask).where(KnowledgeIngestionTask.id == task_id))
    if not task:
        raise HTTPException(status_code=404, detail="Ingestion task not found")
    return task


def _to_read(task: KnowledgeIngestionTask) -> KnowledgeIngestionTaskRead:
    return KnowledgeIngestionTaskRead.model_validate(task, from_attributes=True)
