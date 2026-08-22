from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_admin_user
from app.core.database import get_db
from app.models.user import User
from app.schemas.knowledge import (
    KnowledgeDocumentCreate,
    KnowledgeDocumentRead,
    KnowledgeDocumentUpdate,
    KnowledgeReindexResult,
)
from app.services.knowledge_file_service import build_document_from_upload
from app.services.knowledge_service import (
    KnowledgeDataQualityError,
    KnowledgeIndexingError,
    KnowledgeReindexConflict,
    create_knowledge_document,
    create_uploaded_knowledge_document,
    delete_knowledge_document,
    get_knowledge_document,
    get_reindex_job,
    list_knowledge_documents,
    reindex_knowledge_documents,
    retry_knowledge_document_index,
    update_knowledge_document,
)


router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# 创建一篇知识库文档。
@router.post("/documents", response_model=KnowledgeDocumentRead, status_code=status.HTTP_201_CREATED)
async def create_document(
    payload: KnowledgeDocumentCreate,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentRead:
    try:
        document = await create_knowledge_document(db, payload)
    except KnowledgeDataQualityError as exc:
        await db.commit()
        raise _quality_http_error(exc) from exc
    except KnowledgeIndexingError as exc:
        await db.commit()
        raise _indexing_http_error(exc) from exc
    await db.commit()
    return document


# 获取全部知识库文档。
@router.get("/documents", response_model=list[KnowledgeDocumentRead])
async def list_documents(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> list[KnowledgeDocumentRead]:
    return await list_knowledge_documents(db)


# 获取指定知识库文档。
@router.get("/documents/{document_id}", response_model=KnowledgeDocumentRead)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentRead:
    document = await get_knowledge_document(db, document_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return document


# 更新指定知识库文档。
@router.put("/documents/{document_id}", response_model=KnowledgeDocumentRead)
async def update_document(
    document_id: int,
    payload: KnowledgeDocumentUpdate,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentRead:
    try:
        document = await update_knowledge_document(db, document_id, payload)
    except KnowledgeDataQualityError as exc:
        await db.commit()
        raise _quality_http_error(exc) from exc
    except KnowledgeIndexingError as exc:
        await db.commit()
        raise _indexing_http_error(exc) from exc
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await db.commit()
    return document


# 删除指定知识库文档。
@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> None:
    try:
        deleted = await delete_knowledge_document(db, document_id)
    except KnowledgeIndexingError as exc:
        await db.commit()
        raise _indexing_http_error(exc) from exc
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await db.commit()
    return None


# 解析上传文件并创建知识库文档。
@router.post("/files", response_model=KnowledgeDocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    category: str = Form(...),
    target_position: str | None = Form(default=None),
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentRead:
    parsed_upload = await build_document_from_upload(file, title, category, target_position)
    try:
        document = await create_uploaded_knowledge_document(db, parsed_upload)
    except KnowledgeDataQualityError as exc:
        await db.commit()
        raise _quality_http_error(exc) from exc
    except KnowledgeIndexingError as exc:
        await db.commit()
        raise _indexing_http_error(exc) from exc
    await db.commit()
    return document



@router.post("/documents/{document_id}/retry-index", response_model=KnowledgeDocumentRead)
async def retry_document_index(
    document_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeDocumentRead:
    try:
        document = await retry_knowledge_document_index(db, document_id)
    except KnowledgeDataQualityError as exc:
        await db.commit()
        raise _quality_http_error(exc) from exc
    except KnowledgeIndexingError as exc:
        await db.commit()
        raise _indexing_http_error(exc) from exc
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    await db.commit()
    return document

# 重建知识库文档索引。
@router.post("/reindex", response_model=KnowledgeReindexResult)
async def reindex_knowledge(
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeReindexResult:
    try:
        result = await reindex_knowledge_documents(db)
    except KnowledgeReindexConflict as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    await db.commit()
    return result


@router.get("/reindex/{job_id}", response_model=KnowledgeReindexResult)
async def get_knowledge_reindex_job(
    job_id: int,
    current_user: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeReindexResult:
    result = await get_reindex_job(db, job_id)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Reindex job not found")
    return result


def _indexing_http_error(exc: KnowledgeIndexingError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Document {exc.document_id} was saved but indexing failed; retry indexing later",
    )

def _quality_http_error(exc: KnowledgeDataQualityError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=f"Document {exc.document_id} failed quality checks: {exc.reason}",
    )
