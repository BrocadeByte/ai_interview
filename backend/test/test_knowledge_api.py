import sys
from pathlib import Path
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401
import app.services.knowledge_service as knowledge_service
from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app
from app.models.knowledge import KnowledgeDocument


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def test_session_factory():
    source_url = make_url(settings.database_url)
    test_database = f"ai_interview_test_{uuid4().hex[:12]}"
    admin_url = source_url.set(database=None)
    test_url = source_url.set(database=test_database)

    admin_engine = create_async_engine(admin_url)
    async with admin_engine.begin() as conn:
        await conn.execute(text(f"CREATE DATABASE `{test_database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
    await admin_engine.dispose()

    test_engine = create_async_engine(test_url)
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(bind=test_engine, expire_on_commit=False)
    try:
        yield session_factory
    finally:
        await test_engine.dispose()
        cleanup_engine = create_async_engine(admin_url)
        async with cleanup_engine.begin() as conn:
            await conn.execute(text(f"DROP DATABASE IF EXISTS `{test_database}`"))
        await cleanup_engine.dispose()


@pytest.fixture
async def client(test_session_factory, monkeypatch):
    async def override_get_db():
        async with test_session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    fake_uploads: list[tuple[str, bytes]] = []
    index_control = {"fail": False, "fail_delete": False, "fail_rebuild": False}
    active = {"collection": "knowledge_documents_live"}
    collections: dict[str, dict[str, dict]] = {active["collection"]: {}}
    calls: list[tuple] = []

    class FakeOssObjectStorage:
        async def upload_bytes(self, data: bytes, key: str) -> str:
            fake_uploads.append((key, data))
            return key

        async def download_bytes(self, key: str) -> bytes:
            for stored_key, data in fake_uploads:
                if stored_key == key:
                    return data
            return b""

    async def fake_upsert_chunks(chunks: list[dict], *, collection_name: str | None = None) -> None:
        target = collection_name or active["collection"]
        if index_control["fail"] or (index_control["fail_rebuild"] and collection_name is not None):
            raise RuntimeError("simulated qdrant outage")
        points = collections.setdefault(target, {})
        for chunk in chunks:
            points[str(chunk["point_id"])] = dict(chunk)
        calls.append(("upsert", target, len(chunks)))

    async def fake_replace_document_chunks(document_id: int, index_version: int, chunks: list[dict]) -> None:
        if index_control["fail"]:
            raise RuntimeError("simulated qdrant outage")
        points = collections[active["collection"]]
        for point_id in [key for key, value in points.items() if value["document_id"] == document_id]:
            del points[point_id]
        for chunk in chunks:
            points[str(chunk["point_id"])] = dict(chunk)
        calls.append(("replace", document_id, index_version))

    async def fake_delete_document_chunks(document_id: int, *, collection_name: str | None = None) -> None:
        if index_control["fail_delete"]:
            raise RuntimeError("simulated qdrant delete outage")
        target = collection_name or active["collection"]
        points = collections[target]
        for point_id in [key for key, value in points.items() if value["document_id"] == document_id]:
            del points[point_id]
        calls.append(("delete_document", document_id))

    async def fake_create_collection(collection_name: str) -> None:
        collections[collection_name] = {}
        calls.append(("create_collection", collection_name))

    async def fake_delete_collection(collection_name: str) -> None:
        collections.pop(collection_name, None)
        calls.append(("delete_collection", collection_name))

    async def fake_get_active_collection_target() -> str:
        return active["collection"]

    async def fake_switch_active_collection(collection_name: str) -> str:
        previous = active["collection"]
        active["collection"] = collection_name
        calls.append(("switch", previous, collection_name))
        return previous

    monkeypatch.setattr(knowledge_service, "upsert_chunks", fake_upsert_chunks)
    monkeypatch.setattr(knowledge_service, "replace_document_chunks", fake_replace_document_chunks)
    monkeypatch.setattr(knowledge_service, "delete_document_chunks", fake_delete_document_chunks)
    monkeypatch.setattr(knowledge_service, "create_collection", fake_create_collection)
    monkeypatch.setattr(knowledge_service, "delete_collection", fake_delete_collection)
    monkeypatch.setattr(knowledge_service, "get_active_collection_target", fake_get_active_collection_target)
    monkeypatch.setattr(knowledge_service, "switch_active_collection", fake_switch_active_collection)
    monkeypatch.setattr(knowledge_service, "OssObjectStorage", FakeOssObjectStorage)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as async_client:
        async_client.fake_uploads = fake_uploads
        async_client.test_session_factory = test_session_factory
        async_client.index_control = index_control
        async_client.index_collections = collections
        async_client.index_active = active
        async_client.index_calls = calls
        yield async_client

    app.dependency_overrides.clear()

@pytest.mark.anyio
async def test_mojibake_document_is_failed_and_never_indexed(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    mojibake = "鏂囨。涓湁澶ч噺涔辩爜 " * 20

    response = await client.post(
        "/api/knowledge/documents",
        headers=headers,
        json={
            "title": "Garbled document",
            "category": "scoring",
            "target_position": "Python Backend Engineer",
            "content": mojibake,
            "metadata": {"source": "quality-test"},
        },
    )

    assert response.status_code == 422
    documents = (await client.get("/api/knowledge/documents", headers=headers)).json()
    failed = next(document for document in documents if document["title"] == "Garbled document")
    assert failed["metadata"]["parse_status"] == "failed"
    assert failed["metadata"]["quality_report"]["accepted"] is False
    assert not any(
        chunk["document_id"] == failed["id"]
        for chunk in client.index_collections[client.index_active["collection"]].values()
    )


@pytest.mark.anyio
async def test_bad_update_keeps_previous_ready_version(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    created = await create_document(client, headers, title="Stable quality", content="valid JWT scoring content")

    response = await client.put(
        f"/api/knowledge/documents/{created['id']}",
        headers=headers,
        json={
            "title": "Rejected garbled update",
            "category": "scoring",
            "target_position": "Python Backend Engineer",
            "content": "锟斤拷鏂囨。涓湁涔辩爜 " * 20,
            "metadata": {"version": 2},
        },
    )

    assert response.status_code == 422
    detail = (await client.get(f"/api/knowledge/documents/{created['id']}", headers=headers)).json()
    assert detail["title"] == "Stable quality"
    assert detail["metadata"]["parse_status"] == "ready"
    active_points = client.index_collections[client.index_active["collection"]].values()
    assert any(chunk["document_id"] == created["id"] for chunk in active_points)

@pytest.mark.anyio
async def test_update_knowledge_document_is_incremental_and_persists(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    created = await create_document(client, headers, title="Original title", content="old FastAPI content")

    response = await client.put(
        f"/api/knowledge/documents/{created['id']}",
        headers=headers,
        json={
            "title": "Updated title",
            "category": "updated-category",
            "target_position": "Python Backend Engineer",
            "content": "updated JWT content",
            "metadata": {"source": "api-test", "version": 2},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == created["id"]
    assert data["title"] == "Updated title"
    assert data["content"] == "updated JWT content"
    assert data["metadata"]["source"] == "api-test"
    assert data["metadata"]["version"] == 2
    assert data["metadata"]["chunk_count"] == 1
    assert data["metadata"]["parse_status"] == "ready"
    assert data["metadata"]["index_version"] == 2
    assert ("replace", created["id"], 2) in client.index_calls
    active_points = client.index_collections[client.index_active["collection"]].values()
    assert any(chunk["title"] == "Updated title" for chunk in active_points)
    assert not any(chunk["title"] == "Original title" for chunk in active_points)
    assert not any(call[0] == "create_collection" for call in client.index_calls)

    detail = await client.get(f"/api/knowledge/documents/{created['id']}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["title"] == "Updated title"


@pytest.mark.anyio
async def test_delete_knowledge_document_is_incremental_and_keeps_other_document(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    first = await create_document(client, headers, title="Delete me", content="temporary content")
    second = await create_document(client, headers, title="Keep me", content="remaining content")
    client.index_calls.clear()

    response = await client.delete(f"/api/knowledge/documents/{first['id']}", headers=headers)

    assert response.status_code == 204
    assert ("delete_document", first["id"]) in client.index_calls
    active_points = client.index_collections[client.index_active["collection"]].values()
    assert not any(chunk["document_id"] == first["id"] for chunk in active_points)
    assert any(chunk["document_id"] == second["id"] for chunk in active_points)
    assert not any(call[0] == "create_collection" for call in client.index_calls)

    detail = await client.get(f"/api/knowledge/documents/{first['id']}", headers=headers)
    assert detail.status_code == 404


@pytest.mark.anyio
async def test_upload_txt_and_md_files_create_knowledge_documents(client: AsyncClient) -> None:
    headers = await register_admin_user(client)

    txt_response = await client.post(
        "/api/knowledge/files",
        headers=headers,
        data={"category": "upload-test", "target_position": "Python Backend Engineer"},
        files={"file": ("auth.txt", b"JWT token expiration and password hashing", "text/plain")},
    )
    md_response = await client.post(
        "/api/knowledge/files",
        headers=headers,
        data={"title": "Markdown Doc", "category": "upload-test", "target_position": "Frontend Engineer"},
        files={"file": ("frontend.md", b"# Vue Interview\n\nComponent state and routing.", "text/markdown")},
    )

    assert txt_response.status_code == 201
    assert txt_response.json()["title"] == "auth"
    assert txt_response.json()["metadata"]["file_type"] == "txt"
    assert "JWT token" in txt_response.json()["content"]
    assert md_response.status_code == 201
    assert md_response.json()["title"] == "Markdown Doc"
    assert md_response.json()["metadata"]["file_type"] == "md"
    assert txt_response.json()["metadata"]["original_oss_key"]
    assert txt_response.json()["metadata"]["parsed_text_oss_key"]
    assert len(client.fake_uploads) == 4
    active_points = client.index_collections[client.index_active["collection"]].values()
    assert any(chunk["title"] == "auth" for chunk in active_points)
    assert any(chunk["title"] == "Markdown Doc" for chunk in active_points)


@pytest.mark.anyio
async def test_upload_unsupported_file_returns_400(client: AsyncClient) -> None:
    headers = await register_admin_user(client)

    response = await client.post(
        "/api/knowledge/files",
        headers=headers,
        data={"category": "upload-test", "target_position": "Python Backend Engineer"},
        files={"file": ("resume.docx", b"docx bytes", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert response.status_code == 400



@pytest.mark.anyio
async def test_upload_file_defaults_scope_when_target_position_is_omitted(client: AsyncClient) -> None:
    headers = await register_admin_user(client)

    response = await client.post(
        "/api/knowledge/files",
        headers=headers,
        data={"category": "job-intelligence"},
        files={"file": ("jobs.txt", "BOSS job posting and resume search tips".encode("utf-8"), "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["target_position"] == "general"


@pytest.mark.anyio
async def test_update_and_delete_missing_knowledge_document_return_404(client: AsyncClient) -> None:
    headers = await register_admin_user(client)

    update_response = await client.put(
        "/api/knowledge/documents/999999",
        headers=headers,
        json={
            "title": "Missing",
            "category": "missing",
            "target_position": "Python Backend Engineer",
            "content": "missing content",
            "metadata": {},
        },
    )
    delete_response = await client.delete("/api/knowledge/documents/999999", headers=headers)

    assert update_response.status_code == 404
    assert delete_response.status_code == 404
    assert not client.index_calls


@pytest.mark.anyio
async def test_regular_user_cannot_access_knowledge_management(client: AsyncClient) -> None:
    headers = await register_user(client)

    list_response = await client.get("/api/knowledge/documents", headers=headers)
    create_response = await client.post(
        "/api/knowledge/documents",
        headers=headers,
        json={
            "title": "Forbidden",
            "category": "api-test",
            "target_position": "Python Backend Engineer",
            "content": "regular user should not create knowledge",
            "metadata": {},
        },
    )
    upload_response = await client.post(
        "/api/knowledge/files",
        headers=headers,
        data={"category": "upload-test", "target_position": "Python Backend Engineer"},
        files={"file": ("auth.txt", b"content", "text/plain")},
    )

    assert list_response.status_code == 403
    assert create_response.status_code == 403
    assert upload_response.status_code == 403



@pytest.mark.anyio
async def test_failed_index_is_persisted_and_retry_is_idempotent(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    client.index_control["fail"] = True

    response = await client.post(
        "/api/knowledge/documents",
        headers=headers,
        json={
            "title": "Retry document",
            "category": "rubric",
            "target_position": "Python Backend Engineer",
            "content": "JWT expiration and refresh strategy",
            "metadata": {"source": "failure-test"},
        },
    )

    assert response.status_code == 503
    documents = (await client.get("/api/knowledge/documents", headers=headers)).json()
    failed = next(document for document in documents if document["title"] == "Retry document")
    assert failed["metadata"]["parse_status"] == "failed"
    assert "simulated qdrant outage" in failed["metadata"]["parse_error"]
    assert not any(
        chunk["document_id"] == failed["id"]
        for chunk in client.index_collections[client.index_active["collection"]].values()
    )

    client.index_control["fail"] = False
    first_retry = await client.post(
        f"/api/knowledge/documents/{failed['id']}/retry-index",
        headers=headers,
    )
    second_retry = await client.post(
        f"/api/knowledge/documents/{failed['id']}/retry-index",
        headers=headers,
    )

    assert first_retry.status_code == 200
    assert first_retry.json()["metadata"]["parse_status"] == "ready"
    assert second_retry.status_code == 200
    matching_chunks = [
        chunk
        for chunk in client.index_collections[client.index_active["collection"]].values()
        if chunk["document_id"] == failed["id"]
    ]
    assert len(matching_chunks) == 1

@pytest.mark.anyio
async def test_failed_incremental_update_keeps_previous_document_and_index(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    created = await create_document(client, headers, title="Stable title", content="stable content")
    client.index_control["fail"] = True

    response = await client.put(
        f"/api/knowledge/documents/{created['id']}",
        headers=headers,
        json={
            "title": "Rejected title",
            "category": "scoring",
            "target_position": "Python Backend Engineer",
            "content": "unpublished content",
            "metadata": {"version": 2},
        },
    )

    assert response.status_code == 503
    detail = (await client.get(f"/api/knowledge/documents/{created['id']}", headers=headers)).json()
    assert detail["title"] == "Stable title"
    assert detail["content"] == "stable content"
    active_points = client.index_collections[client.index_active["collection"]].values()
    assert any(chunk["title"] == "Stable title" for chunk in active_points)
    assert not any(chunk["title"] == "Rejected title" for chunk in active_points)


@pytest.mark.anyio
async def test_failed_incremental_delete_keeps_document_and_index(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    created = await create_document(client, headers, title="Keep on failure", content="still searchable")
    client.index_control["fail_delete"] = True

    response = await client.delete(f"/api/knowledge/documents/{created['id']}", headers=headers)

    assert response.status_code == 503
    assert (await client.get(f"/api/knowledge/documents/{created['id']}", headers=headers)).status_code == 200
    active_points = client.index_collections[client.index_active["collection"]].values()
    assert any(chunk["document_id"] == created["id"] for chunk in active_points)


@pytest.mark.anyio
async def test_failed_full_reindex_keeps_active_collection_and_records_recovery(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    await create_document(client, headers, title="Live document", content="live searchable content")
    previous_collection = client.index_active["collection"]
    previous_points = dict(client.index_collections[previous_collection])
    client.index_control["fail_rebuild"] = True

    response = await client.post("/api/knowledge/reindex", headers=headers)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "failed"
    assert result["processed_documents"] == 0
    assert "simulated qdrant outage" in result["error"]
    assert "Active alias remains" in result["recovery_strategy"]
    assert client.index_active["collection"] == previous_collection
    assert client.index_collections[previous_collection] == previous_points
    assert not any(call[0] == "switch" for call in client.index_calls)

    job_response = await client.get(f"/api/knowledge/reindex/{result['job_id']}", headers=headers)
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "failed"


@pytest.mark.anyio
async def test_full_reindex_quarantines_existing_mojibake_document(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    async with client.test_session_factory() as db:
        document = KnowledgeDocument(
            title="Legacy garbled",
            category="scoring",
            target_position="Python Backend Engineer",
            content="\u951f\u65a4\u62f7 broken encoding " * 20,
            metadata_json="{}",
            parse_status="ready",
            index_version=1,
            chunk_count=1,
        )
        db.add(document)
        await db.commit()
        document_id = document.id
    client.index_collections[client.index_active["collection"]][f"legacy-{document_id}"] = {
        "point_id": f"legacy-{document_id}",
        "document_id": document_id,
        "text": "stale garbled chunk",
    }

    response = await client.post("/api/knowledge/reindex", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    detail = (await client.get(f"/api/knowledge/documents/{document_id}", headers=headers)).json()
    assert detail["metadata"]["parse_status"] == "failed"
    assert detail["metadata"]["quality_report"]["accepted"] is False
    assert detail["metadata"]["chunk_count"] == 0
    for points in client.index_collections.values():
        assert not any(chunk.get("document_id") == document_id for chunk in points.values())

@pytest.mark.anyio
async def test_successful_full_reindex_switches_only_after_all_documents_are_written(client: AsyncClient) -> None:
    headers = await register_admin_user(client)
    first = await create_document(client, headers, title="First live", content="first content")
    second = await create_document(client, headers, title="Second live", content="second content")
    previous_collection = client.index_active["collection"]
    client.index_calls.clear()

    response = await client.post("/api/knowledge/reindex", headers=headers)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    assert result["processed_documents"] == result["document_count"]
    assert client.index_active["collection"] == result["target_collection"]
    assert client.index_active["collection"] != previous_collection
    rebuilt_points = client.index_collections[result["target_collection"]].values()
    rebuilt_ids = {chunk["document_id"] for chunk in rebuilt_points}
    assert {first["id"], second["id"]}.issubset(rebuilt_ids)
    assert client.index_calls[-1][0] == "switch"

async def register_user(client: AsyncClient) -> dict[str, str]:
    response = await client.post(
        "/api/auth/register",
        json={
            "email": f"knowledge_{uuid4().hex}@example.com",
            "username": "knowledge-user",
            "password": "Password123",
        },
    )
    assert response.status_code == 201
    return {"Authorization": f"Bearer {response.json()['access_token']}", "X-Test-Email": response.json()["user"]["email"]}


async def register_admin_user(client: AsyncClient) -> dict[str, str]:
    headers = await register_user(client)
    email = headers["X-Test-Email"]
    async with client.test_session_factory() as db:
        await db.execute(text("UPDATE users SET is_admin = 1 WHERE email = :email"), {"email": email})
        await db.commit()
    login_response = await client.post("/api/auth/login", json={"email": email, "password": "Password123"})
    assert login_response.status_code == 200
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


async def create_document(client: AsyncClient, headers: dict[str, str], title: str, content: str) -> dict:
    response = await client.post(
        "/api/knowledge/documents",
        headers=headers,
        json={
            "title": title,
            "category": "api-test",
            "target_position": "Python Backend Engineer",
            "content": content,
            "metadata": {"source": "api-test"},
        },
    )
    assert response.status_code == 201
    return response.json()
