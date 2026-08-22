from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

import alibabacloud_oss_v2 as oss
import alibabacloud_oss_v2.aio as oss_aio
from fastapi import HTTPException, status

from app.core.config import settings


class OssObjectStorage:
    def __init__(self) -> None:
        if not settings.aliyun_oss_region or not settings.aliyun_oss_bucket:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Aliyun OSS is not configured. Set ALIYUN_OSS_REGION and ALIYUN_OSS_BUCKET.",
            )

    async def upload_bytes(self, data: bytes, key: str) -> str:
        client = self._create_client()
        try:
            await client.put_object(
                oss.PutObjectRequest(
                    bucket=settings.aliyun_oss_bucket,
                    key=key,
                    body=data,
                )
            )
            return key
        finally:
            await client.close()

    async def download_bytes(self, key: str) -> bytes:
        client = self._create_client()
        try:
            result = await client.get_object(oss.GetObjectRequest(bucket=settings.aliyun_oss_bucket, key=key))
            body = result.body
            if hasattr(body, "read"):
                data = body.read()
                if hasattr(data, "__await__"):
                    data = await data
                return data
            raise RuntimeError("OSS response body is not readable")
        finally:
            await client.close()

    def _create_client(self):
        credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
        cfg = oss.config.load_default()
        cfg.credentials_provider = credentials_provider
        cfg.region = settings.aliyun_oss_region
        if settings.aliyun_oss_endpoint:
            cfg.endpoint = settings.aliyun_oss_endpoint
        return oss_aio.AsyncClient(cfg)


def build_knowledge_oss_key(kind: str, filename: str) -> str:
    now = datetime.now(UTC)
    safe_filename = _safe_filename(filename)
    prefix = settings.aliyun_oss_prefix.rstrip("/") or "knowledge"
    return f"{prefix}/{kind}/{now:%Y/%m}/{uuid4().hex}-{safe_filename}"


def _safe_filename(filename: str) -> str:
    name = Path(filename or "uploaded").name.replace("\\", "_").replace("/", "_")
    return name or "uploaded"
