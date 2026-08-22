import asyncio
import sys
from datetime import datetime, UTC
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import alibabacloud_oss_v2 as oss
import alibabacloud_oss_v2.aio as oss_aio

from app.core.config import settings


async def test_oss_upload() -> None:
    if not settings.aliyun_oss_region:
        raise SystemExit("ALIYUN_OSS_REGION is missing")
    if not settings.aliyun_oss_bucket:
        raise SystemExit("ALIYUN_OSS_BUCKET is missing")

    credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
    cfg = oss.config.load_default()
    cfg.credentials_provider = credentials_provider
    cfg.region = settings.aliyun_oss_region
    if settings.aliyun_oss_endpoint:
        cfg.endpoint = settings.aliyun_oss_endpoint

    client = oss_aio.AsyncClient(cfg)
    key = f"{settings.aliyun_oss_prefix.rstrip('/')}/healthcheck/codex-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}.txt"
    try:
        result = await client.put_object(
            oss.PutObjectRequest(
                bucket=settings.aliyun_oss_bucket,
                key=key,
                body=b"AI interview OSS healthcheck\n",
            )
        )
        print({"status_code": result.status_code, "request_id": result.request_id, "etag": result.etag, "key": key})
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_oss_upload())
