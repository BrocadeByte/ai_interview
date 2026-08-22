import asyncio
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401
from app.core.database import SessionLocal
from app.services.knowledge_cleanup_service import delete_knowledge_documents_by_marker
from app.services.knowledge_service import reindex_knowledge_documents


TEST_MARKER_PREFIX = "RAG_PROMPT_INJECTION_MARKER_"


async def cleanup_test_knowledge(marker: str = TEST_MARKER_PREFIX) -> None:
    async with SessionLocal() as db:
        deleted_count = await delete_knowledge_documents_by_marker(db, marker)
        await db.commit()
        result = await reindex_knowledge_documents(db)
        await db.commit()

    print({"deleted_count": deleted_count, "reindex": result.model_dump()})


if __name__ == "__main__":
    marker_arg = sys.argv[1] if len(sys.argv) > 1 else TEST_MARKER_PREFIX
    asyncio.run(cleanup_test_knowledge(marker_arg))
