import asyncio
import time

from app.rag.embeddings import close_embedding_session, embed_text
from app.rag.retriever import close_retriever_client, search_chunks_advanced


async def verify_non_blocking(operation, name: str) -> None:
    heartbeat_at: float | None = None
    operation_finished_at: float | None = None
    started_at = time.perf_counter()

    async def heartbeat() -> None:
        nonlocal heartbeat_at
        await asyncio.sleep(0.01)
        heartbeat_at = time.perf_counter()

    async def measured_operation():
        nonlocal operation_finished_at
        result = await operation()
        operation_finished_at = time.perf_counter()
        return result

    result, _ = await asyncio.gather(measured_operation(), heartbeat())
    if heartbeat_at is None or operation_finished_at is None:
        raise RuntimeError(f"{name}: timing markers were not recorded")
    if heartbeat_at >= operation_finished_at:
        raise RuntimeError(f"{name}: event-loop heartbeat ran only after network operation completed")
    print(
        {
            "operation": name,
            "heartbeat_ms": round((heartbeat_at - started_at) * 1000, 2),
            "operation_ms": round((operation_finished_at - started_at) * 1000, 2),
            "result_size": len(result),
            "non_blocking": True,
        }
    )


async def main() -> None:
    await verify_non_blocking(
        lambda: embed_text("如何避免接口重复提交"),
        "dashscope_embedding",
    )
    await verify_non_blocking(
        lambda: search_chunks_advanced("求职基础信息", limit=3),
        "qdrant_semantic_search",
    )
    await close_retriever_client()
    await close_embedding_session()


if __name__ == "__main__":
    asyncio.run(main())
