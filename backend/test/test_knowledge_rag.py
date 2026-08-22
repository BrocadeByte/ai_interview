import asyncio
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sqlalchemy import select

from app.core.database import Base, SessionLocal, engine
from app.models.knowledge import KnowledgeDocument
from app.schemas.knowledge import KnowledgeDocumentCreate
from app.services.knowledge_service import create_knowledge_document, format_knowledge_context, list_knowledge_documents


SEED_DOCUMENTS = [
    KnowledgeDocumentCreate(
        title="Python 后端岗位能力模型",
        category="岗位能力",
        target_position="Python 后端工程师",
        content="""
一、核心能力
- 熟悉 Python 基础语法、数据结构、面向对象和常用标准库。
- 熟悉 FastAPI / Flask 等 Web 框架，理解路由、依赖注入、请求响应、异常处理。
- 熟悉 MySQL 设计、索引、事务、SQL 优化。
- 熟悉 Redis、缓存、限流、会话管理。

二、常见追问
- 登录鉴权如何做？密码如何存储？Token 如何失效？
- 项目中如何处理性能瓶颈？
- 线上问题如何定位、排查、回滚和复盘？

三、评分要点
- 是否说清楚做了什么。
- 是否说清楚为什么这样做。
- 是否说清楚遇到了什么问题、如何分析、如何解决。
- 是否能结合项目结果说明价值。
""".strip(),
        metadata={"source": "seed", "version": 1},
    ),
    KnowledgeDocumentCreate(
        title="JWT 与登录安全评分标准",
        category="评分标准",
        target_position="Python 后端工程师",
        content="""
一、专业准确性
- 是否正确描述 bcrypt 哈希密码。
- 是否说明 JWT 签发、过期、刷新、登出失效。

二、表达清晰度
- 是否按流程描述注册、登录、鉴权、失效处理。
- 是否有背景、方案、结果的结构。

三、安全性
- 是否提到盐值、密钥管理、HttpOnly Cookie、CSRF、防暴力破解。

四、优秀回答特征
- 能说清楚为什么选 bcrypt。
- 能说清楚为什么选 JWT 而不是 session。
- 能说清楚 token 过期和刷新策略。
""".strip(),
        metadata={"source": "seed", "version": 1},
    ),
    KnowledgeDocumentCreate(
        title="FastAPI 项目追问题库",
        category="面试题库",
        target_position="Python 后端工程师",
        content="""
一、项目追问方向
- 项目背景是什么？你负责哪一部分？
- 为什么这么设计接口和表结构？
- 有没有遇到线上异常？怎么解决？
- 有没有做过性能优化？怎么验证效果？

二、优秀回答样例要点
- 不要只列技术名词。
- 要说明具体场景、关键决策、结果。
- 最好带上指标、耗时、并发量、提升百分比。
""".strip(),
        metadata={"source": "seed", "version": 1},
    ),
    KnowledgeDocumentCreate(
        title="面试报告优秀回答样例",
        category="优秀回答样例",
        target_position="Python 后端工程师",
        content="""
示范回答结构：
1. 背景：说明场景和约束。
2. 方案：说明你选了什么方案。
3. 实现：说明你怎么做。
4. 结果：说明带来了什么结果。
5. 复盘：说明如果重做你会如何优化。

示范要求：
- 语言清楚。
- 有过程。
- 有结果。
- 有反思。
""".strip(),
        metadata={"source": "seed", "version": 1},
    ),
]


async def seed_documents() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with SessionLocal() as db:
        existing_titles = set(await db.scalars(select(KnowledgeDocument.title)))
        created = []
        for document in SEED_DOCUMENTS:
            if document.title in existing_titles:
                continue
            created.append(await create_knowledge_document(db, document))
        await db.commit()

        print("seeded:", [doc.title for doc in created])
        print("all docs:")
        for doc in await list_knowledge_documents(db):
            print(f"- {doc.id}: {doc.title} / {doc.category} / {doc.target_position}")

    print("\nquery test:")
    print(await format_knowledge_context("JWT 登录安全 过期 刷新 防暴力破解", limit=3))


if __name__ == "__main__":
    asyncio.run(seed_documents())
