# AI 模拟面试系统

这是根据项目计划搭建的 AI 模拟面试系统 MVP：注册登录、求职画像、文字面试、AI 出题/追问/评分、面试报告、知识库 RAG 和中期记忆压缩。

## 目录

```text
backend/   FastAPI 后端
frontend/  Vue 3 + Vite 前端
```

## 环境依赖

- Python 3.11+
- Node.js 18+
- MySQL 8.x
- Qdrant，用于知识库向量检索
- 可用的大模型 API Key，默认兼容 DeepSeek/OpenAI 风格接口

## 后端配置

后端读取 `backend/.env`。当前项目使用 MySQL，不使用 SQLite。

示例配置：

```env
DATABASE_URL=mysql+aiomysql://root:123456@127.0.0.1:3306/ai_interview?charset=utf8mb4
OPENAI_API_KEY=你的模型 API Key
OPENAI_API_BASE=https://api.deepseek.com
OPENAI_MODEL=deepseek-chat
QDRANT_URL=http://127.0.0.1:6333
QDRANT_COLLECTION_NAME=knowledge_documents
EMBEDDING_PROVIDER=dashscope
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIM=1024
EMBEDDING_SCORE_THRESHOLD=0.35
EMBEDDING_TIMEOUT_SECONDS=30
DASHSCOPE_API_KEY=your DashScope API key
```

启动后端前请先创建 MySQL 数据库：

```sql
CREATE DATABASE ai_interview DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

Production uses DashScope semantic embeddings by default. Hash vectors are test-only and require explicit opt-in:

```env
EMBEDDING_PROVIDER=hash
ALLOW_HASH_EMBEDDINGS=true
```

When the provider, model, or vector dimension changes, rebuild the Qdrant collection:

```text
POST /api/knowledge/reindex
```

Run the built-in semantic retrieval evaluation with:

```bash
cd backend
python -m scripts.evaluate_semantic_embeddings
```
应用启动时会通过 SQLAlchemy 自动创建当前模型对应的数据表。开发期已有少量兼容字段补齐逻辑；后续正式维护建议接入 Alembic。

## 后端启动

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

后端接口地址为 `http://127.0.0.1:8000`。

健康检查：

```text
GET http://127.0.0.1:8000/api/health
```

## 前端启动

```bash
cd frontend
npm install
npm run dev
```

前端默认运行在 `http://127.0.0.1:5173`，并通过 Vite 代理访问 `/api`。

## 已实现接口

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/profile/me`
- `PUT /api/profile/me`
- `POST /api/interviews`
- `GET /api/interviews`
- `GET /api/interviews/{session_id}`
- `POST /api/interviews/{session_id}/answer`
- `POST /api/interviews/{session_id}/finish`
- `GET /api/interviews/{session_id}/scores`
- `GET /api/interviews/{session_id}/report`
- `GET /api/reports`
- `GET /api/reports/{report_id}`
- `POST /api/knowledge/documents`
- `GET /api/knowledge/documents`
- `GET /api/knowledge/documents/{document_id}`
- `POST /api/knowledge/reindex`

## 当前状态

项目主链路已经基本跑通，但还在 MVP 收尾阶段。待完成和缺陷记录见：

```text
待完成与缺陷清单.md
```
