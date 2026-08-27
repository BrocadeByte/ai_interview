# AI 模拟面试官

一个面向技术求职者的 AI 面试训练系统。用户可以导入简历和职位描述，由 AI 生成针对性的面试计划、动态追问和评分报告，并通过专项练习持续改善薄弱项。

## 主要功能

- 账号注册、登录与个人求职画像
- 简历上传、AI 解析与结构化管理
- 职位描述（JD）解析及岗位能力提取
- 按岗位、难度和面试类型生成个性化面试计划
- 多轮问答、动态追问、逐题评分与改进建议
- 面试报告、能力维度分析、薄弱项专项练习
- 基于 Qdrant 的知识库检索、引用溯源和语义重排
- 用户行为与业务漏斗分析

## 技术架构

| 模块 | 技术 |
| --- | --- |
| Web 前端 | Vue 3、TypeScript、Vite、Element Plus、Pinia |
| API 后端 | FastAPI、SQLAlchemy、Pydantic |
| 数据库 | MySQL 8.4 |
| 消息队列 | RabbitMQ 4 |
| 向量数据库 | Qdrant |
| AI 接口 | OpenAI 兼容接口；默认示例为 DeepSeek |
| Embedding / Rerank | DashScope text-embedding-v4、gte-rerank-v2 |
| 生产入口 | Nginx + Docker Compose |

项目目录：

```text
backend/       FastAPI API、业务服务和异步 Worker
frontend/      Vue 单页应用与 Nginx 配置
compose.yaml   完整的容器化部署编排
```

## Docker 一键部署

### 1. 准备服务器

推荐使用 Linux 服务器，并安装 Docker Engine 与 Docker Compose 插件：

```bash
docker --version
docker compose version
```

生产环境还需要准备：

- 一个可用的 OpenAI 兼容大模型 API Key
- 使用知识库语义检索时所需的 DashScope API Key
- 使用知识文件上传时所需的阿里云 OSS 配置（可选）

### 2. 配置环境变量

在项目根目录执行：

```bash
cp .env.production.example .env.production
openssl rand -hex 32
openssl rand -hex 32
```

编辑 `.env.production`，至少替换以下配置：

```env
MYSQL_PASSWORD=数据库用户密码
MYSQL_ROOT_PASSWORD=数据库 root 密码
RABBITMQ_PASSWORD=消息队列密码
JWT_SECRET_KEY=第一段随机密钥
REFRESH_JWT_SECRET_KEY=第二段独立随机密钥
OPENAI_API_KEY=大模型 API Key
DASHSCOPE_API_KEY=DashScope API Key
```

密码会进入连接 URL，请使用字母、数字、下划线、短横线或点等 URL 安全字符。不要提交 `.env.production`。

### 3. 构建并启动

```bash
docker compose --env-file .env.production up -d --build
docker compose --env-file .env.production ps
```

Compose 会自动启动：

- `frontend`：Nginx 与前端静态页面，对外提供访问入口
- `backend`：FastAPI 接口服务
- `resume-worker`：异步解析简历
- `knowledge-worker`：异步处理知识库文件
- `mysql`：业务数据库
- `rabbitmq`：任务消息队列
- `qdrant`：向量数据库

默认访问地址：

- 应用首页：`http://服务器地址/`
- API 文档：`http://服务器地址/docs`
- 前端健康检查：`http://服务器地址/health`
- 后端就绪检查：`http://服务器地址/api/health/ready`
- RabbitMQ 管理页：仅服务器本机 `http://127.0.0.1:15672`

如果 80 端口已占用，在 `.env.production` 中修改：

```env
HTTP_PORT=8080
```

### 4. 查看状态和日志

```bash
docker compose --env-file .env.production ps
docker compose --env-file .env.production logs -f backend
docker compose --env-file .env.production logs -f resume-worker knowledge-worker
```

所有容器均启用了日志轮转，单个日志文件最多 10 MB，每个服务最多保留 3 个文件。

### 5. 更新版本

```bash
git pull
docker compose --env-file .env.production up -d --build
docker compose --env-file .env.production ps
```

### 6. 停止或卸载

停止服务但保留数据：

```bash
docker compose --env-file .env.production down
```

MySQL、RabbitMQ 和 Qdrant 数据分别保存在 Docker 命名卷中。不要执行下面的命令，除非确认要永久删除全部业务数据：

```bash
docker compose --env-file .env.production down -v
```

## 数据备份

备份 MySQL：

```bash
docker compose --env-file .env.production exec -T mysql \
  sh -c 'exec mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE"' \
  > ai_interview_$(date +%F).sql
```

Qdrant 中保存的是知识库向量索引，重要环境应同时备份 `qdrant_data` 卷或使用 Qdrant Snapshot。知识库原文件使用 OSS 时，还需要单独保障 OSS 数据安全。

## HTTPS 部署

公网环境建议在本项目之前配置 Caddy、Traefik 或宿主机 Nginx，并只让反向代理访问应用端口：

```env
HTTP_BIND=127.0.0.1
HTTP_PORT=8080
AUTH_COOKIE_SECURE=true
```

随后将域名反向代理到 `http://127.0.0.1:8080`，并配置有效的 TLS 证书。

## 本地开发

### 后端

准备 Python 3.12、MySQL、RabbitMQ 和 Qdrant，并在 `backend/.env` 中填写对应连接信息：

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

另开终端启动两个异步 Worker：

```powershell
cd backend
.venv\Scripts\Activate.ps1
python -m app.workers.resume_worker
```

```powershell
cd backend
.venv\Scripts\Activate.ps1
python -m app.workers.knowledge_worker
```

后端默认地址为 `http://127.0.0.1:8000`。

### 前端

准备 Node.js 22：

```powershell
cd frontend
npm install
npm run dev
```

前端默认地址为 `http://127.0.0.1:5173`，开发服务器会将 `/api` 代理到后端。

## 健康检查

| 地址 | 含义 |
| --- | --- |
| `/health` | Nginx / 前端容器存活 |
| `/api/health/live` | FastAPI 进程存活 |
| `/api/health/ready` | MySQL、RabbitMQ、Qdrant 均可访问 |
| `/api/health` | 兼容旧部署，等同于 ready |

## 注意事项

- 后端当前会在启动时自动创建表并补齐部分字段；正式多实例部署前建议迁移到 Alembic。
- 更改 Embedding 模型或向量维度后，需要通过知识库重建接口重新生成 Qdrant 索引。
- `EMBEDDING_PROVIDER=hash` 仅用于测试，并且必须同时设置 `ALLOW_HASH_EMBEDDINGS=true`。
- 公网部署时不要暴露 MySQL、RabbitMQ AMQP 或 Qdrant 端口。
