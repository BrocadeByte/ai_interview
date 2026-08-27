# Linux 部署说明

Linux Docker 部署流程已合并到项目根目录的 [README.md](README.md#docker-一键部署)。

当前 `compose.yaml` 已包含 MySQL、RabbitMQ、Qdrant、FastAPI、两个异步 Worker 和前端 Nginx，不再需要手动创建外部 Docker 网络。
