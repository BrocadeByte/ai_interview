# Linux Docker 部署与运维命令

本文是 AI 模拟面试系统在 Linux 虚拟机上的日常操作手册。服务器地址为 `192.168.150.101`，项目目录为：

```text
/root/ai面试官
```

当前 Compose 管理：

```text
mysql
backend
frontend
```

Qdrant 使用服务器上已有的独立容器 `qdrant`。

## 1. 所有操作的起点

先进入项目根目录：

```bash
cd "/root/ai面试官"
```

确认核心文件存在：

```bash
ls -la compose.yaml .env.production
```

检查 Compose 配置语法：

```bash
docker compose --env-file .env.production config --quiet
```

命令没有输出且正常返回，表示配置有效。

查看 Compose 管理的服务：

```bash
docker compose --env-file .env.production config --services
```

应显示：

```text
mysql
backend
frontend
```

## 2. 首次部署启动

### 2.1 创建共享网络

```bash
docker network inspect ai-interview-network >/dev/null 2>&1 || \
  docker network create ai-interview-network
```

### 2.2 接入已有 Qdrant

```bash
docker network inspect ai-interview-network \
  --format '{{json .Containers}}' | grep -q '"Name":"qdrant"' || \
  docker network connect ai-interview-network qdrant
```

检查网络成员：

```bash
docker network inspect ai-interview-network \
  --format '{{range .Containers}}{{println .Name}}{{end}}'
```

输出中应包含：

```text
qdrant
```

### 2.3 构建并启动全部服务

```bash
docker compose --env-file .env.production up -d --build
```

查看状态：

```bash
docker compose --env-file .env.production ps
```

首次启动 MySQL 需要初始化，等待约 30 至 90 秒。持续观察：

```bash
watch -n 2 'docker compose --env-file .env.production ps'
```

按 `Ctrl+C` 退出观察。

正常状态：

```text
mysql     healthy
backend   healthy
frontend  healthy
```

## 3. 日常启动

容器已经创建，只需要启动：

```bash
cd "/root/ai面试官"
docker compose --env-file .env.production start
```

也可以使用：

```bash
docker compose --env-file .env.production up -d
```

`up -d` 会确保缺失的容器被创建，更适合作为日常通用启动命令。

启动后检查：

```bash
docker compose --env-file .env.production ps
curl -fsS http://127.0.0.1/health
curl -fsS http://127.0.0.1/api/health
```

网站入口：

```text
http://192.168.150.101/
```

## 4. 停止服务

### 4.1 停止但保留容器

```bash
docker compose --env-file .env.production stop
```

后续使用 `start` 即可恢复。

### 4.2 停止并删除应用容器

```bash
docker compose --env-file .env.production down
```

该命令会删除 Compose 创建的容器和临时资源，但保留 MySQL 的命名数据卷。

已有的独立 Qdrant 容器不归本项目 Compose 管理，不会被 `down` 删除。

### 4.3 禁止误用的命令

除非明确要删除全部 MySQL 数据，否则不要执行：

```bash
docker compose --env-file .env.production down -v
```

其中 `-v` 会删除 `mysql_data` 数据卷。

## 5. 重启服务

重启全部现有容器：

```bash
docker compose --env-file .env.production restart
```

只重启后端：

```bash
docker compose --env-file .env.production restart backend
```

只重启前端 Nginx：

```bash
docker compose --env-file .env.production restart frontend
```

只重启 MySQL：

```bash
docker compose --env-file .env.production restart mysql
```

注意：`restart` 只重启旧容器，不会加载新源码、不会更新镜像，也不会重新注入修改后的 Compose 环境变量。

## 6. 修改配置后的更新命令

### 6.1 只修改 `.env.production`

例如修改 API Key、模型名或 Token 有效期，不需要构建镜像，但要重新创建后端容器：

```bash
docker compose --env-file .env.production up -d \
  --force-recreate --no-deps backend
```

确认状态：

```bash
docker compose --env-file .env.production ps backend
```

### 6.2 修改前端 Vue 源码

```bash
docker compose --env-file .env.production build --no-cache frontend

docker compose --env-file .env.production up -d \
  --force-recreate --no-deps frontend
```

检查前端产物哈希：

```bash
docker compose --env-file .env.production exec frontend \
  grep -oE 'index-[A-Za-z0-9_-]+\.js' \
  /usr/share/nginx/html/index.html
```

前端更新后，浏览器按 `Ctrl+Shift+R` 强制刷新。

### 6.3 修改 Nginx 配置

Nginx 配置被复制进前端镜像，也需要重建前端：

```bash
docker compose --env-file .env.production build frontend

docker compose --env-file .env.production up -d \
  --force-recreate --no-deps frontend
```

### 6.4 修改后端 Python 源码

```bash
docker compose --env-file .env.production build backend

docker compose --env-file .env.production up -d \
  --force-recreate --no-deps backend
```

### 6.5 修改 Python 依赖

修改 `backend/requirements.txt` 后建议无缓存构建：

```bash
docker compose --env-file .env.production build --no-cache backend

docker compose --env-file .env.production up -d \
  --force-recreate --no-deps backend
```

### 6.6 更新全部项目

```bash
git pull
docker compose --env-file .env.production up -d --build
docker compose --env-file .env.production ps
```

## 7. 查看日志

### 7.1 实时查看后端日志

```bash
docker compose --env-file .env.production logs -f --tail=200 backend
```

按 `Ctrl+C` 退出日志查看，不会停止容器。

### 7.2 查看后端最近 200 行

```bash
docker compose --env-file .env.production logs --tail=200 backend
```

### 7.3 查看最近 10 分钟后端日志

```bash
docker compose --env-file .env.production logs --since=10m backend
```

### 7.4 显示时间戳

```bash
docker compose --env-file .env.production logs \
  -f --timestamps --tail=200 backend
```

### 7.5 筛选后端错误

```bash
docker compose --env-file .env.production logs --tail=500 backend 2>&1 \
  | grep -Ei 'error|exception|traceback|failed|critical|401|403|500'
```

### 7.6 查看前端 Nginx 日志

```bash
docker compose --env-file .env.production logs -f --tail=200 frontend
```

### 7.7 查看 MySQL 日志

```bash
docker compose --env-file .env.production logs -f --tail=200 mysql
```

### 7.8 同时查看全部服务

```bash
docker compose --env-file .env.production logs -f --tail=100
```

### 7.9 按容器名查看

```bash
docker logs -f --tail=200 ai-interview-backend-1
docker logs -f --tail=200 ai-interview-frontend-1
docker logs -f --tail=200 ai-interview-mysql-1
docker logs -f --tail=200 qdrant
```

### 7.10 保存日志到文件

```bash
docker compose --env-file .env.production logs \
  --since=1h --timestamps backend > backend.log
```

查看：

```bash
less backend.log
```

在 `less` 中按 `q` 退出。

## 8. 健康检查

查看全部容器状态：

```bash
docker compose --env-file .env.production ps
```

检查 Nginx：

```bash
curl -i http://127.0.0.1/health
```

检查经过 Nginx 代理的 FastAPI：

```bash
curl -i http://127.0.0.1/api/health
```

正常响应：

```json
{"status":"ok"}
```

从后端容器内部检查 FastAPI：

```bash
docker compose --env-file .env.production exec backend \
  curl -fsS http://127.0.0.1:8000/api/health
```

查看后端健康检查历史：

```bash
docker inspect ai-interview-backend-1 \
  --format '{{range .State.Health.Log}}{{println .Start .ExitCode .Output}}{{end}}'
```

## 9. Qdrant 检查命令

查看独立 Qdrant 容器：

```bash
docker ps --filter name=qdrant
```

通过宿主机测试：

```bash
curl -fsS http://127.0.0.1:6333/collections
```

通过共享 Docker 网络测试：

```bash
docker run --rm --network ai-interview-network curlimages/curl:8.10.1 \
  -fsS http://qdrant:6333/collections
```

查看 Qdrant 日志：

```bash
docker logs -f --tail=200 qdrant
```

Qdrant Dashboard：

```text
http://192.168.150.101:6333/dashboard
```

## 10. MySQL 操作

查看数据库：

```bash
docker compose --env-file .env.production exec mysql \
  sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" -e "SHOW DATABASES;"'
```

查看项目数据表：

```bash
docker compose --env-file .env.production exec mysql \
  sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" -e "SHOW TABLES;"'
```

查看用户数据：

```bash
docker compose --env-file .env.production exec mysql \
  sh -c 'mysql -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE" -e "SELECT id,email,username,is_admin,created_at FROM users;"'
```

备份 MySQL：

```bash
docker compose --env-file .env.production exec -T mysql \
  sh -c 'exec mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE"' \
  > ai_interview_$(date +%F_%H%M%S).sql
```

检查备份：

```bash
ls -lh ai_interview_*.sql
```

## 11. 管理员操作

先在网页注册用户，再提升为管理员：

```bash
docker compose --env-file .env.production exec backend \
  python scripts/promote_admin.py "注册邮箱"
```

例如：

```bash
docker compose --env-file .env.production exec backend \
  python scripts/promote_admin.py "admin@example.com"
```

重新登录后访问知识库管理页面：

```text
http://192.168.150.101/knowledge
```

## 12. 清理无用镜像

更新完成并确认服务正常后，可以清理悬空镜像：

```bash
docker image prune -f
```

查看 Docker 磁盘占用：

```bash
docker system df
```

不建议在不了解影响时执行 `docker system prune -a`，它可能删除后续仍需使用的镜像和构建缓存。

## 13. 最常用命令速查

### 启动

```bash
cd "/root/ai面试官"
docker compose --env-file .env.production up -d
```

### 停止但保留容器

```bash
docker compose --env-file .env.production stop
```

### 停止并删除容器，但保留数据库卷

```bash
docker compose --env-file .env.production down
```

### 查看状态

```bash
docker compose --env-file .env.production ps
```

### 实时查看后端日志

```bash
docker compose --env-file .env.production logs -f --tail=200 backend
```

### 更新前端

```bash
docker compose --env-file .env.production build --no-cache frontend
docker compose --env-file .env.production up -d \
  --force-recreate --no-deps frontend
```

### 更新后端

```bash
docker compose --env-file .env.production build backend
docker compose --env-file .env.production up -d \
  --force-recreate --no-deps backend
```

### 修改环境变量后重建后端容器

```bash
docker compose --env-file .env.production up -d \
  --force-recreate --no-deps backend
```

### 完整健康检查

```bash
docker compose --env-file .env.production ps
curl -fsS http://127.0.0.1/health
curl -fsS http://127.0.0.1/api/health
curl -fsS http://127.0.0.1:6333/collections
```
