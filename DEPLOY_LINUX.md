# Linux Docker Deployment

This deployment reuses the existing Docker container named `qdrant`. Compose starts only MySQL, FastAPI, and the Vue/Nginx frontend.

## 1. Install Docker

Install Docker Engine and the Docker Compose plugin. Confirm that both commands work:

```bash
docker --version
docker compose version
```

## 2. Connect the existing Qdrant container

Create one shared Docker network and connect the existing container to it:

```bash
docker network inspect ai-interview-network >/dev/null 2>&1 || \
  docker network create ai-interview-network

docker network inspect ai-interview-network \
  --format '{{json .Containers}}' | grep -q '"Name":"qdrant"' || \
  docker network connect ai-interview-network qdrant
```

Verify connectivity from a temporary container:

```bash
docker run --rm --network ai-interview-network curlimages/curl:8.10.1 \
  -fsS http://qdrant:6333/collections
```

The application uses `http://qdrant:6333`. Port `6334` is not required by the current Python client configuration.

Your current Qdrant publishes `6333-6334` on `0.0.0.0`, which makes both ports reachable on every server interface unless blocked by a firewall. Once container-network access is confirmed, restrict those ports in the cloud security group/firewall or recreate Qdrant without public port mappings.

## 3. Configure production secrets

From the project root:

```bash
cp .env.production.example .env.production
openssl rand -hex 32
nano .env.production
```

Set strong MySQL passwords, the generated JWT secret, model API key, and DashScope API key. Configure OSS variables when knowledge file upload is required.

Database passwords should use URL-safe characters because Compose builds `DATABASE_URL` from them. Letters, numbers, `_`, `-`, and `.` are safe choices.

## 4. Build and start

```bash
docker compose --env-file .env.production up -d --build
docker compose --env-file .env.production ps
```

The application is available at `http://SERVER_IP/`. Only frontend Nginx publishes an application port. MySQL and FastAPI remain on the shared Docker network.

When port 80 is occupied, set `HTTP_PORT=8080` in `.env.production` and visit `http://SERVER_IP:8080/`.

## 5. Inspect logs and health

```bash
docker compose --env-file .env.production logs -f backend
curl http://127.0.0.1/health
curl http://127.0.0.1/api/health
```

To confirm that the backend resolves Qdrant:

```bash
docker compose --env-file .env.production exec backend \
  python -c "import urllib.request; print(urllib.request.urlopen('http://qdrant:6333/collections').status)"
```

## 6. Updates

```bash
git pull
docker compose --env-file .env.production up -d --build
docker image prune -f
```

The current application still creates and adjusts tables during backend startup. Before multi-instance deployment, replace this with Alembic migrations.

## 7. Backups

Back up MySQL regularly:

```bash
docker compose --env-file .env.production exec -T mysql \
  sh -c 'exec mysqldump -uroot -p"$MYSQL_ROOT_PASSWORD" "$MYSQL_DATABASE"' \
  > ai_interview_$(date +%F).sql
```

The `mysql_data` named volume persists across ordinary container recreation. The existing Qdrant container keeps using its original storage configuration and is not deleted or modified by this Compose project.

Do not run `docker compose down -v` unless intentionally deleting the application MySQL data.

## 8. HTTPS

For an internet-facing deployment, place Caddy, Traefik, or host Nginx in front of this stack. Bind the application to a local port and allow only public ports 80 and 443 in the server firewall.

```env
HTTP_PORT=127.0.0.1:8080
```

Then proxy the public domain to `http://127.0.0.1:8080` and issue a valid TLS certificate.
