# Deployment

The backend and the terminal ([algotr-frontend](https://github.com/Melkabli05/algotr-frontend)) are built and shipped separately. The terminal only needs the API's address.

## How the terminal reaches the API

| Setup | What to do |
| --- | --- |
| One address for both (default) | The terminal's nginx image serves `/ui/` and proxies every other path, and the `/stream` WebSocket, to the API (`API_UPSTREAM`). The browser sees one origin, so there is nothing to configure and no CORS. Docker Compose and the Helm chart use this. |
| Another origin | List the terminal's origin in `CORS_ORIGINS` (a JSON list). No origin is allowed unless it is listed. The terminal itself calls its own origin, so this needs a reverse proxy in front of both rather than a terminal setting. |

## Docker Compose

```bash
cp .env.example .env     # ALPACA_API_KEY and ALPACA_API_SECRET (paper keys)
docker compose up -d --build
```

Starts Postgres, Redis, a one-off `migrate` job, the API (8000), the trading node, a Celery worker and beat, and the terminal (8080, built from the frontend repository's `main`). The terminal is at http://localhost:8080/ui/ (`/` redirects there). The node stays in `NODE_MODE=validate` until `.env` says otherwise. State lives in the `data`, `catalog`, `events`, `results` and `pgdata` volumes. `deploy/oracle/` adds a Caddy login for a public host.

## Helm

`deploy/helm/algotr` runs the API, the live node, Celery, the research jobs and the terminal. The terminal is `frontend.image` (build it from the frontend repository and push it); `ingress.enabled` publishes it on `ingress.host`, and it proxies to the API's Service.

## Models

The API image carries FinRL's environment. On start the API checks `data/models/finbert` and `data/models/kronos` and fetches what is missing in the background, each file checked against a pinned SHA-256, without delaying startup. `data/` is a volume, so it happens once. FinBERT's weights are CC BY-NC-SA 4.0 (non-commercial). `AUTO_FETCH_MODELS=false` (Helm: `config.autoFetchModels`) stops it; `uv run poe setup` fetches them by hand.
