# Hosting algotr on Oracle Cloud Always Free

One Ampere A1 VM (Always Free: up to 4 OCPUs / 24 GB RAM) runs the whole
platform in Docker: Postgres, Redis, the API + trading node + terminal,
and Caddy on port 80. Nothing here costs money as long as the VM stays inside the
Always Free limits — and the instance is created with the Always Free
shape, not a paid one.

The stack itself is `docker-compose.yml` at the repo root; the file in
this folder is only an override that locks the public host down (no
public 8000/8080, Caddy login in front). Local runs use the root file
alone.

## 1. The VM

- Shape: **VM.Standard.A1.Flex**, 4 OCPUs / 24 GB RAM (the Always Free
  Ampere allowance). Image: Canonical Ubuntu 24.04.
- Boot volume: 50–100 GB (Always Free allows 200 GB total).
- VCN + subnet: allow ingress **TCP 80** (and **22** for SSH, ideally
  from your IP only) in the subnet security list. Do NOT open 8000.
- SSH in with the key pair created at instance launch.

## 2. On the VM

```bash
sudo apt update && sudo apt install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER   # log out and back in
git clone https://github.com/Melkabli05/algotr-backend.git
cd algotr-backend
cp .env.example .env
```

Fill in `.env` (never commit it):

- `ALPACA_API_KEY` / `ALPACA_API_SECRET` — **paper** keys only.
- `ALPACA_TRADING_MODE=paper`, `NODE_MODE=validate` (the node never
  trades in validate mode; that is the safe default).
- `POSTGRES_PASSWORD` — a long random password for the local Postgres.
- `SEC_CONTACT` — your name + email, so the scanner can read SEC filings.
- `CADDY_USER` — a username for the terminal login.
- `CADDY_PASSWORD_HASH` — a bcrypt hash of the terminal password, made
  with: `docker run --rm caddy:2-alpine caddy hash-password --plaintext 'your-password'`

## 3. Start it

```bash
docker compose -f docker-compose.yml -f deploy/oracle/docker-compose.yml up -d --build
docker compose -f docker-compose.yml -f deploy/oracle/docker-compose.yml exec api python -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read())"
```

Then open `http://<VM public IP>/ui/` and log in with the Caddy
username/password. The terminal is served by the frontend container;
the API is at the same host (`/api/...`, proxied through it).

## 4. Notes and limits

- **No login in the app itself.** Caddy basic-auth is the only gate.
  Keep the password strong, and never expose port 8000 or 8080 publicly.
- **Paper only.** Set `ALPACA_TRADING_MODE=paper` and
  `NODE_MODE=validate` in `.env` (the node never trades in validate
  mode). Changing either needs a deliberate edit and a restart — do
  not do it casually.
- **Always Free reclamation.** Oracle may reclaim an idle Always Free
  instance. Keep the VM doing its daily jobs (scanner/autopilot) and
  check it at least weekly; keep a backup of `.env` and the `pgdata`
  volume somewhere safe.
- **Updates.** Pull both repos, rebuild, re-run migrations:
  `git pull` in both repos, then `docker compose -f docker-compose.yml
  -f deploy/oracle/docker-compose.yml up -d --build` and `alembic upgrade head`
  (the migrate service runs it automatically on start).
