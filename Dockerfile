# algotr backend only. The React terminal lives in the algotr-frontend repo
# and is served by its own nginx image (see the `frontend` service in
# docker-compose.yml); this image is a pure API and does not build it.
FROM python:3.13-slim AS sidecar
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/
WORKDIR /app
COPY finrl_sidecar/requirements.txt finrl_sidecar/requirements.txt
RUN uv venv --python 3.13 finrl_sidecar/.venv \
    && uv pip install --python finrl_sidecar/.venv/bin/python torch \
        --index-url $TORCH_INDEX \
    && uv pip install --python finrl_sidecar/.venv/bin/python -r finrl_sidecar/requirements.txt

FROM python:3.13-slim AS runtime
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY finrl_sidecar/ finrl_sidecar/
COPY --from=sidecar /app/finrl_sidecar/.venv finrl_sidecar/.venv
COPY src/ src/
COPY trading_node/ trading_node/
COPY migrations/ migrations/
COPY alembic.ini ./
RUN uv sync --frozen --no-dev
RUN useradd --create-home --uid 10001 algotr && mkdir -p /data /catalog /events /results \
    && chown -R algotr /app /data /catalog /events /results
USER algotr
ENV PYTHONPATH=/app:/app/src
EXPOSE 8000 9101
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
