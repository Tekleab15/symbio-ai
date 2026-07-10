FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=8000 \
    FIREWORKS_BASE_URL="https://api.fireworks.ai/inference/v1" \
    FIREWORKS_MODEL="accounts/fireworks/models/gemma2-9b-it" \
    FIREWORKS_MODEL_CHEAP="accounts/fireworks/models/gemma2-9b-it" \
    FIREWORKS_MODEL_FACTUAL="accounts/fireworks/models/gemma2-9b-it" \
    FIREWORKS_MODEL_CODE="accounts/fireworks/models/gemma2-27b-it" \
    SYMBIO_ENABLE_CACHE=1 \
    SYMBIO_DISABLE_CLOUD=0 \
    SYMBIO_MAX_CONCURRENCY=8 \
    SYMBIO_MAX_BATCH_SIZE=512 \
    SYMBIO_TIMEOUT_SECONDS=20 \
    SYMBIO_SANDBOX_TIMEOUT_SECONDS=2.0 \
    SYMBIO_REPLACE_CODEGEN_WITH_STDOUT=0 \
    SYMBIO_LOG_LEVEL=INFO

WORKDIR /workspace/Project/symbio-ai

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --prefer-binary --no-cache-dir -r requirements.txt

COPY app ./app

RUN groupadd --system symbio \
    && useradd --system --gid symbio --home-dir /workspace/Project/symbio-ai --shell /usr/sbin/nologin symbio \
    && chown -R symbio:symbio /workspace/Project/symbio-ai

USER symbio

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

ENTRYPOINT ["tini", "--"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--timeout-keep-alive", "5", "--log-level", "info"]