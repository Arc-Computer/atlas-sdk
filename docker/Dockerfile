FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY atlas ./atlas
COPY examples ./examples
COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY docker/configs ./docker/configs

RUN pip install --upgrade pip \
    && pip install --no-cache-dir .

RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]
