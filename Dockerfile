FROM python:3.12-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /install /usr/local

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser


COPY alembic.ini .
COPY pyproject.toml .
COPY ./app ./app
COPY ./migrations ./migrations

RUN mkdir -p ./uploads && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["fastapi", "run", "--port", "8000"]
