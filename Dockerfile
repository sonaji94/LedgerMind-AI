# ============================================================
# LedgerMind AI — Production Multi-Stage Backend Dockerfile
# Base: Python 3.11-slim (Native ONNX Runtime, XGBoost & Linux wheels)
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir .

# ── Stage 2: Runtime ────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

COPY app/ ./app/
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# Ensure ONNX artifacts and schemas are verified or built
RUN python -c "from pathlib import Path; (Path('ml/models/xgb_champion.onnx').is_file() and Path('ml/models/iso_forest.onnx').is_file()) or (__import__('ml.train').train.run_training() and __import__('ml.export_onnx').export_onnx.run_export())"

RUN groupadd --system appuser && \
    useradd --system --gid appuser appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/readyz || exit 1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
