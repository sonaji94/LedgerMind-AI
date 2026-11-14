# ============================================================
# LedgerMind AI — Frontend Streamlit Dockerfile
# Base: Python 3.11-slim
# ============================================================
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit altair pandas numpy httpx pydantic pydantic-settings

COPY frontend/ ./frontend/
COPY app/ ./app/
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run frontend/dashboard.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
