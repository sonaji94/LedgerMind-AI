# ============================================================
# LedgerMind AI — Frontend Streamlit Dockerfile
# ============================================================
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit altair pandas numpy httpx

COPY frontend/ ./frontend/
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
