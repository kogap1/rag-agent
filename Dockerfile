FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/uploaded_files /app/server_vector_db /app/state /app/runs /app/models \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/v1/health/live', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

