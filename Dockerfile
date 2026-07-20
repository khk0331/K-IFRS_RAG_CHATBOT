FROM python:3.11-slim

ARG EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

COPY pyproject.toml README.md ./
COPY src ./src
COPY data/sample ./data/sample
RUN if [ -n "$EXTRAS" ]; then pip install ".[${EXTRAS}]"; else pip install .; fi

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "kifrs_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
