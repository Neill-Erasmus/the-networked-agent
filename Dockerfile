FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    GRAPHRAG_STORE_PATH=/app/data/graphrag_store.json \
    AGENT_VISUALIZATION_DIR=/app/data/visualizations

WORKDIR /app

RUN addgroup --system app \
    && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /app/data/visualizations \
    && chown -R app:app /app /home/app

COPY --chown=app:app . /app

USER app

VOLUME ["/app/data"]

ENTRYPOINT ["python", "main.py"]