FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    POCKETRAG_DATA=/app/data FASTEMBED_CACHE_PATH=/app/models \
    MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=131072
WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY core ./core
COPY server.py .

EXPOSE 8550
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8550"]
