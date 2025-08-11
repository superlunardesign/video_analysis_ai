
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg wkhtmltopdf libgl1 libglib2.0-0 fontconfig ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt


COPY . /app

# optional: bake the knowledge index during build
# RUN python ingest_knowledge.py || true

ENV PORT=8080
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-t", "180", "-b", "0.0.0.0:8080", "app:app"]
