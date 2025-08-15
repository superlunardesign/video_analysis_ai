FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    xfonts-75dpi \
    xfonts-base \
    libgl1 \
    libglib2.0-0 \
    fontconfig \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD sh -c 'gunicorn -w 2 -k gthread -t 180 -b 0.0.0.0:${PORT:-8080} app:app'
