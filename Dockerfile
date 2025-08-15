FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    xfonts-75dpi \
    xfonts-base \
    libgl1 \
    libglib2.0-0 \
    fontconfig \
    ca-certificates \
    curl \
    gnupg \
 && curl -sSL https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.buster_amd64.deb -o wkhtmltox.deb \
 && apt-get install -y ./wkhtmltox.deb \
 && rm wkhtmltox.deb \
 && rm -rf /var/lib/apt/lists/*

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY . /app

# Use $PORT from Render, default 8080 locally
CMD sh -c 'gunicorn -w 2 -k gthread -t 180 -b 0.0.0.0:${PORT:-8080} app:app'
