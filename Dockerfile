FROM python:3.12-slim

# libzbar0 is required by pyzbar for barcode decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

# 1 worker + 8 threads suits Cloud Run's per-instance concurrency.
# Timeout 300s to accommodate large PDFs.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 main:app
