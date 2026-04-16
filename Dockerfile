FROM python:3.12-slim

# libzbar0 is the native shared library required by pyzbar for barcode decoding.
# ldconfig refreshes the linker cache so the .so is found at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    && ldconfig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# PORT is overridden at runtime by Cloud Run; 8080 is the local default.
ENV PORT=8080

# 1 worker + 8 threads suits Cloud Run's per-instance concurrency model.
# Timeout 300s to accommodate large PDF uploads and processing.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 main:app
