FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies yang diperlukan oleh pypdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies terlebih dahulu (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code aplikasi
COPY app/ ./app/

# Copy dokumen PDF ke dalam image
COPY dokumen/ ./dokumen/

# Copy vector store lokal (vectors.npy + metadata.json) — hasil `python scripts/reindex.py`,
# di-commit ke git seperti dokumen/. Jalur ini legacy/tidak dipakai di produksi (lihat
# README), tapi disertakan supaya tidak diam-diam rusak kalau ada yang pakai Docker.
COPY data/ ./data/

# Copy Logo untuk web interface
COPY Logo.png ./Logo.png

# Jangan jalankan sebagai root di container. Folder logs/ dibuat & di-chown lebih dulu supaya
# _log_request() di app/main.py tetap bisa menulis setelah pindah ke user non-root ini.
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

# Expose port 8080 (standar Render)
EXPOSE 8080

# Jalankan FastAPI dengan uvicorn — PORT di-inject Railway secara otomatis
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
