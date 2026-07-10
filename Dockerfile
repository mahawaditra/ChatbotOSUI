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

# Copy Logo untuk web interface
COPY Logo.png ./Logo.png

# Expose port 8080 (standar Render)
EXPOSE 8080

# Jalankan FastAPI dengan uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
