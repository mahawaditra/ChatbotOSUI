"""
Konfigurasi aplikasi dari environment variables.
Semua secret dibaca dari .env (lokal) atau environment variables (produksi Render).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env saat development lokal; di Render tidak ada .env tapi env vars sudah di-set
load_dotenv()

# --- Google AI Studio ---
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# --- Upstash Vector ---
UPSTASH_VECTOR_REST_URL: str = os.environ["UPSTASH_VECTOR_REST_URL"]
UPSTASH_VECTOR_REST_TOKEN: str = os.environ["UPSTASH_VECTOR_REST_TOKEN"]

# --- Admin ---
ADMIN_KEY: str = os.environ["ADMIN_KEY"]

# --- Nama koleksi / namespace Upstash (opsional, default: org-rag) ---
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "org-rag")

# --- Path folder dokumen PDF ---
# Di container Docker, dokumen/ ada di root project (/app/dokumen)
DOKUMEN_DIR: Path = Path(__file__).parent.parent / "dokumen"

# --- RAG settings ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K: int = 5  # jumlah chunks yang diambil saat retrieval

# --- LLM settings ---
LLM_MODEL: str = "gemini-3.1-flash-lite"
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 1536  # Upstash free tier max, gemini-embedding-001 support truncation
