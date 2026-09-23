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

# --- Upstash Redis (dipakai khusus untuk rate limiting /api/chat, database terpisah
# dari Upstash Vector di atas — buat database Redis baru di dashboard Upstash) ---
UPSTASH_REDIS_REST_URL: str = os.environ["UPSTASH_REDIS_REST_URL"]
UPSTASH_REDIS_REST_TOKEN: str = os.environ["UPSTASH_REDIS_REST_TOKEN"]

# --- Admin ---
ADMIN_KEY: str = os.environ["ADMIN_KEY"]
# Minimal 32 karakter (generator resmi di README/CLAUDE.md pakai secrets.token_hex(32) -> 64 hex
# char) — mencegah ADMIN_KEY kosong/lemah lolos begitu saja tanpa ketahuan sampai ada yang coba
# bypass /admin/reindex dengan header X-Admin-Key kosong.
if len(ADMIN_KEY) < 32:
    raise RuntimeError(
        "ADMIN_KEY terlalu pendek atau kosong (minimal 32 karakter). "
        "Generate dengan: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

# --- Nama koleksi / namespace Upstash (opsional, default: org-rag) ---
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "org-rag")

# --- Path folder dokumen PDF ---
# Di container Docker, dokumen/ ada di root project (/app/dokumen)
DOKUMEN_DIR: Path = Path(__file__).parent.parent / "dokumen"

# --- RAG settings ---
CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200

# Batas ukuran khusus untuk chunking sadar-Pasal (lihat indexing.py::_split_pasal_aware).
# Satu Pasal adalah satu unit semantik utuh, jadi bolehnya lebih longgar dari CHUNK_SIZE
# biasa — dipilih 2000 karena Pasal terpanjang di ADART-OSUIMahawaditra-2022.pdf ~1874
# karakter. Nilai ini WAJIB lebih besar dari Pasal terpanjang di dokumen, kalau tidak
# satu Pasal bisa kepotong lagi di tengah kalimat (pernah kejadian: klausa "2/3 dari
# jumlah Badan Pengurus" di Pasal 31 terpotong jadi "2/3 dari jumlah" saja waktu masih
# pakai CHUNK_SIZE=1000, bikin LLM menebak sendiri kata sambungannya dan salah kutip).
PASAL_CHUNK_SIZE: int = 2000

TOP_K: int = 5  # jumlah chunk maksimum yang akhirnya masuk ke prompt LLM

# Jumlah kandidat yang diminta ke Upstash SEBELUM dipangkas ke TOP_K.
# WAJIB lebih besar dari TOP_K: diverifikasi empiris bahwa approximate nearest-neighbor
# search Upstash pada corpus ini (skor antar chunk sangat rapat, ~0.75-0.81) baru stabil
# menemukan hasil #1 yang benar mulai top_k>=20 — top_k=5/10 langsung ke Upstash bisa
# melewatkan chunk paling relevan sama sekali. Filter SIMILARITY_THRESHOLD & pemangkasan
# ke TOP_K dilakukan setelah fetch ini, di retrieval.py.
RETRIEVAL_FETCH_K: int = 25

# Skor similarity minimum (dari Upstash) agar sebuah chunk dianggap relevan.
# Nilai awal, perlu dikalibrasi berdasarkan skor asli di log produksi —
# metrik similarity tergantung setting index Upstash (dibuat via dashboard, bukan di kode ini).
SIMILARITY_THRESHOLD: float = 0.5

# --- LLM settings ---
LLM_MODEL: str = "gemini-3.1-flash-lite"
EMBEDDING_MODEL: str = "gemini-embedding-001"
EMBEDDING_DIMENSION: int = 1536  # Upstash free tier max, gemini-embedding-001 support truncation

# --- Batas input & rate limiting /api/chat ---
MAX_QUESTION_LENGTH: int = 500
RATE_LIMIT_REQUESTS: int = 10  # jumlah request maksimum per IP per jendela waktu di bawah
RATE_LIMIT_WINDOW_SECONDS: int = 60
