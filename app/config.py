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

# --- Upstash Redis (dipakai khusus untuk rate limiting /api/chat — database terpisah
# dari vector store lokal di bawah, buat database Redis baru di dashboard Upstash) ---
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

# --- Path folder dokumen PDF ---
# Di container Docker, dokumen/ ada di root project (/app/dokumen)
DOKUMEN_DIR: Path = Path(__file__).parent.parent / "dokumen"

# --- Path folder vector store lokal (vectors.npy + metadata.json) ---
# Di-generate oleh scripts/reindex.py dan di-commit ke git, supaya proyek tidak
# bergantung pada akun cloud siapa pun untuk vector search.
VECTOR_STORE_DIR: Path = Path(__file__).parent.parent / "data" / "vector_store"

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

# Jumlah kandidat yang diambil dari vector store SEBELUM dipangkas ke TOP_K.
# Peninggalan dari era Upstash: approximate nearest-neighbor search Upstash pada corpus
# ini (skor antar chunk sangat rapat, ~0.75-0.81) baru stabil menemukan hasil #1 yang
# benar mulai top_k>=20 — top_k=5/10 langsung ke Upstash bisa melewatkan chunk paling
# relevan sama sekali. Vector store lokal (numpy, exact search) tidak punya masalah ini,
# tapi nilai ini sengaja dipertahankan apa adanya untuk saat ini. Filter
# SIMILARITY_THRESHOLD & pemangkasan ke TOP_K dilakukan setelah fetch ini, di retrieval.py.
RETRIEVAL_FETCH_K: int = 25

# Skor similarity minimum agar sebuah chunk dianggap relevan. Skor dihitung sebagai cosine
# similarity yang dinormalisasi ke rentang [0,1] via (1 + cosine_similarity) / 2 (formula
# yang sama seperti skor yang dulu dikembalikan Upstash untuk metric COSINE) — lihat
# app/rag/vector_store.py::query(). Nilai ini dikalibrasi terhadap skor ternormalisasi itu.
SIMILARITY_THRESHOLD: float = 0.5

# --- LLM settings ---
LLM_MODEL: str = "gemini-3.1-flash-lite"
EMBEDDING_MODEL: str = "gemini-embedding-001"
# Ukuran vektor embedding — JANGAN diubah tanpa reindex ulang total (data/vector_store/).
# Vector store lokal tidak memvalidasi kecocokan dimensi antara vektor tersimpan dan query
# baru; mismatch akan menghasilkan cosine similarity yang tidak berarti, bukan error.
EMBEDDING_DIMENSION: int = 1536

# --- Batas input & rate limiting /api/chat ---
MAX_QUESTION_LENGTH: int = 500
RATE_LIMIT_REQUESTS: int = 10  # jumlah request maksimum per IP per jendela waktu di bawah
RATE_LIMIT_WINDOW_SECONDS: int = 60
