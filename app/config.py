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
# Satu-satunya secret yang benar-benar dibutuhkan untuk indexing (dipakai baik oleh server
# maupun scripts/reindex.py) — satu-satunya yang tetap WAJIB (os.environ[...]) di sini.
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]

# --- Upstash Redis (dipakai khusus untuk rate limiting /api/chat — database terpisah
# dari vector store lokal di bawah, buat database Redis baru di dashboard Upstash) ---
# Dibaca dengan default kosong (BUKAN os.environ[...] wajib) supaya mengimpor modul ini
# secara transitif (mis. scripts/reindex.py -> app.rag.indexing -> app.config) tidak ikut
# memaksa secret produksi yang tidak dipakai reindex tersedia di mesin manapun yang
# menjalankannya. Wajib-nya ditegakkan oleh validate_server_config() di bawah, dipanggil
# app/main.py saat startup server sungguhan — bukan di sini.
UPSTASH_REDIS_REST_URL: str = os.getenv("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_REST_TOKEN: str = os.getenv("UPSTASH_REDIS_REST_TOKEN", "")

# --- Admin ---
# Sama seperti di atas: default kosong di sini, panjang ≥32 karakter ditegakkan oleh
# validate_server_config(), bukan langsung saat modul ini di-import.
ADMIN_KEY: str = os.getenv("ADMIN_KEY", "")


def validate_server_config() -> None:
    """
    Menegakkan secret yang dibutuhkan server sungguhan (dipanggil app/main.py saat startup)
    tapi TIDAK dibutuhkan alur reindex-only (scripts/reindex.py, yang hanya perlu
    GEMINI_API_KEY di atas). Dipisah dari deklarasi module-level supaya mengimpor
    app.config lewat app.rag.indexing tidak transitif memaksa nilai-nilai ini tersedia.
    """
    if len(ADMIN_KEY) < 32:
        raise RuntimeError(
            "ADMIN_KEY terlalu pendek atau kosong (minimal 32 karakter). "
            "Generate dengan: python -c \"import secrets; print(secrets.token_hex(32))\""
        )
    if not UPSTASH_REDIS_REST_URL or not UPSTASH_REDIS_REST_TOKEN:
        raise RuntimeError(
            "UPSTASH_REDIS_REST_URL dan UPSTASH_REDIS_REST_TOKEN wajib diisi (dipakai untuk "
            "rate limiting /api/chat) — buat database Redis terpisah di "
            "https://console.upstash.com/redis"
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
# tapi nilai ini sengaja dipertahankan apa adanya untuk saat ini. Gate SIMILARITY_THRESHOLD
# (pada skor top-1) & pemangkasan ke TOP_K dilakukan setelah fetch ini, di retrieval.py.
RETRIEVAL_FETCH_K: int = 25

# Gate relevansi: skor TOP-1 minimum agar pertanyaan dianggap ada hubungannya dengan dokumen.
# Di bawah ini, Gemini TIDAK dipanggil sama sekali dan jawabannya tanpa sumber. Skor = cosine
# similarity yang dinormalisasi ke [0,1] via (1 + cosine_similarity) / 2 — lihat
# app/rag/vector_store.py::query().
#
# Nilai 0.775 dipilih dari data (246 chunk, embedding tanpa taskType):
#   - pertanyaan nyata dari log yang berhasil DIJAWAB: top-1 terendah 0.790 (di luar sapaan "halo")
#   - pertanyaan jelas di luar topik: 0.737-0.794 (sebagian kecil yang berbau UI/Depok 0.80-0.84)
# Kedua kelompok TUMPANG-TINDIH, jadi tidak ada threshold yang memisahkan bersih; angka ini sengaja
# konservatif (hanya memotong ekor paling jauh, kira-kira seperempat input tak relevan) supaya
# pertanyaan sungguhan tidak salah ditolak. Sisanya ditangani setelah Gemini menjawab: sumber
# disembunyikan untuk jawaban penolakan (lihat app/rag/chain.py). Sudah dicoba dan TIDAK membantu:
# taskType RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT/QUESTION_ANSWERING, z-score, dan selisih terhadap rata-rata.
# Gate TIDAK berlaku untuk pertanyaan berbahasa Inggris/non-Latin (chain._gate_applies): kemiripan
# lintas-bahasa sistematis lebih rendah, pertanyaan Inggris sah terukur 0.768-0.834.
# Setel ulang kalau dokumen berubah banyak (mis. pergantian SOP tahunan) atau ada pertanyaan valid ditolak.
SIMILARITY_THRESHOLD: float = 0.775

# --- LLM settings ---
LLM_MODEL: str = "gemini-3.1-flash-lite"
EMBEDDING_MODEL: str = "gemini-embedding-001"
# Ukuran vektor embedding — JANGAN diubah tanpa reindex ulang total (data/vector_store/).
# Kalau ini diubah tanpa reindex, numpy langsung raise ValueError saat perkalian matriks di
# vector_store.py::query() (dimensi tidak cocok) — bukan silently salah, request itu gagal
# bersih dengan 503. Yang TIDAK terdeteksi otomatis: mengganti EMBEDDING_MODEL ke model lain
# yang kebetulan berdimensi sama — itu tidak crash, tapi vector_store.py::_load() menolak
# melayani query (bukan "informasi tidak ditemukan" yang salah kaprah) kalau manifest.json
# mencatat model berbeda dari yang sekarang dikonfigurasi di sini.
EMBEDDING_DIMENSION: int = 1536

# --- Batas input & rate limiting /api/chat ---
MAX_QUESTION_LENGTH: int = 500
RATE_LIMIT_REQUESTS: int = 10  # jumlah request maksimum per IP per jendela waktu di bawah
RATE_LIMIT_WINDOW_SECONDS: int = 60
