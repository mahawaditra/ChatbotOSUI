"""
Reindex mandiri — memanggil app.rag.indexing.run_indexing() langsung tanpa lewat HTTP.

Dipakai sebagai jalur reindex utama untuk instance production di Vercel. Alasan
utamanya BUKAN LAGI cuma batas durasi Vercel Function (walau itu tetap benar):
filesystem Vercel read-only di luar /tmp, jadi proses yang berjalan di sana tidak bisa
menulis data/vector_store/vectors.npy + metadata.json secara permanen. Vector store
lokal harus di-generate di sini, lalu di-commit ke git dan di-deploy ulang seperti file
kode biasa — lihat README.md bagian "Updating documents" / "Deployment".

Penggunaan:
    python scripts/reindex.py

Pastikan .env (atau environment variable) sudah berisi GEMINI_API_KEY yang valid
SEBELUM menjalankan ini — script ini akan menimpa total data/vector_store/vectors.npy
dan metadata.json dengan hasil reindex penuh dari seluruh PDF di dokumen/.
"""

import sys
from pathlib import Path

# Supaya "import app...." tetap resolve terlepas dari cwd saat script ini dipanggil.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.indexing import run_indexing  # noqa: E402


def main() -> None:
    print("Memulai reindex...")
    result = run_indexing()
    print(f"Selesai. Total chunk ter-index: {result['total_chunks_indexed']}")
    print(f"File diproses: {', '.join(result['files_processed'])}")


if __name__ == "__main__":
    main()
