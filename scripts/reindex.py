"""
Reindex mandiri — memanggil app.rag.indexing.run_indexing() langsung tanpa lewat HTTP.

Dipakai sebagai jalur reindex utama untuk instance production di Vercel: script ini
bicara langsung ke Upstash + Gemini, bukan ke aplikasi yang di-deploy, jadi tidak
terikat batas durasi Vercel Function sama sekali (yang di plan Hobby jauh lebih pendek
dari waktu reindex kita).

Penggunaan:
    python scripts/reindex.py

Pastikan .env (atau environment variable) sudah berisi GEMINI_API_KEY,
UPSTASH_VECTOR_REST_URL, dan UPSTASH_VECTOR_REST_TOKEN yang sesuai target
(lokal atau production) SEBELUM menjalankan ini — script ini akan menghapus
total index yang ditunjuk kredensial tersebut lalu membangunnya ulang.
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
