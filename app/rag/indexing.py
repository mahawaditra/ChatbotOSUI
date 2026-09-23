"""
Modul indexing: memproses PDF → chunks → embed → simpan ke vector store lokal.
Menggunakan REST API v1 langsung untuk embedding (text-embedding-004 hanya ada di v1, bukan v1beta).
"""

import logging
import re
import requests
import time
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    GEMINI_API_KEY,
    DOKUMEN_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PASAL_CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)
from app.rag.prompts import _neutralize_tags
from app.rag.vector_store import save as save_vector_store

logger = logging.getLogger(__name__)

# Endpoint Gemini Embedding API v1
_EMBED_URL = "https://generativelanguage.googleapis.com/v1/models/{model}:embedContent"

# Header Pasal asli di ADART-OSUIMahawaditra-2022.pdf selalu memakai dua spasi literal
# ("Pasal  1"), sedangkan referensi silang di dalam isi pasal (mis. "diatur dalam Pasal 5")
# memakai spasi tunggal/artefak line-wrap dari ekstraksi PDF — sehingga pola ini tidak
# salah memecah di tengah kalimat. Diverifikasi terhadap dokumen asli sebelum dipakai.
_PASAL_HEADER_PATTERN = re.compile(r"Pasal {2}\d+")
_MIN_PASAL_MATCHES = 3  # di bawah ini dianggap bukan dokumen berstruktur Pasal (mis. SOP)

# Splitter khusus untuk chunk sadar-Pasal, dengan chunk_size lebih longgar (PASAL_CHUNK_SIZE)
# daripada splitter default — supaya satu Pasal tidak kepotong di tengah kalimat kalau
# panjangnya melebihi CHUNK_SIZE biasa (lihat komentar PASAL_CHUNK_SIZE di config.py).
_PASAL_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=PASAL_CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# Reindex memanggil embed API ratusan kali berturut-turut (satu per chunk), jadi butuh
# retry yang lebih tahan banting daripada _call_llm_with_retry di chain.py (yang cuma
# 1x retry untuk permintaan chat langsung) — proses admin ini boleh lebih lambat asal
# tidak gagal total gara-gara rate limit sesaat/menit.
_EMBED_MAX_ATTEMPTS = 4
_EMBED_RETRY_DELAYS = [3, 8, 20]  # detik, exponential backoff antar percobaan
_EMBED_CALL_SPACING = 0.3  # jeda kecil antar panggilan sukses, biar tidak burst ke rate limit


def _embed_one_with_retry(text: str) -> list[float]:
    """
    Embed satu teks lewat Gemini REST API v1, dengan retry + exponential backoff kalau
    kena error sesaat/rate limit (429/503/quota/unavailable).
    """
    url = _EMBED_URL.format(model=EMBEDDING_MODEL)
    for attempt in range(_EMBED_MAX_ATTEMPTS):
        try:
            response = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json={
                    "content": {"parts": [{"text": text}]},
                    "outputDimensionality": EMBEDDING_DIMENSION,
                },
                timeout=30,
            )
            response.raise_for_status()
            return response.json()["embedding"]["values"]

        except requests.exceptions.RequestException as e:
            error_str = str(e).lower()
            is_transient = (
                "429" in error_str or "resource exhausted" in error_str or "quota" in error_str
                or "503" in error_str or "unavailable" in error_str
            )
            if is_transient and attempt < _EMBED_MAX_ATTEMPTS - 1:
                delay = _EMBED_RETRY_DELAYS[attempt]
                logger.warning(
                    f"Embed API error sesaat (percobaan {attempt + 1}/{_EMBED_MAX_ATTEMPTS}), "
                    f"retry dalam {delay} detik... ({e})"
                )
                time.sleep(delay)
            else:
                raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Menghasilkan embedding untuk setiap teks menggunakan Gemini REST API v1.

    Args:
        texts: List teks yang akan di-embed

    Returns:
        List of embedding vectors (list of float)
    """
    vectors = []
    for i, text in enumerate(texts):
        if i > 0:
            time.sleep(_EMBED_CALL_SPACING)
        vectors.append(_embed_one_with_retry(text))
    return vectors


def _build_offset_page_map(pages: list[Document]) -> list[tuple[int, int]]:
    """
    Peta (offset awal, nomor halaman 1-indexed) untuk tiap halaman di dalam full_text
    hasil gabungan `"\\n".join(page.page_content for page in pages)`.
    """
    offsets = []
    cursor = 0
    for page_doc in pages:
        page_num = page_doc.metadata.get("page", 0) + 1
        offsets.append((cursor, page_num))
        cursor += len(page_doc.page_content) + 1  # +1 untuk newline penggabung
    return offsets


def _page_for_offset(offsets: list[tuple[int, int]], pos: int) -> int:
    """Cari nomor halaman yang memuat posisi karakter `pos` di full_text gabungan."""
    page = offsets[0][1]
    for start, page_num in offsets:
        if start > pos:
            break
        page = page_num
    return page


def _split_pasal_aware(pages: list[Document], file_name: str) -> list[Document] | None:
    """
    Split dokumen di boundary "Pasal  <N>" (bukan murni per-karakter) supaya satu Pasal
    AD/ART tidak terpotong jadi 2 chunk. Pasal yang tetap kepanjangan (>PASAL_CHUNK_SIZE,
    kasus langka) masih dipecah lebih lanjut oleh `_PASAL_SPLITTER`.

    Returns:
        list[Document] kalau dokumen terdeteksi berstruktur Pasal, None kalau tidak —
        pemanggil lalu fallback ke `splitter.split_documents(pages)` seperti biasa
        (ini yang membuat dokumen SOP yang tidak berstruktur Pasal otomatis tidak terpengaruh
        dan aman kalau format ADART berubah suatu saat).
    """
    full_text = "\n".join(page_doc.page_content for page_doc in pages)

    if len(_PASAL_HEADER_PATTERN.findall(full_text)) < _MIN_PASAL_MATCHES:
        return None

    offsets = _build_offset_page_map(pages)

    parts: list[str] = []
    metadatas: list[dict] = []
    cursor = 0
    for part in re.split(f"(?={_PASAL_HEADER_PATTERN.pattern})", full_text):
        if part.strip():
            parts.append(part.strip())
            metadatas.append({"file": file_name, "page": _page_for_offset(offsets, cursor)})
        cursor += len(part)

    return _PASAL_SPLITTER.create_documents(parts, metadatas=metadatas)


def run_indexing() -> dict[str, Any]:
    """
    Proses utama indexing:
    1. Scan semua PDF di folder dokumen/
    2. Load dan split setiap PDF menjadi chunks
    3. Embed setiap chunk dengan Gemini
    4. Simpan ke vector store lokal (data/vector_store/)

    Returns:
        Dict berisi total_chunks_indexed dan files_processed
    """
    pdf_files = list(DOKUMEN_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Tidak ada file PDF di folder: {DOKUMEN_DIR}")

    logger.info(f"Ditemukan {len(pdf_files)} file PDF untuk diproses.")

    # Setup text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    files_processed = []

    for pdf_path in pdf_files:
        file_name = pdf_path.name
        logger.info(f"Memproses: {file_name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

            # Netralkan tag pembatas prompt (<konteks_dokumen>, <pertanyaan_user>, dst.) kalau
            # secara literal muncul di teks PDF, supaya dokumen yang "diracuni" tidak bisa
            # menyamar sebagai instruksi sistem saat teks ini dikutip ke dalam prompt LLM nanti.
            for page_doc in pages:
                page_doc.page_content = _neutralize_tags(page_doc.page_content)

            pasal_chunks = _split_pasal_aware(pages, file_name)
            if pasal_chunks is not None:
                chunks = pasal_chunks
                logger.info(f"  Struktur Pasal terdeteksi pada {file_name}, pakai chunking sadar-Pasal.")
            else:
                chunks = splitter.split_documents(pages)

                # Pastikan metadata file_name tersimpan di setiap chunk
                for chunk in chunks:
                    chunk.metadata["file"] = file_name
                    # page sudah otomatis diset oleh PyPDFLoader (0-indexed), ubah ke 1-indexed
                    if "page" in chunk.metadata:
                        chunk.metadata["page"] = chunk.metadata["page"] + 1

            all_chunks.extend(chunks)
            files_processed.append(file_name)
            logger.info(f"  → {len(chunks)} chunks dari {file_name}")

        except Exception as e:
            logger.error(f"Gagal memproses {file_name}: {e}")
            raise

    logger.info(f"Total chunks: {len(all_chunks)}. Mulai embedding...")

    # Embed tiap chunk, ditampung di memori sampai akhir (bukan streaming ke store per-batch)
    # — korpus ini kecil (ratusan chunk), aman ditampung sekaligus sebelum satu kali tulis
    # ke vector store lokal lewat save_vector_store() di bawah.
    all_vectors: list[list[float]] = []
    all_metadatas: list[dict] = []

    BATCH_SIZE = 20  # lebih kecil karena embed 1 per 1
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]

        # Embed teks
        vectors = embed_texts(texts)

        all_vectors.extend(vectors)
        all_metadatas.extend(
            {
                "text": texts[j],
                "file": metadatas[j].get("file", "unknown"),
                "page": metadatas[j].get("page", 0),
            }
            for j in range(len(batch))
        )
        logger.info(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} chunks di-embed.")

    save_vector_store(all_vectors, all_metadatas)

    logger.info("Indexing selesai!")
    return {
        "total_chunks_indexed": len(all_chunks),
        "files_processed": files_processed,
    }
