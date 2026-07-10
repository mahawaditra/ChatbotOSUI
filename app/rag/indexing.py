"""
Modul indexing: memproses PDF → chunks → embed → simpan ke Upstash Vector.
Menggunakan REST API v1 langsung untuk embedding (text-embedding-004 hanya ada di v1, bukan v1beta).
"""

import logging
import requests
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from upstash_vector import Index

from app.config import (
    GEMINI_API_KEY,
    UPSTASH_VECTOR_REST_URL,
    UPSTASH_VECTOR_REST_TOKEN,
    DOKUMEN_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
)

logger = logging.getLogger(__name__)

# Endpoint Gemini Embedding API v1
_EMBED_URL = "https://generativelanguage.googleapis.com/v1/models/{model}:embedContent"


def get_vector_index() -> Index:
    """Membuat koneksi ke Upstash Vector Index."""
    return Index(
        url=UPSTASH_VECTOR_REST_URL,
        token=UPSTASH_VECTOR_REST_TOKEN,
    )


def delete_all_vectors(index: Index) -> None:
    """Menghapus semua vektor lama sebelum reindex."""
    logger.info("Menghapus semua vektor lama dari Upstash...")
    index.reset()
    logger.info("Semua vektor berhasil dihapus.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Menghasilkan embedding untuk setiap teks menggunakan Gemini REST API v1.

    Args:
        texts: List teks yang akan di-embed

    Returns:
        List of embedding vectors (list of float)
    """
    url = _EMBED_URL.format(model=EMBEDDING_MODEL)
    vectors = []
    for text in texts:
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
        values = response.json()["embedding"]["values"]
        vectors.append(values)
    return vectors


def run_indexing() -> dict[str, Any]:
    """
    Proses utama indexing:
    1. Scan semua PDF di folder dokumen/
    2. Load dan split setiap PDF menjadi chunks
    3. Embed setiap chunk dengan Gemini
    4. Simpan ke Upstash Vector

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

    # Koneksi ke Upstash
    index = get_vector_index()

    # Hapus vektor lama
    delete_all_vectors(index)

    all_chunks = []
    files_processed = []

    for pdf_path in pdf_files:
        file_name = pdf_path.name
        logger.info(f"Memproses: {file_name}")

        try:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()

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

    logger.info(f"Total chunks: {len(all_chunks)}. Mulai embedding dan upload ke Upstash...")

    # Embed dan upload ke Upstash dalam batch
    BATCH_SIZE = 20  # lebih kecil karena embed 1 per 1
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]
        metadatas = [chunk.metadata for chunk in batch]

        # Embed teks
        vectors = embed_texts(texts)

        # Siapkan data untuk Upstash
        upsert_data = [
            {
                "id": f"chunk-{i + j}",
                "vector": vectors[j],
                "metadata": {
                    "text": texts[j],
                    "file": metadatas[j].get("file", "unknown"),
                    "page": metadatas[j].get("page", 0),
                },
            }
            for j in range(len(batch))
        ]

        index.upsert(vectors=upsert_data)
        logger.info(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} chunks di-upload.")

    logger.info("Indexing selesai!")
    return {
        "total_chunks_indexed": len(all_chunks),
        "files_processed": files_processed,
    }
