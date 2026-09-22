"""
Modul retrieval: mengambil chunks paling relevan dari Upstash Vector.
Menggunakan REST API v1 langsung untuk embedding (text-embedding-004 hanya ada di v1, bukan v1beta).
"""

import logging
import requests
from typing import Any

from upstash_vector import Index

from app.config import (
    GEMINI_API_KEY,
    UPSTASH_VECTOR_REST_URL,
    UPSTASH_VECTOR_REST_TOKEN,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    TOP_K,
    RETRIEVAL_FETCH_K,
    SIMILARITY_THRESHOLD,
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


def retrieve_context(question: str) -> list[dict[str, Any]]:
    """
    Mengambil top-K chunks paling relevan untuk pertanyaan yang diberikan.

    Args:
        question: Pertanyaan dari user dalam bahasa Indonesia

    Returns:
        List of dict berisi {text, file, page} untuk setiap chunk yang relevan
    """
    # Embed pertanyaan user via REST API v1 langsung
    url = _EMBED_URL.format(model=EMBEDDING_MODEL)
    response = requests.post(
        url,
        params={"key": GEMINI_API_KEY},
        json={
            "content": {"parts": [{"text": question}]},
            "outputDimensionality": EMBEDDING_DIMENSION,
        },
        timeout=30,
    )
    response.raise_for_status()
    question_vector = response.json()["embedding"]["values"]

    # Query Upstash Vector. Minta RETRIEVAL_FETCH_K kandidat (bukan langsung TOP_K) —
    # approximate nearest-neighbor search Upstash pada corpus ini terbukti tidak reliable
    # menemukan hasil ter-relevan kalau top_k yang diminta terlalu kecil (lihat komentar
    # RETRIEVAL_FETCH_K di config.py). Dipangkas ke TOP_K di bawah, setelah difilter skor.
    index = get_vector_index()
    results = index.query(
        vector=question_vector,
        top_k=RETRIEVAL_FETCH_K,
        include_metadata=True,
    )

    if not results:
        logger.warning("Tidak ada hasil retrieval dari Upstash. Database mungkin kosong.")
        return []

    # Buang hasil yang skor similarity-nya di bawah threshold (kemungkinan tidak relevan),
    # lalu ambil TOP_K teratas saja untuk masuk ke prompt LLM
    relevant_results = [r for r in results if r.score >= SIMILARITY_THRESHOLD][:TOP_K]
    if not relevant_results:
        top_score = max((r.score for r in results), default=0.0)
        logger.warning(
            f"Semua {len(results)} hasil retrieval di bawah SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD} "
            f"(skor tertinggi: {top_score:.3f}). Kemungkinan pertanyaan di luar topik dokumen."
        )
        return []

    # Format hasil
    context_chunks = []
    for result in relevant_results:
        metadata = result.metadata or {}
        context_chunks.append(
            {
                "text": metadata.get("text", ""),
                "file": metadata.get("file", "unknown"),
                "page": metadata.get("page", 0),
            }
        )
        logger.debug(
            f"Retrieved chunk dari {metadata.get('file')} hal. {metadata.get('page')} "
            f"(score: {result.score:.3f})"
        )

    return context_chunks
