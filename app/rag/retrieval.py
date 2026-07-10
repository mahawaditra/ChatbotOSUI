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

    # Query Upstash Vector
    index = get_vector_index()
    results = index.query(
        vector=question_vector,
        top_k=TOP_K,
        include_metadata=True,
    )

    if not results:
        logger.warning("Tidak ada hasil retrieval dari Upstash. Database mungkin kosong.")
        return []

    # Format hasil
    context_chunks = []
    for result in results:
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
