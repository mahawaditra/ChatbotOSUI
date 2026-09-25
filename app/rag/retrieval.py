"""
Modul retrieval: mengambil chunks paling relevan dari vector store lokal.
Menggunakan REST API v1 langsung untuk embedding (text-embedding-004 hanya ada di v1, bukan v1beta).
"""

import logging
import requests
from typing import Any

from app.config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    TOP_K,
    RETRIEVAL_FETCH_K,
    SIMILARITY_THRESHOLD,
)
from app.rag.vector_store import query as query_vector_store

logger = logging.getLogger(__name__)

# Endpoint Gemini Embedding API v1
_EMBED_URL = "https://generativelanguage.googleapis.com/v1/models/{model}:embedContent"


def retrieve_context(question: str, apply_gate: bool = True) -> tuple[list[dict[str, Any]], float]:
    """
    Mengambil top-K chunks paling relevan untuk pertanyaan yang diberikan.

    Args:
        question: Pertanyaan dari user dalam bahasa Indonesia
        apply_gate: False untuk melewati gate SIMILARITY_THRESHOLD (dipakai untuk pertanyaan
            non-Indonesia, yang skornya sistematis lebih rendah — lihat chain._gate_applies)

    Returns:
        Tuple (chunks, top_score). `chunks` adalah list dict {text, file, page} (kosong kalau
        vector store kosong atau skor top-1 di bawah SIMILARITY_THRESHOLD), `top_score` adalah
        skor top-1 (0.0 kalau vector store kosong) — dikembalikan juga saat di-gate, supaya bisa
        dicatat untuk kalibrasi.
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

    # Query vector store lokal. Minta RETRIEVAL_FETCH_K kandidat (bukan langsung TOP_K) —
    # lihat komentar RETRIEVAL_FETCH_K di config.py. Dipangkas ke TOP_K di bawah, setelah
    # difilter skor.
    results = query_vector_store(question_vector, top_k=RETRIEVAL_FETCH_K)

    if not results:
        logger.warning("Tidak ada hasil retrieval dari vector store lokal. Database mungkin kosong.")
        return [], 0.0

    # Gate relevansi pada skor TOP-1 (query_vector_store mengurutkan menurun): kalau bahkan chunk
    # terbaik pun di bawah threshold, pertanyaan ini hampir pasti tidak ada hubungannya dengan
    # dokumen — jangan panggil LLM sama sekali. Konteks yang dikirim ke LLM tetap top-K biasa
    # (bukan difilter per-chunk, supaya tidak mengurangi recall untuk jawaban yang butuh beberapa chunk).
    top_score = results[0]["score"]
    if apply_gate and top_score < SIMILARITY_THRESHOLD:
        logger.info(
            f"Retrieval di-gate: skor top-1 {top_score:.3f} < SIMILARITY_THRESHOLD={SIMILARITY_THRESHOLD}. "
            "Kemungkinan pertanyaan di luar topik dokumen."
        )
        return [], top_score

    relevant_results = results[:TOP_K]

    # Format hasil
    context_chunks = []
    for result in relevant_results:
        context_chunks.append(
            {
                "text": result.get("text", ""),
                "file": result.get("file", "unknown"),
                "page": result.get("page", 0),
            }
        )
        logger.debug(
            f"Retrieved chunk dari {result.get('file')} hal. {result.get('page')} "
            f"(score: {result['score']:.3f})"
        )

    return context_chunks, top_score
