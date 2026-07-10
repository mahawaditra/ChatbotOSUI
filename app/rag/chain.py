"""
RAG chain: menggabungkan retrieval + Gemini LLM untuk menghasilkan jawaban.
Menggunakan google-genai SDK (pengganti google-generativeai yang sudah deprecated).
"""

import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import GEMINI_API_KEY, LLM_MODEL
from app.rag.retrieval import retrieve_context
from app.rag.prompts import build_prompt

logger = logging.getLogger(__name__)

# Inisialisasi Gemini client (SDK baru)
_client = genai.Client(api_key=GEMINI_API_KEY)

EMPTY_DB_MESSAGE = "Sistem belum memiliki dokumen. Silakan hubungi pengurus."
RATE_LIMIT_MESSAGE = "Maaf, sedang banyak yang bertanya. Silakan coba lagi dalam beberapa menit."


def get_answer(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    """
    Pipeline RAG lengkap: retrieval → build prompt → LLM → return jawaban.

    Args:
        question: Pertanyaan dari user

    Returns:
        Dict berisi {answer, sources}
        sources adalah list of {file, page}
    """
    # 1. Retrieve context dari Upstash Vector
    chunks = retrieve_context(question)

    if not chunks:
        return {
            "answer": EMPTY_DB_MESSAGE,
            "sources": [],
        }

    # 2. Bangun context string dari chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Sumber {i}: {chunk['file']}, Halaman {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 3. Bangun prompt
    prompt = build_prompt(context=context, question=question, history=history)

    # 4. Panggil Gemini LLM dengan retry 1x jika rate limit
    answer_text = _call_llm_with_retry(prompt)

    # 5. Format sources (deduplikasi file+page yang sama)
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk["file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["file"], "page": chunk["page"]})

    return {
        "answer": answer_text,
        "sources": sources,
    }


def _call_llm_with_retry(prompt: str) -> str:
    """
    Memanggil Gemini LLM dengan 1x retry jika terjadi rate limit (429).

    Args:
        prompt: Prompt lengkap yang sudah mengandung konteks dan pertanyaan

    Returns:
        Teks jawaban dari LLM
    """
    for attempt in range(2):  # maksimal 2 percobaan (1 retry)
        try:
            response = _client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                ),
            )
            return response.text

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "resource exhausted" in error_str or "quota" in error_str
            is_unavailable = "503" in error_str or "unavailable" in error_str

            if (is_rate_limit or is_unavailable) and attempt == 0:
                logger.warning(f"Rate limit/unavailable pada attempt {attempt + 1}, retry dalam 2 detik... ({e})")
                time.sleep(2)
            elif (is_rate_limit or is_unavailable) and attempt == 1:
                logger.error(f"Rate limit tetap terjadi setelah retry: {e}")
                return RATE_LIMIT_MESSAGE
            else:
                logger.error(f"Error tak terduga saat memanggil Gemini: {e}")
                raise
