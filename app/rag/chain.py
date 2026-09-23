"""
RAG chain: menggabungkan retrieval + Gemini LLM untuk menghasilkan jawaban.
Menggunakan google-genai SDK (pengganti google-generativeai yang sudah deprecated).
"""

import logging
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import GEMINI_API_KEY, LLM_MODEL, MAX_QUESTION_LENGTH
from app.rag.retrieval import retrieve_context
from app.rag.prompts import build_prompt, build_rewrite_prompt

logger = logging.getLogger(__name__)

# Inisialisasi Gemini client (SDK baru)
_client = genai.Client(api_key=GEMINI_API_KEY)

# Dipakai baik saat index benar-benar kosong maupun saat semua chunk hasil retrieval
# di bawah SIMILARITY_THRESHOLD (pertanyaan di luar topik dokumen) — kalimat yang sama
# dengan instruksi penolakan di SYSTEM_PROMPT, supaya konsisten dari sudut pandang user.
NO_CONTEXT_MESSAGE = "Maaf, informasi tersebut tidak ditemukan dalam dokumen AD/ART atau SOP organisasi."
RATE_LIMIT_MESSAGE = "Maaf, sedang banyak yang bertanya. Silakan coba lagi dalam beberapa menit."

# Jumlah item history terakhir yang dipakai untuk query rewriting (menjaga prompt tetap pendek)
REWRITE_HISTORY_WINDOW = 4


def get_answer(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    """
    Pipeline RAG lengkap: rewrite query (jika ada history) → retrieval → build prompt → LLM → jawaban.

    Args:
        question: Pertanyaan dari user

    Returns:
        Dict berisi {answer, sources, retrieval_query}
        sources adalah list of {file, page}
        retrieval_query adalah query standalone yang dipakai untuk retrieval (untuk logging/debug)
    """
    # 1. Rewrite pertanyaan jadi standalone query untuk retrieval (hanya jika ada history)
    retrieval_query = _rewrite_query_for_retrieval(question, history)

    # 2. Retrieve context dari vector store lokal
    chunks = retrieve_context(retrieval_query)

    if not chunks:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
            "retrieval_query": retrieval_query,
            "status": "ok",
        }

    # 3. Bangun context string dari chunks
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Sumber {i}: {chunk['file']}, Halaman {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 4. Bangun prompt (pakai `question` asli, bukan hasil rewrite, agar jawaban & sitasi
    #    tetap mengikuti kalimat asli user)
    prompt = build_prompt(context=context, question=question, history=history)

    # 5. Panggil Gemini LLM dengan retry 1x jika rate limit
    answer_text, llm_status = _call_llm_with_retry(prompt)

    # 6. Format sources (deduplikasi file+page yang sama)
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
        "retrieval_query": retrieval_query,
        "status": llm_status,
    }


def _rewrite_query_for_retrieval(question: str, history: list[dict] | None) -> str:
    """
    Mengubah pertanyaan lanjutan (yang mungkin mengandalkan konteks percakapan sebelumnya,
    mis. "terus kalau telat gimana?") menjadi standalone query untuk retrieval, dengan
    memanggil LLM 1x. Tidak pernah melempar exception — kalau gagal, fallback ke `question` asli
    supaya langkah opsional ini tidak pernah menggagalkan alur jawaban utama.

    Args:
        question: Pertanyaan terbaru dari user
        history: Riwayat percakapan, atau None/kosong untuk pertanyaan pertama

    Returns:
        Standalone query untuk dipakai di retrieve_context(); `question` asli kalau
        tidak ada history atau rewrite gagal
    """
    if not history:
        return question

    try:
        prompt = build_rewrite_prompt(question, history[-REWRITE_HISTORY_WINDOW:])
        response = _client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=150,
            ),
        )
        rewritten = (response.text or "").strip()[:MAX_QUESTION_LENGTH]
        if rewritten:
            logger.debug(f"Query rewrite: '{question}' -> '{rewritten}'")
            return rewritten
        return question

    except Exception as e:
        logger.warning(f"Query rewriting gagal, pakai pertanyaan asli: {e}")
        return question


def _call_llm_with_retry(prompt: str) -> tuple[str, str]:
    """
    Memanggil Gemini LLM dengan 1x retry jika terjadi rate limit (429).

    Args:
        prompt: Prompt lengkap yang sudah mengandung konteks dan pertanyaan

    Returns:
        Tuple (teks_jawaban, status) — status adalah "ok" atau "rate_limited". Dipakai
        sebagai field eksplisit alih-alih main.py membandingkan teks jawaban persis dengan
        RATE_LIMIT_MESSAGE (rapuh: jawaban asli dari LLM yang kebetulan sama persis dengan
        kalimat itu akan salah terklasifikasi sebagai rate-limit).
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
            return response.text, "ok"

        except Exception as e:
            error_str = str(e).lower()
            is_rate_limit = "429" in error_str or "resource exhausted" in error_str or "quota" in error_str
            is_unavailable = "503" in error_str or "unavailable" in error_str

            if (is_rate_limit or is_unavailable) and attempt == 0:
                logger.warning(f"Rate limit/unavailable pada attempt {attempt + 1}, retry dalam 2 detik... ({e})")
                time.sleep(2)
            elif (is_rate_limit or is_unavailable) and attempt == 1:
                logger.error(f"Rate limit tetap terjadi setelah retry: {e}")
                return RATE_LIMIT_MESSAGE, "rate_limited"
            else:
                logger.error(f"Error tak terduga saat memanggil Gemini: {e}")
                raise
