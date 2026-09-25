"""
RAG chain: menggabungkan retrieval + Gemini LLM untuk menghasilkan jawaban.
Menggunakan google-genai SDK (pengganti google-generativeai yang sudah deprecated).
"""

import logging
import re
import time
from typing import Any

from google import genai
from google.genai import types as genai_types

from app.config import GEMINI_API_KEY, LLM_MODEL, MAX_QUESTION_LENGTH
from app.rag.retrieval import retrieve_context
from app.rag.prompts import REFUSAL_NOT_FOUND, REFUSAL_OFF_TOPIC, build_prompt, build_rewrite_prompt

logger = logging.getLogger(__name__)

# Inisialisasi Gemini client (SDK baru)
_client = genai.Client(api_key=GEMINI_API_KEY)

# Dipakai saat index kosong DAN saat skor top-1 di bawah SIMILARITY_THRESHOLD (gate, tanpa
# memanggil LLM). Diawali kalimat penolakan baku supaya konsisten dari sudut pandang user dan
# dikenali sebagai penolakan oleh _is_refusal().
NO_CONTEXT_MESSAGE = (
    f"{REFUSAL_NOT_FOUND} Saya hanya bisa menjawab pertanyaan seputar AD/ART dan SOP organisasi — "
    "coba tulis pertanyaannya dengan lebih lengkap."
)
RATE_LIMIT_MESSAGE = "Maaf, sedang banyak yang bertanya. Silakan coba lagi dalam beberapa menit."

# Jumlah item history terakhir yang dipakai untuk query rewriting (menjaga prompt tetap pendek)
REWRITE_HISTORY_WINDOW = 4

# --- Jalan pintas sapaan / terima kasih ---
# Sapaan dan ucapan terima kasih adalah input "tak relevan" yang paling umum. Ditangani tanpa embed
# dan tanpa LLM. Regex ditambatkan ke SELURUH pesan (^...$), jadi pertanyaan sungguhan yang kebetulan
# diawali sapaan ("halo, apa denda telat kas?") TIDAK tertangkap dan tetap diproses normal.
_ADDRESS = r"(?:\s+(?:kak|kakak|min|admin|bang|mas|mbak|bot|semua|teman|pengurus|osui|mahawaditra))*"
_GREETING_RE = re.compile(
    r"^\s*(?:halo+|hallo+|hai+|hi+|hey+|hello+|selamat\s+(?:pagi|siang|sore|malam)"
    r"|assalamu'?alaikum(?:\s+wr\.?\s*wb\.?)?|permisi)" + _ADDRESS + r"[\s!.,?]*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\s*(?:(?:oke|ok|siap|baik)[\s,]+)?(?:terima\s*kasih|makasih|makasi|thanks?|thank\s+you|thx|tks|trims)"
    r"(?:\s+(?:banyak|sekali|banget|ya|yaa|atas\s+bantuannya|kak|kakak|min|admin|bang|mas|mbak|bot))*[\s!.,?]*$",
    re.IGNORECASE,
)
GREETING_REPLY = (
    "Halo! Saya asisten AD/ART & SOP OSUI Mahawaditra. Silakan tanyakan seputar peraturan organisasi, "
    "misalnya denda keterlambatan, masa jabatan pengurus, atau prosedur peminjaman alat."
)
THANKS_REPLY = "Sama-sama! Kalau masih ada pertanyaan soal AD/ART atau SOP, silakan tanya lagi."

# Penanda sumber yang diminta SYSTEM_PROMPT di akhir jawaban: [[SUMBER: 1, 3]] atau [[SUMBER: -]].
_SOURCE_MARKER_RE = re.compile(r"\[\[\s*SUMBER\s*:\s*([^\[\]]*?)\s*\]\]", re.IGNORECASE)


# Kata umum bahasa Inggris (yang hampir tidak pernah muncul di kalimat Indonesia) untuk mendeteksi
# pertanyaan berbahasa Inggris. "a"/"i" sengaja tidak dimasukkan (muncul sebagai butir "a." dsb.).
_EN_STOPWORDS = frozenset(
    "the is are was were what how when where who why which do does did can could should would we you "
    "my our your to of in on for and or if be it this that there with about from need want please tell me".split()
)


def _gate_applies(question: str) -> bool:
    """
    Gate relevansi (SIMILARITY_THRESHOLD) hanya untuk pertanyaan yang tampak berbahasa Indonesia.

    Vector store berisi teks Indonesia, jadi kemiripan lintas-bahasa sistematis lebih rendah: pertanyaan
    Inggris yang SAH terukur 0.768-0.834 (mis. "how long is the chairman's term" = 0.768, di bawah
    threshold), padahal anggota yang kurang fasih Bahasa Indonesia adalah pengguna yang sengaja
    didukung (lihat SYSTEM_PROMPT). Untuk pertanyaan non-Indonesia, gate dilewati dan Gemini yang
    memutuskan; kalau ia menolak, sumber tetap disembunyikan lewat penanda [[SUMBER: -]].
    """
    if any(ch.isalpha() and ord(ch) > 0x24F for ch in question):
        return False  # aksara non-Latin (Mandarin, Arab, dst.)
    tokens = re.findall(r"[a-z']+", question.lower())
    if not tokens:
        return True
    hits = sum(1 for t in tokens if t in _EN_STOPWORDS)
    return not (hits >= 2 or hits / len(tokens) >= 0.34)


def _smalltalk_reply(question: str) -> str | None:
    """Balasan tetap untuk sapaan/terima kasih murni, atau None kalau bukan keduanya."""
    if _GREETING_RE.match(question):
        return GREETING_REPLY
    if _THANKS_RE.match(question):
        return THANKS_REPLY
    return None


def _extract_source_marker(text: str) -> tuple[str, list[int] | None]:
    """
    Buang SEMUA penanda [[SUMBER: ...]] dari teks (supaya tidak pernah bocor ke UI, tombol Salin,
    atau riwayat percakapan) dan kembalikan nomor dari penanda TERAKHIR.

    Returns:
        (teks_bersih, nomor) — nomor adalah list int terurut ([] untuk "[[SUMBER: -]]"), atau None
        kalau tidak ada penanda sama sekali.
    """
    matches = list(_SOURCE_MARKER_RE.finditer(text))
    if not matches:
        return text.strip(), None
    numbers = sorted({int(n) for n in re.findall(r"\d+", matches[-1].group(1))})
    return _SOURCE_MARKER_RE.sub("", text).strip(), numbers


def _is_refusal(answer: str) -> bool:
    """True kalau jawaban diawali salah satu kalimat penolakan baku (spasi/huruf besar diabaikan)."""
    normalized = re.sub(r"\s+", " ", answer).strip().lower()
    return normalized.startswith((REFUSAL_NOT_FOUND.lower(), REFUSAL_OFF_TOPIC.lower()))


def _dedup_sources(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Daftar sumber {file, page} tanpa duplikat, urutan pertama muncul dipertahankan."""
    seen: set[tuple[str, int]] = set()
    sources = []
    for chunk in chunks:
        key = (chunk["file"], chunk["page"])
        if key not in seen:
            seen.add(key)
            sources.append({"file": chunk["file"], "page": chunk["page"]})
    return sources


def _resolve_answer_and_sources(raw_answer: str, chunks: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """
    Pisahkan jawaban dari penanda sumber dan tentukan sumber mana yang ditampilkan.

    Urutan keputusan (fallback berlapis, supaya kegagalan model mengikuti format tidak pernah
    membuat jawaban valid kehilangan sitasi):
      1. Jawaban penolakan berkalimat baku -> tanpa sumber (apa pun isi penandanya).
      2. Penanda berisi nomor valid -> hanya sumber yang dikutip itu.
      3. Penanda eksplisit "tidak ada sumber yang dipakai" ([[SUMBER: -]]) -> tanpa sumber. Ini yang
         mengenali penolakan dalam bahasa lain (kalimatnya tidak baku, tapi penandanya tetap).
      4. Selain itu (penanda hilang/di luar rentang) -> semua sumber ter-dedup, seperti perilaku
         sebelum penanda ada.
    """
    answer, numbers = _extract_source_marker(raw_answer)

    if _is_refusal(answer):
        return answer, []

    if numbers is not None:
        cited = [chunks[n - 1] for n in numbers if 1 <= n <= len(chunks)]
        if cited:
            logger.info(f"Sumber dikutip Gemini: {numbers}")
            return answer, _dedup_sources(cited)
        if not numbers:
            logger.info("Gemini menandai tidak ada sumber yang dipakai ([[SUMBER: -]]).")
            return answer, []

    logger.warning(
        "Penanda [[SUMBER]] hilang/tak terbaca/di luar rentang pada jawaban non-penolakan — "
        "fallback menampilkan semua sumber."
    )
    return answer, _dedup_sources(chunks)


def get_answer(question: str, history: list[dict] | None = None) -> dict[str, Any]:
    """
    Pipeline RAG lengkap: sapaan? → rewrite query (jika ada history) → retrieval + gate relevansi
    → build prompt → LLM → jawaban + sumber yang benar-benar dikutip.

    Args:
        question: Pertanyaan dari user

    Returns:
        Dict berisi {answer, sources, retrieval_query, status, top_score}
        sources adalah list of {file, page}
        retrieval_query adalah query standalone yang dipakai untuk retrieval (untuk logging/debug)
        status: "ok" | "rate_limited" | "gated" (di-gate sebelum LLM) | "smalltalk" (sapaan/terima kasih)
        top_score: skor similarity top-1 (None untuk smalltalk) — dicatat untuk kalibrasi threshold
    """
    # 0. Sapaan/terima kasih murni: balas tetap, tanpa embed dan tanpa LLM
    smalltalk = _smalltalk_reply(question)
    if smalltalk is not None:
        return {
            "answer": smalltalk,
            "sources": [],
            "retrieval_query": question,
            "status": "smalltalk",
            "top_score": None,
        }

    # 1. Rewrite pertanyaan jadi standalone query untuk retrieval (hanya jika ada history)
    retrieval_query = _rewrite_query_for_retrieval(question, history)

    # 2. Retrieve context dari vector store lokal. `chunks` kosong kalau vector store kosong atau
    #    skor top-1 di bawah SIMILARITY_THRESHOLD — dua-duanya: jangan panggil LLM sama sekali.
    chunks, top_score = retrieve_context(retrieval_query, apply_gate=_gate_applies(question))

    if not chunks:
        return {
            "answer": NO_CONTEXT_MESSAGE,
            "sources": [],
            "retrieval_query": retrieval_query,
            "status": "gated",
            "top_score": top_score,
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

    if llm_status != "ok":
        return {
            "answer": answer_text,
            "sources": [],
            "retrieval_query": retrieval_query,
            "status": llm_status,
            "top_score": top_score,
        }

    if not answer_text:
        # response.text bisa None (mis. diblokir safety filter) — perlakukan sebagai error
        # (ditangkap handler generik di main.py -> 503), bukan meneruskan None ke client.
        raise RuntimeError("Gemini mengembalikan jawaban kosong.")

    # 6. Pisahkan penanda sumber dari jawaban dan tentukan sumber yang ditampilkan
    answer, sources = _resolve_answer_and_sources(answer_text, chunks)

    return {
        "answer": answer,
        "sources": sources,
        "retrieval_query": retrieval_query,
        "status": "ok",
        "top_score": top_score,
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
