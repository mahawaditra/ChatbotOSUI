"""
FastAPI application — entry point utama chatbot RAG OSUI Mahawaditra.

Endpoints:
    GET  /          → Web interface (index.html)
    GET  /health    → Health check untuk Render
    POST /api/chat  → RAG chatbot
    POST /admin/reindex → Reindex semua dokumen PDF
"""

import hmac
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from upstash_ratelimit.asyncio import Ratelimit, SlidingWindow
from upstash_redis.asyncio import Redis

from app.config import (
    ADMIN_KEY,
    LLM_MODEL,
    MAX_QUESTION_LENGTH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
)
from app.rag.chain import get_answer, RATE_LIMIT_MESSAGE
from app.rag.indexing import run_indexing

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Folder untuk menyimpan log request harian.
# Best-effort: di platform serverless (mis. Vercel) filesystem read-only di luar /tmp,
# jadi mkdir ini bisa gagal — jangan sampai itu bikin seluruh app gagal start.
LOG_DIR = Path("logs")
try:
    LOG_DIR.mkdir(exist_ok=True)
except OSError as e:
    logger.warning(f"Tidak bisa membuat folder log ({LOG_DIR}), file-based logging dinonaktifkan: {e}")


# --- FastAPI app ---
app = FastAPI(
    title="RAG Chatbot OSUI Mahawaditra",
    description="Chatbot yang menjawab pertanyaan berdasarkan AD/ART dan SOP organisasi.",
    version="1.0.0",
)

# Serve static files (index.html & dokumen PDF)
STATIC_DIR = Path(__file__).parent / "static"
DOKUMEN_DIR = Path(__file__).parent.parent / "dokumen"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if DOKUMEN_DIR.exists():
    app.mount("/dokumen", StaticFiles(directory=str(DOKUMEN_DIR)), name="dokumen")

# Batas panjang input riwayat chat (MAX_QUESTION_LENGTH untuk `message` ada di app/config.py)
MAX_HISTORY_ITEM_LENGTH = 800
MAX_HISTORY_ITEMS = 6  # cermin dari MAX_HISTORY*2 di frontend (app/static/index.html)

# Rate limiter per-IP untuk /api/chat, disimpan di Upstash Redis (bukan in-memory) supaya
# tetap benar walau di-deploy sebagai serverless function (tiap invocation bisa instance baru
# tanpa shared memory — in-memory counter akan reset/tidak konsisten antar instance).
_chat_ratelimit = Ratelimit(
    redis=Redis.from_env(),
    limiter=SlidingWindow(max_requests=RATE_LIMIT_REQUESTS, window=RATE_LIMIT_WINDOW_SECONDS),
    prefix="ratelimit:chat",
)


def _get_client_ip(http_request: Request) -> str:
    """IP asli client di belakang proxy (Vercel/Render mengisi X-Forwarded-For)."""
    forwarded = http_request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return http_request.client.host if http_request.client else "unknown"


def _is_origin_allowed(http_request: Request) -> bool:
    """
    Tolak request lintas-situs dari browser (Origin header ada tapi host-nya beda dari host
    request ini sendiri). Dicek terhadap host request sendiri (bukan daftar domain hardcoded)
    supaya otomatis cocok di domain produksi maupun preview URL Vercel mana pun.
    Origin yang tidak ada (banyak client non-browser/same-origin yang wajar) dianggap lolos —
    rate limiting adalah lapisan pertahanan berikutnya untuk trafik non-browser.
    """
    origin = http_request.headers.get("origin")
    if not origin:
        return True
    try:
        origin_host = urlparse(origin).hostname
    except ValueError:
        return False
    return origin_host == http_request.url.hostname


# --- Pydantic models ---
class HistoryItem(BaseModel):
    role: str      # "user" atau "bot"
    content: str

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ('user', 'bot'):
            raise ValueError("role harus 'user' atau 'bot'")
        return v

    @field_validator('content')
    @classmethod
    def sanitize_content(cls, v: str) -> str:
        # Potong jika terlalu panjang
        return v[:MAX_HISTORY_ITEM_LENGTH]


class ChatRequest(BaseModel):
    message: str
    # Riwayat chat dari frontend — dibatasi max_length supaya tidak bisa dipakai untuk
    # menggembungkan ukuran prompt/biaya panggilan LLM secara sewenang-wenang (frontend
    # sendiri sudah membatasi ke jumlah yang sama, ini adalah penegakan sisi server-nya).
    history: list[HistoryItem] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Pertanyaan tidak boleh kosong.")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Pertanyaan terlalu panjang (maks {MAX_QUESTION_LENGTH} karakter).")
        return v


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


# --- Helper: log request ---
def _log_request(
    question: str,
    answer: str,
    sources: list,
    latency_ms: float,
    status: str,
    retrieval_query: str = "",
) -> None:
    """Menyimpan log request ke file JSON harian."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        # Query standalone hasil query rewriting yang dipakai untuk retrieval — berguna untuk
        # mengecek manual apakah rewrite membantu pertanyaan lanjutan di percakapan multi-turn.
        "retrieval_query": retrieval_query,
        "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer,
        "sources": [f"{s['file']} hal. {s['page']}" for s in sources],
        "chunks_retrieved": len(sources),
        "latency_ms": round(latency_ms, 2),
        "llm_model": LLM_MODEL,
        "status": status,
    }

    log_file = LOG_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.json"
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Gagal menulis log: {e}")


# --- Endpoints ---

@app.get("/Logo.png", include_in_schema=False)
async def serve_logo():
    """Serve logo OSUI dari root project."""
    logo_path = Path(__file__).parent.parent / "Logo.png"
    if not logo_path.exists():
        raise HTTPException(status_code=404, detail="Logo tidak ditemukan.")
    return FileResponse(str(logo_path), media_type="image/png")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    """Menyajikan halaman web interface chatbot."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health_check():
    """Health check endpoint untuk Render."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, http_request: Request):
    """
    Endpoint utama chatbot RAG.
    Menerima pertanyaan user dan mengembalikan jawaban beserta sumber dokumen.
    """
    if not chat_request.message.strip():
        raise HTTPException(status_code=422, detail="Pertanyaan tidak boleh kosong.")

    if not _is_origin_allowed(http_request):
        raise HTTPException(status_code=403, detail="Origin tidak diizinkan.")

    client_ip = _get_client_ip(http_request)
    try:
        rl_allowed = (await _chat_ratelimit.limit(client_ip)).allowed
    except Exception as e:
        # Fail-open: kalau Upstash Redis sendiri bermasalah/salah konfigurasi, jangan sampai
        # itu membuat seluruh /api/chat down untuk semua orang — itu risiko yang lebih besar
        # daripada sementara tidak ada rate limiting.
        logger.warning(f"Rate limiter error, melewatkan pengecekan rate limit: {e}")
        rl_allowed = True

    if not rl_allowed:
        return JSONResponse(
            status_code=429,
            content={
                "answer": "Terlalu banyak permintaan. Silakan coba lagi sebentar lagi.",
                "sources": [],
            },
        )

    start_time = time.perf_counter()
    status = "success"

    try:
        history = [{"role": h.role, "content": h.content} for h in chat_request.history]
        result = get_answer(chat_request.message, history=history)
        answer = result["answer"]
        sources = result["sources"]

        # Cek apakah jawaban adalah pesan rate limit
        if answer == RATE_LIMIT_MESSAGE:
            status = "rate_limit"
            return JSONResponse(
                status_code=503,
                content={"answer": answer, "sources": []},
            )

        return ChatResponse(answer=answer, sources=sources)

    except Exception as e:
        status = "error"
        logger.error(f"Error pada /api/chat: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "answer": "Maaf, terjadi kesalahan pada sistem. Silakan coba lagi.",
                "sources": [],
            },
        )

    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        _log_request(
            question=chat_request.message,
            answer=locals().get("answer", ""),
            sources=locals().get("sources", []),
            latency_ms=latency_ms,
            status=status,
            retrieval_query=locals().get("result", {}).get("retrieval_query", ""),
        )


@app.post("/admin/reindex")
async def reindex(x_admin_key: str = Header(None)):
    """
    Endpoint admin untuk reindex semua dokumen PDF.
    Membutuhkan header X-Admin-Key yang sesuai dengan ADMIN_KEY di environment.
    """
    if not hmac.compare_digest(x_admin_key or "", ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info("Reindex dimulai oleh admin...")
    try:
        result = run_indexing()
        logger.info(f"Reindex selesai: {result}")
        return {"status": "success", **result}

    except FileNotFoundError as e:
        logger.error(f"Reindex gagal - file tidak ditemukan: {e}")
        raise HTTPException(status_code=500, detail="Reindex gagal. Periksa log server untuk detail.")

    except (OSError, PermissionError) as e:
        logger.error(f"Reindex gagal - tidak bisa menulis vector store lokal: {e}")
        raise HTTPException(
            status_code=500,
            detail=(
                "Reindex gagal: tidak bisa menulis vector store lokal (kemungkinan filesystem "
                "read-only di instance ini, mis. Vercel). Jalankan `python scripts/reindex.py` "
                "secara lokal, lalu commit dan deploy ulang hasilnya."
            ),
        )

    except Exception as e:
        logger.error(f"Reindex gagal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Reindex gagal. Periksa log server untuk detail.")
