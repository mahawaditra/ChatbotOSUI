"""
FastAPI application — entry point utama chatbot RAG OSUI Mahawaditra.

Endpoints:
    GET  /          → Web interface (index.html)
    GET  /health    → Health check untuk Render
    POST /api/chat  → RAG chatbot
    POST /admin/reindex → Reindex semua dokumen PDF
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

from app.config import ADMIN_KEY, LLM_MODEL
from app.rag.chain import get_answer, RATE_LIMIT_MESSAGE
from app.rag.indexing import run_indexing

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Folder untuk menyimpan log request harian
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# --- FastAPI app ---
app = FastAPI(
    title="RAG Chatbot OSUI Mahawaditra",
    description="Chatbot yang menjawab pertanyaan berdasarkan AD/ART dan SOP organisasi.",
    version="1.0.0",
)

# Serve static files (index.html)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Batas panjang input
MAX_QUESTION_LENGTH = 500
MAX_HISTORY_ITEM_LENGTH = 800


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
    history: list[HistoryItem] = []  # Riwayat chat dari frontend

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
def _log_request(question: str, answer: str, sources: list, latency_ms: float, status: str) -> None:
    """Menyimpan log request ke file JSON harian."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
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
async def chat(request: ChatRequest):
    """
    Endpoint utama chatbot RAG.
    Menerima pertanyaan user dan mengembalikan jawaban beserta sumber dokumen.
    """
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="Pertanyaan tidak boleh kosong.")

    start_time = time.perf_counter()
    status = "success"

    try:
        history = [{"role": h.role, "content": h.content} for h in request.history]
        result = get_answer(request.message, history=history)
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
            question=request.message,
            answer=locals().get("answer", ""),
            sources=locals().get("sources", []),
            latency_ms=latency_ms,
            status=status,
        )


@app.post("/admin/reindex")
async def reindex(x_admin_key: str = Header(None)):
    """
    Endpoint admin untuk reindex semua dokumen PDF.
    Membutuhkan header X-Admin-Key yang sesuai dengan ADMIN_KEY di environment.
    """
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    logger.info("Reindex dimulai oleh admin...")
    try:
        result = run_indexing()
        logger.info(f"Reindex selesai: {result}")
        return {"status": "success", **result}

    except FileNotFoundError as e:
        logger.error(f"Reindex gagal - file tidak ditemukan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Reindex gagal: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={"status": "error", "message": str(e)},
        )
