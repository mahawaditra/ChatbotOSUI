"""
FastAPI application — entry point utama chatbot RAG OSUI Mahawaditra.

Endpoints:
    GET  /          → Web interface (index.html)
    GET  /health    → Health check untuk Render
    POST /api/chat  → RAG chatbot
    POST /admin/reindex → Reindex semua dokumen PDF
"""

import asyncio
import hmac
import json
import logging
import mimetypes
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from upstash_ratelimit.asyncio import Ratelimit, SlidingWindow
from upstash_redis.asyncio import Redis

from app.config import (
    ADMIN_KEY,
    LLM_MODEL,
    MAX_QUESTION_LENGTH,
    RATE_LIMIT_REQUESTS,
    RATE_LIMIT_WINDOW_SECONDS,
    validate_server_config,
)
from app.rag.chain import get_answer
from app.rag.indexing import run_indexing

# Menegakkan secret yang dibutuhkan server sungguhan (ADMIN_KEY, Upstash Redis) — dipisah
# dari app/config.py supaya mengimpor config lewat jalur lain (mis. scripts/reindex.py)
# tidak ikut wajib menyediakan secret yang tidak dipakainya. Lihat app/config.py.
validate_server_config()

# StaticFiles menebak Content-Type dari tabel MIME sistem, yang di container Vercel tidak dijamin
# memuat dua ekstensi ini. .mjs WAJIB text/javascript (browser menolak modul dengan MIME lain —
# dipakai viewer PDF.js di app/static/vendor/), .webmanifest untuk "Tambah ke Layar Utama".
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/manifest+json", ".webmanifest")

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

# Mencegah dua /admin/reindex berjalan bersamaan (lihat reindex() di bawah).
_reindex_lock = threading.Lock()


def _get_client_ip(http_request: Request) -> str:
    """
    IP asli client di belakang proxy. Vercel MENIMPA (bukan menambah/append) header
    X-Forwarded-For dengan IP client yang benar-benar diamatinya dan membuang nilai apapun
    yang dikirim client — jadi mengambil entri pertama di sini aman dari spoofing SELAMA
    deploy di Vercel tanpa proxy/CDN lain (mis. Cloudflare) di depannya. Ini jaminan
    platform-specific yang bisa berubah diam-diam kalau proxy lain ditambahkan di depan, atau
    kalau deployment pindah ke platform lain (Render/Railway/Docker) — perilaku X-Forwarded-For
    di sana BELUM diverifikasi sama amannya, jangan asumsikan otomatis sama.
    """
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


# Batas ukuran body mentah untuk POST /api/chat, ditegakkan di ChatGuardMiddleware SEBELUM
# body dibaca/di-parse sama sekali. Cukup longgar untuk message 500 karakter + 6 item histori
# (masing-masing maks 800 karakter) plus overhead JSON, jauh di bawah level abuse.
MAX_CHAT_BODY_BYTES = 10_000

# Berapa lama menunggu Upstash Redis merespons cek rate limit sebelum menyerah dan fail-open.
# Library upstash_redis mengonstruksi HTTP client-nya dengan timeout=None (tanpa batas) dan
# tidak menyediakan cara resmi untuk mengubahnya lewat Redis.from_env()/Ratelimit — jadi batas
# waktu ditegakkan di sisi pemanggilan lewat asyncio.wait_for, bukan di client itu sendiri.
# Tanpa ini, Upstash yang lambat/menggantung tidak akan gagal cepat seperti yang diasumsikan
# alasan fail-open di bawah — dia akan menggantung sampai Vercel membunuh seluruh request.
_RATELIMIT_CHECK_TIMEOUT_SECONDS = 5.0


class ChatGuardMiddleware(BaseHTTPMiddleware):
    """
    Middleware pengecekan body-size + Origin + rate-limit untuk POST /api/chat, dijalankan
    SEBELUM FastAPI/Pydantic membaca & memvalidasi body request.

    Kenapa ini middleware dan bukan kode di dalam handler chat(): Pydantic memvalidasi body
    (ChatRequest) sebagai bagian dari resolusi dependency SEBELUM isi fungsi chat() sempat
    berjalan. Kalau pengecekan Origin/rate-limit ada di dalam chat(), request yang sengaja
    dibuat gagal validasi (mis. message > 500 karakter, atau history > 6 item) mendapat 422
    tanpa PERNAH menyentuh kedua pengecekan itu — artinya siapa pun bisa mem-flood endpoint
    ini tanpa batas dan gratis selama payload-nya "salah". Middleware berjalan di level ASGI,
    sebelum body dibaca sama sekali, jadi tidak bisa dilewati dengan cara ini.
    """

    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != "/api/chat":
            return await call_next(request)

        content_length = request.headers.get("content-length")
        if content_length is not None and content_length.isdigit() and int(content_length) > MAX_CHAT_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"answer": "Permintaan terlalu besar.", "sources": []},
            )

        if not _is_origin_allowed(request):
            return JSONResponse(
                status_code=403,
                content={"answer": "Origin tidak diizinkan.", "sources": []},
            )

        client_ip = _get_client_ip(request)
        try:
            rl_allowed = (
                await asyncio.wait_for(
                    _chat_ratelimit.limit(client_ip), timeout=_RATELIMIT_CHECK_TIMEOUT_SECONDS
                )
            ).allowed
        except Exception as e:
            # Fail-open: kalau Upstash Redis sendiri bermasalah/salah konfigurasi/lambat, jangan
            # sampai itu membuat seluruh /api/chat down untuk semua orang — itu risiko yang lebih
            # besar daripada sementara tidak ada rate limiting. Level ERROR (bukan warning) karena
            # ini artinya rate limiting sedang NONAKTIF total untuk semua user, bukan kejadian kecil.
            logger.error(f"Rate limiter error/timeout, rate limiting DINONAKTIFKAN sementara: {e}")
            rl_allowed = True

        if not rl_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "answer": "Terlalu banyak permintaan. Silakan coba lagi sebentar lagi.",
                    "sources": [],
                },
            )

        return await call_next(request)


def _cache_control_for(path: str) -> str | None:
    """
    Cache-Control per jenis path. Tanpa ini Vercel memberi `max-age=0, must-revalidate` untuk semua
    respons dari function, jadi tiap kunjungan (di HP: boros data + cold start) mengulang request
    untuk logo, favicon, library PDF.js (~1,8 MB), dan PDF. Halaman `/` dan /api/* sengaja TIDAK
    di-cache (None).
    """
    if path.startswith("/static/vendor/"):
        # Folder ber-versi (mis. pdfjs-6.3.289): isinya tidak pernah berubah di bawah nama yang sama.
        return "public, max-age=31536000, immutable"
    if path.startswith("/static/") or path == "/manifest.webmanifest":
        return "public, max-age=86400"
    if path.startswith("/dokumen/"):
        # Dokumen diganti tahunan dengan nama file yang bisa sama; ETag tetap merevalidasi setelah 1 jam.
        return "public, max-age=3600"
    return None


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Header keamanan standar di setiap response. HSTS tidak perlu ditambahkan manual —
    Vercel sudah menerapkannya otomatis untuk domain *.vercel.app maupun custom domain.

    Catatan soal CSP script-src: index.html punya puluhan atribut event handler inline
    (onclick=/oninput= di markup statis sidebar, tombol, dsb.), bukan cuma satu blok <script>.
    Nonce CSP hanya berlaku untuk elemen <script>, TIDAK untuk atribut handler seperti itu —
    menghapusnya semua ke addEventListener() adalah refactor frontend besar di luar cakupan
    pass ini. 'unsafe-inline' di script-src karena itu adalah kompromi sadar, bukan oversight:
    nilai keamanan CSP di sini tetap nyata lewat frame-ancestors (anti-clickjacking) dan
    membatasi origin eksternal yang boleh dimuat, walau tidak menutup total XSS berbasis script.

    Pengecualian /dokumen/*.pdf: viewer PDF desktop memakai <iframe> same-origin. `X-Frame-Options:
    DENY` + `frame-ancestors 'none'` memblokir framing bahkan oleh halaman kita sendiri (dulu bikin
    viewer PDF di production kosong), jadi untuk path itu dipakai SAMEORIGIN. CSP tidak dikirim
    sama sekali untuk PDF: ini dokumen statis (tanpa script), dan `default-src` di respons PDF bisa
    mengganggu viewer bawaan browser. Clickjacking dari situs LAIN tetap tertutup oleh SAMEORIGIN.
    """

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if path.startswith("/dokumen/"):
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
        else:
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "frame-ancestors 'none'"
            )
        # Hanya untuk respons sukses; jangan meng-cache 404/500 dan jangan menimpa header yang
        # sudah diatur handler itu sendiri.
        cache_control = _cache_control_for(path)
        if cache_control and response.status_code in (200, 206) and "cache-control" not in response.headers:
            response.headers["Cache-Control"] = cache_control
        return response


app.add_middleware(ChatGuardMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


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
    top_score: float | None = None,
) -> None:
    """Menyimpan log request ke file JSON harian."""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        # Query standalone hasil query rewriting yang dipakai untuk retrieval — berguna untuk
        # mengecek manual apakah rewrite membantu pertanyaan lanjutan di percakapan multi-turn.
        "retrieval_query": retrieval_query,
        # Skor similarity top-1 — bahan kalibrasi SIMILARITY_THRESHOLD dari trafik nyata
        # (None untuk sapaan yang tidak melalui retrieval).
        "top_score": None if top_score is None else round(top_score, 4),
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


@app.get("/manifest.webmanifest", include_in_schema=False)
async def serve_manifest():
    """Manifest untuk "Tambah ke Layar Utama". Di root (bukan /static/) supaya scope "/" sah."""
    manifest_path = STATIC_DIR / "manifest.webmanifest"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Manifest tidak ditemukan.")
    return FileResponse(str(manifest_path), media_type="application/manifest+json")


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

    Pengecekan Origin dan rate-limit ada di ChatGuardMiddleware (jalan sebelum body di-parse
    Pydantic), bukan di sini — lihat docstring middleware tersebut untuk alasannya.
    """
    if not chat_request.message.strip():
        raise HTTPException(status_code=422, detail="Pertanyaan tidak boleh kosong.")

    start_time = time.perf_counter()
    status = "success"

    try:
        history = [{"role": h.role, "content": h.content} for h in chat_request.history]
        # get_answer() melakukan panggilan jaringan sinkron/blocking (embed + Gemini, termasuk
        # time.sleep saat retry) — dijalankan di threadpool supaya tidak menahan event loop dan
        # ikut menunda user lain yang concurrent (lebih terasa di deployment single-worker
        # non-serverless seperti Dockerfile/Render/Railway).
        result = await run_in_threadpool(get_answer, chat_request.message, history=history)
        answer = result["answer"]
        sources = result["sources"]

        if result.get("status") == "rate_limited":
            status = "rate_limit"
            return JSONResponse(
                status_code=503,
                content={"answer": answer, "sources": []},
            )

        # "gated" (ditolak sebelum LLM) dan "smalltalk" dicatat apa adanya supaya bisa dihitung
        # terpisah dari jawaban biasa ("success") saat mengevaluasi threshold.
        if result.get("status") in ("gated", "smalltalk"):
            status = result["status"]

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
        top_score = locals().get("result", {}).get("top_score")
        _log_request(
            question=chat_request.message,
            answer=locals().get("answer", ""),
            sources=locals().get("sources", []),
            latency_ms=latency_ms,
            status=status,
            retrieval_query=locals().get("result", {}).get("retrieval_query", ""),
            top_score=top_score,
        )
        # Baris log TANPA teks pertanyaan/jawaban: di Vercel file log tidak persist, tapi stdout
        # masuk Function Logs — cukup untuk memantau sebaran skor dan berapa banyak yang di-gate.
        logger.info(
            f"chat status={status} top_score={'-' if top_score is None else f'{top_score:.3f}'} "
            f"sumber={len(locals().get('sources', []))} latency_ms={latency_ms:.0f}"
        )


@app.post("/admin/reindex")
async def reindex(x_admin_key: str = Header(None)):
    """
    Endpoint admin untuk reindex semua dokumen PDF.
    Membutuhkan header X-Admin-Key yang sesuai dengan ADMIN_KEY di environment.
    """
    if not hmac.compare_digest(x_admin_key or "", ADMIN_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Non-blocking acquire: run_indexing() makan waktu beberapa menit (embed ratusan chunk
    # satu-satu). Kalau dijalankan langsung tanpa run_in_threadpool, itu menahan event loop
    # dan membekukan SEMUA request lain (termasuk /api/chat) selama itu. Lock mencegah dua
    # reindex nyaris bersamaan menulis vector_store.save() secara konkuren (walau save() sudah
    # atomic per-panggilan, dua panggilan overlap tetap bisa saling menimpa hasil satu sama lain).
    if not _reindex_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Reindex lain sedang berjalan, coba lagi nanti.")

    try:
        logger.info("Reindex dimulai oleh admin...")
        result = await run_in_threadpool(run_indexing)
        logger.info(f"Reindex selesai: {result}")
        return {"status": "success", **result}

    except FileNotFoundError as e:
        logger.error(f"Reindex gagal - file tidak ditemukan: {e}")
        raise HTTPException(status_code=500, detail="Reindex gagal. Periksa log server untuk detail.")

    except requests.exceptions.RequestException as e:
        # HARUS dicek SEBELUM (OSError, PermissionError) di bawah: requests.exceptions.
        # RequestException (termasuk HTTPError dari embed API) mewarisi IOError, yang di
        # Python 3 adalah alias OSError — tanpa cabang ini, kegagalan panggilan API embedding
        # (mis. quota habis, key salah) akan salah tertangkap sebagai "OSError" dan dilaporkan
        # dengan pesan "filesystem read-only" yang menyesatkan admin yang men-debug.
        logger.error(f"Reindex gagal - panggilan API embedding gagal: {e}")
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

    finally:
        _reindex_lock.release()
