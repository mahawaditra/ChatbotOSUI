# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

RAG (Retrieval Augmented Generation) chatbot that answers OSUI Mahawaditra organization members' questions about AD/ART (bylaws) and divisional SOPs, sourced from PDFs in `dokumen/`. FastAPI backend, single-page vanilla HTML/JS frontend served directly by FastAPI. All user-facing text, prompts, and code comments are in Indonesian — keep new user-facing strings and log messages in Indonesian for consistency.

This is a real production app used by actual org members, not a demo — treat error handling and the "answer only from context" constraint as load-bearing, not incidental.

## Commands

```bash
# Install deps (Python 3.11+; project uses a local .venv on Windows)
pip install -r requirements.txt

# Run dev server (reload on change)
uvicorn app.main:app --reload --port 8080

# Reindex all PDFs in dokumen/ into Upstash Vector (required after first setup or doc changes)
curl -X POST http://localhost:8080/admin/reindex -H "X-Admin-Key: $ADMIN_KEY"

# Generate a secure ADMIN_KEY
python -c "import secrets; print(secrets.token_hex(32))"
```

There is no test suite, linter, or type-checker configured in this repo — don't invent commands for them.

### Required environment variables (`.env` locally; real env vars in deployment)

`GEMINI_API_KEY`, `UPSTASH_VECTOR_REST_URL`, `UPSTASH_VECTOR_REST_TOKEN`, `ADMIN_KEY`, and optional `COLLECTION_NAME` (default `org-rag`, currently unused — `get_vector_index()` in `retrieval.py`/`indexing.py` connects with just the Upstash URL/token, no namespace param, so changing this env var has no effect yet). `app/config.py` reads these eagerly at import time via `os.environ[...]` (not `.get`) for the required ones — a missing var crashes on startup, not at request time.

## Architecture

Request flow: `app/main.py` (FastAPI routes) → `app/rag/chain.py` (`get_answer`, orchestrates the RAG pipeline) → `app/rag/retrieval.py` (embed query, top-K search in Upstash Vector) → `app/rag/prompts.py` (`build_prompt`, injects context + history) → Gemini LLM via the `google-genai` SDK → answer + deduplicated sources back to `main.py`.

Indexing is a separate, admin-triggered pipeline: `app/rag/indexing.py`'s `run_indexing()` wipes all existing vectors (`index.reset()`), reloads every PDF in `dokumen/` with `PyPDFLoader`, splits with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=200), embeds each chunk individually via a direct Gemini REST call, and upserts to Upstash in batches of 20. It is always a full rebuild, not an incremental update — there is no per-document delete/update path.

Key points a future change needs to respect:

- **Embeddings are called via raw REST, not the LangChain/genai SDK wrapper**, in both `indexing.py` and `retrieval.py` (`_EMBED_URL` = `.../v1/models/{model}:embedContent`). This is deliberate — comments note the `v1beta` SDK path didn't support the embedding model needed. Keep `EMBEDDING_MODEL`/`EMBEDDING_DIMENSION` in `app/config.py` consistent between indexing and retrieval, since a mismatch there silently produces bad similarity search results (Upstash won't error on a dimension mismatch that happens to still parse, it just returns garbage neighbors).
- **`app/config.py` is stale relative to the code**: it currently sets `LLM_MODEL = "gemini-3.1-flash-lite"` and `EMBEDDING_MODEL = "gemini-embedding-001"` (1536 dims), while `README.md` and `rag-chatbot-specification.md` still describe "Gemini 2.0 Flash + text-embedding-004". Trust `app/config.py`, not the docs, for what's actually running.
- **Deployment target is also inconsistent across files**: `rag-chatbot-specification.md` describes an original GKE Autopilot + Knative design, `README.md` documents deploying to Render, and the current `Dockerfile` CMD reads `${PORT:-8080}` specifically for Railway's auto-injected `PORT` (per the most recent commit). When touching deployment-related code, confirm with the user which platform is actually in use rather than trusting any single doc.
- **Sources dedup**: `chain.py` dedupes retrieved chunks by `(file, page)` before returning `sources`, so the same page cited by multiple chunks appears once.
- **Rate-limit handling**: `_call_llm_with_retry` in `chain.py` retries once after a 2s sleep on 429/"resource exhausted"/"quota"/503/"unavailable" errors, then returns the sentinel `RATE_LIMIT_MESSAGE` string (compared by value in `main.py` to decide whether to return HTTP 503). Any other exception is re-raised and caught by the generic handler in `main.py`'s `/api/chat`.
- **Prompt injection defense**: `build_prompt` wraps document context in `<konteks_dokumen>`, user input in `<pertanyaan_user>`, and chat history in `<user>`/`<asisten>` tags, with an explicit system instruction to ignore instructions found inside the user-question tag. Preserve this tagging structure if you modify prompt construction.
- **Logging**: every `/api/chat` call is appended as one JSON line per day to `logs/YYYY-MM-DD.json` (best-effort — write failures are caught and only logged, never surfaced to the user). This directory is gitignored and not persisted across container restarts by design.
- **Admin auth** is a single shared-secret header (`X-Admin-Key` compared to `ADMIN_KEY`), guarding only `/admin/reindex`. There's no other auth in the app.
- **Input limits**: `main.py` caps incoming messages at `MAX_QUESTION_LENGTH` (500 chars) and each history item's content at `MAX_HISTORY_ITEM_LENGTH` (800 chars, silently truncated rather than rejected). The frontend is the single static page at `app/static/index.html`, served inline by `serve_index()` — it POSTs to `/api/chat` with the running chat history and renders answers as markdown client-side (via `marked.js`).
