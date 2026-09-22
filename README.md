# OSUI Mahawaditra RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that lets members of **OSUI Mahawaditra** (Orkes Simfoni Universitas Indonesia) ask questions in plain Indonesian about the organization's **AD/ART** (bylaws) and divisional **SOPs**, and get answers grounded in — and cited to — the actual documents.

## Why this exists

AD/ART and SOP documents are long and dense. In practice, members (and even committee holders) forget specific clauses all the time — what the quorum for an extraordinary general meeting is, how a reimbursement request is supposed to be filed, what the fine is for a late due payment, and so on. Digging through a 20-page bylaws PDF every time is friction nobody wants.

This chatbot exists so members can just ask, in natural language, and get an answer sourced directly from the real document — with a link straight to the page it came from, so they can verify it themselves.

This is also the author's first hands-on project exploring RAG. Some of the retrieval design (see [How Retrieval Works](#how-retrieval-works)) reflects lessons learned by hitting real issues with a small, low-resource setup (free-tier LLM + vector DB) rather than a from-the-start "correct" architecture — noted here in case it's useful context for anyone reading the code.

## Features

- **Grounded Q&A in Indonesian** — answers come only from the actual AD/ART and SOP content; if it's not in the documents, the bot says so instead of guessing.
- **Cited, clickable sources** — every answer lists the source file(s) and page(s) it was drawn from. Clicking a source opens the PDF in-app and jumps straight to that page.
- **Context-aware follow-ups** — short follow-up questions ("terus kalau resign gimana?") are automatically rewritten into standalone queries using the conversation history, so retrieval quality doesn't degrade in multi-turn conversations.
- **Admin-triggered reindexing** — whenever documents change, an admin can rebuild the entire searchable index with one authenticated request.

## User flow

**Member:**
1. Open the chat page — a welcome screen with a few suggested prompts is shown.
2. Ask a question in Indonesian, e.g. *"Apa syarat untuk mengganti isi AD/ART?"*.
3. Get an answer plus a **"Sumber Dokumen Rujukan"** (source) list — e.g. *ADART-OSUIMahawaditra-2022.pdf, page 16*.
4. Click a source to open that PDF directly at the cited page, to double-check the wording yourself.
5. Ask a follow-up without repeating context — the app resolves what you mean using the chat so far.
6. If the question is off-topic or genuinely not covered by the documents, the bot says so explicitly rather than making something up.

**Admin:**
1. Add or replace PDF files in `dokumen/`.
2. Call `POST /admin/reindex` with the admin key to rebuild the entire vector index (always a full rebuild — see [Known Limitations](#known-limitations)).

## Tech stack

| Layer | Choice |
|---|---|
| Backend | [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+) |
| Frontend | Single-page vanilla HTML/CSS/JS, served directly by FastAPI — no separate frontend framework or build step |
| LLM (answers) | Google Gemini, via the `google-genai` SDK. Model set by `LLM_MODEL` in `app/config.py` |
| Embeddings | Google Gemini embedding model, called via **raw REST** rather than the SDK (see comments in `app/rag/retrieval.py` / `app/rag/indexing.py` — the SDK's `v1beta` path didn't support the embedding model this project needs). Configured via `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` in `app/config.py` |
| Vector database | [Upstash Vector](https://upstash.com/docs/vector) (serverless, REST-based) |
| PDF processing | [LangChain](https://python.langchain.com/) (`PyPDFLoader` + `RecursiveCharacterTextSplitter`), plus a custom structure-aware chunker for AD/ART-style documents (see below) |

## How it works

### Indexing (`POST /admin/reindex`, full rebuild)
1. Every PDF in `dokumen/` is loaded page by page.
2. Documents that follow a `"Pasal <N>"` article structure (currently the AD/ART) are split along those article boundaries instead of by raw character count, so a single article isn't cut in half mid-sentence. Documents without that structure (the SOPs) fall back to standard character-based chunking.
3. Each chunk is embedded individually via the Gemini embedding API (with automatic retry/backoff on rate limits) and upserted into Upstash Vector. **The entire existing index is wiped first** (`index.reset()`) — there is no incremental/per-document update.

### Answering a question (`POST /api/chat`)
1. If there's prior conversation history, the question is first rewritten by the LLM into a standalone query (so "terus kalau resign gimana?" keeps referring to whatever was being discussed).
2. That query is embedded and used to fetch a wider pool of candidate chunks from Upstash than what's actually needed (see `RETRIEVAL_FETCH_K` in [Known Limitations](#known-limitations)), which are then filtered by a minimum similarity score and trimmed down to the top few.
3. If nothing relevant survives that filter, the bot returns a refusal message **without calling the LLM at all** — saves cost/latency and avoids answering from irrelevant context.
4. Otherwise, the surviving chunks, the original question, and recent history are assembled into a prompt — with document context, user input, and history each wrapped in separate tags as a prompt-injection guard — and sent to Gemini.
5. The answer is returned with a deduplicated list of sources (file + page).

## Getting started

### Prerequisites
- Python 3.11+
- Your organization's PDF documents placed in `dokumen/`

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in real values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | yes | Google AI Studio API key |
| `UPSTASH_VECTOR_REST_URL` | yes | Upstash Vector REST endpoint |
| `UPSTASH_VECTOR_REST_TOKEN` | yes | Upstash Vector REST token |
| `ADMIN_KEY` | yes | Shared secret required by `POST /admin/reindex` |
| `COLLECTION_NAME` | no (default `org-rag`) | Currently unused by the code — reserved for future Upstash namespacing |

Generate a secure `ADMIN_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

> All required variables are read eagerly at startup (`app/config.py`) — the app fails immediately on launch if any are missing, rather than failing later on a request.

### 3. Run the dev server
```bash
uvicorn app.main:app --reload --port 8080
```
Open `http://localhost:8080` in your browser.

### 4. Index your documents (required before first use, and after any document change)

This wipes and rebuilds the **entire** vector index. With more/larger documents this can take a couple of minutes, since each chunk is embedded one at a time with automatic retry on rate limits.

**macOS / Linux / Git Bash:**
```bash
curl -X POST http://localhost:8080/admin/reindex -H "X-Admin-Key: YOUR_ADMIN_KEY"
```

**Windows PowerShell:**
```powershell
curl.exe -X POST http://localhost:8080/admin/reindex -H "X-Admin-Key: YOUR_ADMIN_KEY"
```
> On Windows, plain `curl` in PowerShell is usually an alias for `Invoke-WebRequest`, which does **not** understand curl-style `-X`/`-H` flags and will fail. Use `curl.exe` explicitly (forces the real curl binary) as shown above, or use a native PowerShell cmdlet instead:
> ```powershell
> Invoke-RestMethod -Method Post -Uri http://localhost:8080/admin/reindex -Headers @{ "X-Admin-Key" = "YOUR_ADMIN_KEY" }
> ```

Replace `YOUR_ADMIN_KEY` with the actual value from your `.env` file directly. Don't rely on shell variable expansion (`$ADMIN_KEY` in bash, `$env:ADMIN_KEY` in PowerShell) unless you've explicitly set it in that same terminal session — `.env` is only loaded by the Python process, not by your shell.

A successful response looks like:
```json
{"status": "success", "total_chunks_indexed": 246, "files_processed": ["ADART-OSUIMahawaditra-2022.pdf", "SOP-Divisi-Acara-2026.pdf", "..."]}
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web chat interface |
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Ask the chatbot a question |
| `POST` | `/admin/reindex` | Rebuild the entire vector index (requires `X-Admin-Key` header) |

### Example: asking a question

**macOS / Linux / Git Bash:**
```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Berapa masa jabatan ketua?"}'
```

**Windows PowerShell:**
```powershell
curl.exe -X POST http://localhost:8080/api/chat -H "Content-Type: application/json" -d '{\"message\": \"Berapa masa jabatan ketua?\"}'
```

Response:
```json
{
  "answer": "...",
  "sources": [{"file": "ADART-OSUIMahawaditra-2022.pdf", "page": 5}]
}
```

## Updating documents

1. Add/replace PDF files in `dokumen/`.
2. Redeploy, so the running instance has the new files.
3. Call `POST /admin/reindex` again — this always does a full rebuild; there's no partial/incremental update.

## Deployment

The deployment target has changed a couple of times during development. **The current plan is [Vercel](https://vercel.com/)**, but this app wasn't originally architected for a serverless platform, and a few things need attention before it will work well there:

- **`/admin/reindex` can take a couple of minutes** (each chunk is embedded one at a time, with retries on rate limits). This will very likely exceed Vercel's serverless function time limit (10s on Hobby, 60s on Pro). Consider triggering reindexing from a separate script/CI job instead of the hosted endpoint, at least for large rebuilds.
- **Local file-based request logging (`logs/*.json`) won't persist** on Vercel's ephemeral, mostly-read-only filesystem. This fails silently today (write errors are caught and only logged, never surfaced to users), so the app itself keeps working — you'd just lose the logs. Point this at an external store if you need it in production.
- Serving the PDFs (for the in-app "open source document" viewer) and the static frontend should work fine as-is, since Vercel serves bundled static files without issue.

*(Historical note, in case old instructions resurface: the `Dockerfile` in this repo currently targets Railway's injected `PORT` env var, and an earlier version of this README documented deploying to Render. Neither reflects the current Vercel plan — if you're touching deployment config, go with what's actually decided at the time, not any single doc.)*

## Known limitations

- **Indexing is always a full rebuild.** There's no per-document delete/update — every `/admin/reindex` call re-processes and re-embeds all PDFs in `dokumen/`.
- **`SIMILARITY_THRESHOLD`** (`app/config.py`) is a starting value, not a calibrated one. If valid answers start getting filtered out (threshold too high) or clearly off-topic questions still get answered (threshold too low), it needs adjusting against real query logs.
- **`RETRIEVAL_FETCH_K` intentionally over-fetches** from Upstash before trimming down to `TOP_K`. This is a workaround, not a stylistic choice: Upstash's approximate nearest-neighbor search was found empirically to be unreliable at very small `top_k` values on this corpus (similarity scores across chunks cluster very tightly), occasionally missing the single most relevant chunk entirely when asked for only the top 5.
- **Gemini free-tier rate limits apply.** Live chat answers retry once before returning a "please try again" message; indexing retries several times with exponential backoff, but sustained rate limiting will still surface as an error.

## No test suite

There's no automated test suite, linter, or type-checker configured in this repo yet.
