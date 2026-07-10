# Chatbot RAG OSUI Mahawaditra

Chatbot berbasis RAG (Retrieval Augmented Generation) untuk menjawab pertanyaan anggota organisasi seputar AD/ART dan SOP divisi OSUI Mahawaditra.

## Stack

- **Backend**: FastAPI + LangChain
- **LLM & Embedding**: Google Gemini 2.0 Flash + text-embedding-004
- **Vector DB**: Upstash Vector
- **Deployment**: Render (scale-to-zero gratis)

## Setup Lokal

### 1. Prasyarat

- Python 3.11+
- File PDF dokumen sudah ada di folder `dokumen/`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Buat file `.env`

Salin `.env.example` dan isi dengan credentials asli:

```bash
cp .env.example .env
```

Isi `.env`:
```
GEMINI_API_KEY=AIza...
UPSTASH_VECTOR_REST_URL=https://...
UPSTASH_VECTOR_REST_TOKEN=...
ADMIN_KEY=random-string-panjang
COLLECTION_NAME=org-rag
```

Untuk membuat `ADMIN_KEY` yang aman:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Jalankan server

```bash
uvicorn app.main:app --reload --port 8080
```

Buka browser ke `http://localhost:8080`

### 5. Index dokumen (wajib dilakukan sekali)

```bash
curl -X POST http://localhost:8080/admin/reindex \
  -H "X-Admin-Key: YOUR_ADMIN_KEY"
```

## Endpoint API

| Method | Path | Keterangan |
|---|---|---|
| `GET` | `/` | Web interface chatbot |
| `GET` | `/health` | Health check |
| `POST` | `/api/chat` | Tanya chatbot |
| `POST` | `/admin/reindex` | Reindex semua PDF |

### Contoh request chat

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Berapa masa jabatan ketua?"}'
```

## Deployment ke Render

1. Push kode ke GitHub
2. Buka [render.com](https://render.com) → **New** → **Web Service**
3. Connect ke repo GitHub ini
4. Konfigurasi:
   - **Runtime**: Docker
   - **Region**: Singapore (terdekat)
   - **Instance Type**: Free
5. Tambahkan environment variables di Render dashboard (sama dengan `.env`)
6. Deploy → tunggu build selesai
7. Panggil `/admin/reindex` sekali untuk index dokumen di cloud:

```bash
curl -X POST https://YOUR-APP.onrender.com/admin/reindex \
  -H "X-Admin-Key: YOUR_ADMIN_KEY"
```

## Update Dokumen

Jika ada dokumen baru atau dokumen diupdate:

1. Tambah/ganti file PDF di folder `dokumen/`
2. Commit dan push ke GitHub
3. Render otomatis rebuild dan deploy
4. Panggil `/admin/reindex` lagi

## Catatan

- **Cold start**: Render free tier sleep setelah 15 menit idle. Request pertama butuh ~20-30 detik (normal).
- **Log**: Tersimpan di folder `logs/` dalam container, per hari. Akan hilang saat container restart (best-effort).
- **Rate limit**: Gemini free tier memiliki batas request. Jika terkena limit, chatbot akan retry 1x dan memberi pesan ke user.
