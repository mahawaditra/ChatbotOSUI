# Spesifikasi Proyek: RAG Chatbot AD/ART & SOP Organisasi Musik

## 1. Konteks & Tujuan

Sebuah organisasi musik memiliki dokumen AD/ART (Anggaran Dasar/Anggaran Rumah Tangga) dan SOP untuk setiap divisi. Total keseluruhan dokumen sekitar 60 halaman, dalam format text PDF (ditulis di Microsoft Word lalu export ke PDF). Anggota organisasi sering lupa atau bingung dengan aturan-aturan ini dan biasanya bertanya ke pengurus inti.

**Tujuan**: Membangun chatbot berbasis RAG (Retrieval Augmented Generation) yang dapat menjawab pertanyaan anggota berdasarkan dokumen AD/ART dan SOP divisi. Chatbot ini akan diakses melalui web interface.

**Ini bukan demo. Ini adalah aplikasi produksi nyata yang akan digunakan oleh anggota organisasi.**

---

## 2. Arsitektur

```
[Anggota Organisasi]
      │
      └─> [Web Interface] (HTML/JS, disajikan langsung oleh FastAPI)
              │
              ▼
        [FastAPI Backend - Knative/GKE Autopilot]
              │
        ┌─────┼─────┐
        │     │     │
   [Upstash  [Gemini  [Gemini
    Vector]  Embed]   LLM]
```

### 2.1 Alur Kerja RAG

#### Saat Indexing (upload dokumen):
1. PDF dimuat menggunakan `PyPDFLoader`
2. Dokumen dipecah menjadi chunks menggunakan `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=200)
3. Setiap chunk diberi metadata: nama file sumber, nomor halaman
4. Chunks di-embed menggunakan Gemini Embedding API
5. Vektor disimpan di Upstash Vector

#### Saat Querying (user bertanya):
1. Pertanyaan user di-embed menggunakan Gemini Embedding API
2. Sistem mencari top-3 chunks paling mirip (cosine similarity) di Upstash Vector
3. Konteks (3 chunks) + pertanyaan user dikirim ke Gemini 2.0 Flash sebagai prompt
4. LLM menjawab berdasarkan konteks yang diberikan
5. Jawaban dikembalikan ke user

### 2.2 Prompt Template untuk LLM

```
System Message:
Kamu adalah asisten yang menjawab pertanyaan anggota organisasi berdasarkan dokumen AD/ART dan SOP.

ATURAN KETAT:
- Jawab HANYA berdasarkan konteks yang diberikan di bawah
- Jika informasi tidak ada di konteks, jawab: "Maaf, informasi tersebut tidak ditemukan dalam dokumen AD/ART atau SOP organisasi."
- JANGAN mengarang atau menambahkan informasi dari luar konteks
- Jika pertanyaan tidak terkait AD/ART atau SOP, jawab: "Maaf, saya hanya bisa menjawab pertanyaan terkait AD/ART dan SOP organisasi."
- Gunakan bahasa Indonesia yang jelas dan sopan
- Jika ada pasal/ayat yang relevan, sebutkan nomor pasal/ayatnya

KONTEKS:
{context}

User Question:
{question}
```

---

## 3. Tech Stack

| Komponen | Teknologi | Versi/Keterangan |
|---|---|---|
| Bahasa | Python | 3.11+ |
| Web Framework | FastAPI | Latest |
| RAG Framework | LangChain | Latest |
| LLM | Gemini 2.0 Flash | Via Google AI Studio (gratis) |
| Embedding | Gemini Embedding | Via Google AI Studio (gratis) |
| Vector Database | Upstash Vector | Free tier |
| Containerization | Docker | Untuk build image |
| Orchestration | Kubernetes + Knative | GKE Autopilot |
| CI/CD | GitHub Actions | Build & deploy otomatis |
| Version Control | GitHub | Public repo (gratis) |

### 3.1 Keterangan Budget

**Seluruh komponen harus gratis ($0).** Tidak ada budget untuk infrastruktur berbayar.

- GKE Autopilot: control plane gratis, node scale-to-zero bersama Knative
- Gemini API: free tier melalui Google AI Studio
- Upstash Vector: free tier (10.000 vektor, lebih dari cukup)
- GitHub: public repo gratis, GitHub Actions gratis

---

## 4. Dokumen Sumber

- Lokasi file: folder `dokumen/` di root proyek
- Format: text PDF (bukan scanned PDF, bisa di-select dan copy-paste teksnya)
- Isi: 1 file AD/ART + beberapa file SOP per divisi
- Total: sekitar 60 halaman
- Semua dokumen dalam **bahasa Indonesia**

### 4.1 Penamaan File (konvensi)

```
dokumen/
├── AD-ART-Organisasi-2024.pdf
├── SOP-Divisi-Humas.pdf
├── SOP-Divisi-Akademik.pdf
├── SOP-Divisi-Keuangan.pdf
└── SOP-Divisi-Logistik.pdf
```

>Nama file di atas hanya contoh. Yang penting ikuti pola yang deskriptif. Saat indexing, nama file disimpan sebagai metadata di setiap chunk.

---

## 5. Endpoint API

### 5.1 Chat Endpoint

```
POST /api/chat
Content-Type: application/json

Request Body:
{
  "message": "Berapa masa jabatan ketua?"
}

Response (200):
{
  "answer": "Berdasarkan Pasal 15 Ayat 1 AD/ART, masa jabatan ketua adalah...",
  "sources": [
    {"file": "AD-ART-Organisasi-2024.pdf", "page": 5},
    {"file": "AD-ART-Organisasi-2024.pdf", "page": 5}
  ]
}

Response (503 - rate limit):
{
  "answer": "Maaf, sedang banyak yang bertanya. Silakan coba lagi dalam beberapa menit.",
  "sources": []
}
```

### 5.2 Health Check Endpoint

```
GET /health

Response (200):
{
  "status": "ok"
}
```

Endpoint ini digunakan oleh Knative untuk menentukan apakah pod sehat dan siap menerima traffic.

### 5.3 Admin Reindex Endpoint

```
POST /admin/reindex
Headers:
  X-Admin-Key: <secret key dari environment variable>

Request Body: (tidak ada body, langsung proses ulang semua file di folder dokumen/)

Response (200):
{
  "status": "success",
  "total_chunks_indexed": 245,
  "files_processed": ["AD-ART-Organisasi-2024.pdf", "SOP-Divisi-Humas.pdf", ...]
}

Response (401):
{
  "detail": "Unauthorized"
}

Response (500):
{
  "status": "error",
  "message": "Deskripsi error"
}
```

Endpoint ini:
- Menghapus seluruh vektor lama di Upstash
- Memproses ulang semua PDF di folder `dokumen/`
- Split → embed → simpan ke Upstash Vector

### 5.4 Web Interface

FastAPI menyajikan halaman HTML statis di route `/`. Halaman ini berisi:
- Input teks untuk pertanyaan
- Tombol kirim
- Area tampilan jawaban
- Indikator loading (teks "Sedang memproses jawaban..." atau spinner) saat menunggu respons
- Referensi sumber yang ditampilkan di bawah jawaban (file dan halaman)

> **Catatan tentang cold start**: Knative scale-to-zero menyebabkan request pertama setelah idle memiliki latency 5-15 detik. Frontend WAJIB menampilkan indikator loading sejak awal, bukan hanya setelah server merespon. Gunakan timeout di frontend — jika tidak ada respons dalam 30 detik, tampilkan pesan "Permintaan memakan waktu lebih lama dari biasanya. Silakan tunggu atau coba lagi."

Web interface cukup sederhana fungsional. Tidak perlu desain yang mewah, tapi harus rapi dan nyaman digunakan di mobile (responsive).

---

## 6. Error Handling & Graceful Degradation

### 6.1 LLM Rate Limit

Gemini free tier punya rate limit. Ketika terjadi error dari Gemini API:

1. **Coba ulang 1 kali** dengan jeda 2 detik (retry)
2. Kalau tetap gagal, kirim pesan ke user: *"Maaf, sedang banyak yang bertanya. Silakan coba lagi dalam beberapa menit."*
3. Log error untuk monitoring

### 6.2 Upstash Vector Error

Kalau Upstash tidak bisa dijangkau:
- Return 503 dengan pesan error generik
- Log error

### 6.3 Dokumen Kosong

Kalau vector database kosong (belum pernah di-index):
- Return pesan: *"Sistem belum memiliki dokumen. Silakan hubungi pengurus."*

---

## 7. Logging

Setiap pertanyaan yang masuk di-log untuk keperluan debugging dan insight.

**Format log per request:**
```json
{
  "timestamp": "2025-01-15T10:30:00Z",
  "question": "Berapa masa jabatan ketua?",
  "answer_preview": "Berdasarkan Pasal 15...",
  "sources": ["AD-ART-Organisasi-2024.pdf hal. 5"],
  "chunks_retrieved": 3,
  "latency_ms": 2300,
  "llm_model": "gemini-2.0-flash",
  "status": "success"
}
```

Log disimpan sebagai file JSON (satu file per hari) di folder lokal pod. Tidak perlu persistence khusus — log ini bersifat best-effort dan akan hilang jika pod di-restart.

---

## 8. Struktur Proyek

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app, routes, startup/shutdown
│   ├── config.py               # Konfigurasi dari environment variables
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── indexing.py          # Fungsi indexing (PDF → chunks → embed → store)
│   │   ├── retrieval.py         # Fungsi retrieval (query → embed → search)
│   │   ├── chain.py             # RAG chain (retrieval + LLM generation)
│   │   └── prompts.py           # Prompt templates
│   └── static/
│       └── index.html           # Web interface
├── dokumen/                     # PDF files (AD/ART + SOP)
│   └── .gitkeep
├── Dockerfile
├── service.yaml                 # Knative Service manifest
├── .github/
│   └── workflows/
│       └── deploy.yaml          # GitHub Actions CI/CD
├── requirements.txt
└── README.md
```

---

## 9. Environment Variables

Semua konfigurasi melalui environment variables. Tidak boleh ada hardcoded secret.

| Variable | Keterangan | Contoh |
|---|---|---|
| `GEMINI_API_KEY` | API key Google AI Studio | `AIza...` |
| `UPSTASH_VECTOR_REST_URL` | Upstash Vector endpoint URL | `https://...` |
| `UPSTASH_VECTOR_REST_TOKEN` | Upstash Vector token | `...` |
| `ADMIN_KEY` | Secret key untuk endpoint admin reindex | `some-random-string` |
| `COLLECTION_NAME` | Nama koleksi di Upstash Vector | `org-rag` |

---

## 10. Deployment

### 10.1 Kubernetes + Knative di GKE Autopilot

1. Buat GKE Autopilot cluster
2. Install Knative serving
3. Buat Kubernetes Secret untuk environment variables
4. Deploy menggunakan Knative Service manifest (`service.yaml`)
5. Knative otomatis mengelola scaling (termasuk scale-to-zero)

### 10.2 Knative Service Manifest

```yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rag-chatbot
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/min-scale: "0"
        autoscaling.knative.dev/max-scale: "1"
    spec:
      containers:
        - image: <container-image-from-GAR>
          env:
            - name: GEMINI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: chatbot-secrets
                  key: gemini-api-key
            - name: UPSTASH_VECTOR_REST_URL
              valueFrom:
                secretKeyRef:
                  name: chatbot-secrets
                  key: upstash-url
            - name: UPSTASH_VECTOR_REST_TOKEN
              valueFrom:
                secretKeyRef:
                  name: chatbot-secrets
                  key: upstash-token
            - name: ADMIN_KEY
              valueFrom:
                secretKeyRef:
                  name: chatbot-secrets
                  key: admin-key
          ports:
            - containerPort: 8080
          readinessProbe:
            httpGet:
              path: /health
              port: 8080
```

### 10.3 GitHub Actions CI/CD

Alur:
1. Push ke `main` branch memicu workflow
2. Build Docker image
3. Push image ke Google Artifact Registry (GAR)
4. Update Knative Service image ke versi terbaru
5. Knative melakukan rolling update

### 10.4 Dockerfile

Gunakan Python base image yang ringan. Pastikan:
- `pip install -r requirements.txt`
- Copy folder `dokumen/` ke dalam image
- Expose port 8080
- CMD menjalankan FastAPI dengan uvicorn

---

## 11. Konfigurasi Upstash Vector

- Gunakan index dimension yang sesuai dengan model embedding Gemini (768 dimensi untuk `text-embedding-004`)
- Distance metric: Cosine
- Free tier: cukup untuk 60 halaman dokumen (~150-250 chunks)

---

## 12. Testing Sebelum Deploy

Sebelum deploy ke GKE, pastikan:
1. Indexing berjalan lokal: semua PDF berhasil diproses dan di-embed
2. Query berjalan lokal: pertanyaan menghasilkan jawaban yang akurat
3. Health check endpoint merespon 200
4. Admin reindex endpoint bekerja (hapus lama, index ulang)
5. Error handling terjadi dengan benar (LLM error, Upstash error)
6. Web interface berfungsi dan responsive

---

## 13. Catatan Penting

- **Bahasa**: Seluruh dokumen dan pertanyaan dalam bahasa Indonesia. Pastikan embedding model dan LLM handle bahasa Indonesia dengan baik (Gemini bisa).
- **Tidak ada LINE bot**: Fitur LINE dihapus dari scope. Hanya web interface.
- **Tidak perlu domain publik**: Karena tidak ada LINE webhook, tidak perlu domain berbayar. GKE Knative menyediakan URL internal.
- **Tidak ada database relasional**: Tidak perlu PostgreSQL atau database lain. Hanya Upstash Vector untuk vektor.
- **Scale-to-zero**: Aplikasi harus benar-benar $0 saat idle. Pastikan `min-scale: "0"` di Knative.
- **GKE Autopilot**: Gunakan GKE Autopilot (bukan Standard) agar tidak ada biaya control plane dan node bisa scale to zero.