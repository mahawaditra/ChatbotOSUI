"""
Vector store lokal: menyimpan embeddings sebagai file (vectors.npy + metadata.json) di
data/vector_store/, di-commit ke git alih-alih disimpan di database cloud pihak ketiga.

Cocok untuk korpus dokumen kecil (ratusan chunk) — cosine similarity dihitung exact
(brute-force) dengan numpy, bukan approximate nearest-neighbor, jadi hasil pencarian
selalu benar-benar yang paling mirip tanpa perlu over-fetch besar seperti era Upstash.
"""

import json
import logging
import os
from typing import Any

import numpy as np

from app.config import VECTOR_STORE_DIR, EMBEDDING_DIMENSION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_VECTORS_PATH = VECTOR_STORE_DIR / "vectors.npy"
_METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"
_MANIFEST_PATH = VECTOR_STORE_DIR / "manifest.json"

_cached_vectors: np.ndarray | None = None
_cached_metadata: list[dict[str, Any]] | None = None
_loaded: bool = False
# Terpisah dari _loaded: kalau vector store rusak/hilang/tidak cocok, ini dicoba baca sekali
# lalu selalu return False sampai proses di-restart — tanpa ini, query() akan mengulang I/O
# disk yang sama-sama gagal di SETIAP request selama vector store dalam keadaan rusak.
_load_failed: bool = False


def save(vectors: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
    """
    Menulis ulang seluruh vector store lokal, menggantikan isi lama sepenuhnya (full
    overwrite, dipanggil sekali di akhir run_indexing() setelah semua chunk di-embed —
    tidak ada langkah "delete" terpisah karena ini selalu menimpa total).

    Ditulis via temp file + os.replace (atomic pada filesystem yang sama) untuk ketiga file
    (vectors.npy, metadata.json, manifest.json) — supaya proses yang terhenti di tengah jalan
    (kill, OOM, disk penuh) tidak pernah meninggalkan pasangan file yang setengah-tertulis
    atau tidak konsisten satu sama lain.

    Args:
        vectors: List embedding vector, satu per chunk.
        metadatas: List dict {"text", "file", "page"}, index-aligned dengan `vectors`.
    """
    if len(vectors) != len(metadatas):
        raise ValueError(
            f"Jumlah vectors ({len(vectors)}) dan metadatas ({len(metadatas)}) tidak sama."
        )

    if vectors:
        array = np.asarray(vectors, dtype=np.float32)
    else:
        array = np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    # Dicatat supaya query() bisa mendeteksi vector store yang di-generate dengan model
    # embedding berbeda dari yang sekarang dipakai, walau dimensinya kebetulan sama
    # (kasus itu tidak crash secara alami — cosine similarity tetap terhitung tapi antar
    # dua ruang embedding yang tidak sebanding, menghasilkan jawaban salah yang percaya diri).
    manifest = {"embedding_model": EMBEDDING_MODEL, "embedding_dimension": EMBEDDING_DIMENSION}

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    # File .tmp ditulis dengan nama eksplisit (bukan diserahkan ke np.save untuk ditebak) dan
    # dibuka sebagai file handle biasa, supaya np.save() tidak menambahkan ekstensi .npy sendiri
    # ke nama file .tmp (perilaku default np.save kalau diberi path yang belum berakhiran .npy).
    vectors_tmp = _VECTORS_PATH.with_name(_VECTORS_PATH.name + ".tmp")
    metadata_tmp = _METADATA_PATH.with_name(_METADATA_PATH.name + ".tmp")
    manifest_tmp = _MANIFEST_PATH.with_name(_MANIFEST_PATH.name + ".tmp")

    with open(vectors_tmp, "wb") as f:
        np.save(f, array)
    with open(metadata_tmp, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    os.replace(vectors_tmp, _VECTORS_PATH)
    os.replace(metadata_tmp, _METADATA_PATH)
    os.replace(manifest_tmp, _MANIFEST_PATH)

    global _cached_vectors, _cached_metadata, _loaded, _load_failed
    _cached_vectors = array
    _cached_metadata = metadatas
    _loaded = True
    _load_failed = False

    logger.info(f"Vector store lokal disimpan: {len(vectors)} vektor di {VECTOR_STORE_DIR}")


def _load() -> bool:
    """Lazy-load vectors.npy + metadata.json ke cache level-modul (sekali per proses)."""
    global _cached_vectors, _cached_metadata, _loaded, _load_failed

    if _loaded:
        return True
    if _load_failed:
        return False

    if not _VECTORS_PATH.exists() or not _METADATA_PATH.exists():
        logger.error(
            f"Vector store lokal belum ada di {VECTOR_STORE_DIR} — SEMUA pertanyaan akan dijawab "
            "'informasi tidak ditemukan' sampai ini diperbaiki. Jalankan `python scripts/reindex.py`, "
            "commit hasilnya, lalu deploy ulang."
        )
        _load_failed = True
        return False

    try:
        with open(_VECTORS_PATH, "rb") as f:
            vectors = np.load(f)
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError, EOFError) as e:
        logger.error(
            f"Vector store lokal RUSAK/tidak terbaca di {VECTOR_STORE_DIR}: {e}. SEMUA pertanyaan "
            "akan dijawab 'informasi tidak ditemukan' sampai `python scripts/reindex.py` dijalankan ulang."
        )
        _load_failed = True
        return False

    if vectors.shape[0] != len(metadata):
        logger.error(
            f"Vector store lokal TIDAK KONSISTEN: {vectors.shape[0]} vektor vs {len(metadata)} "
            "metadata. SEMUA pertanyaan akan dijawab 'informasi tidak ditemukan' sampai "
            "`python scripts/reindex.py` dijalankan ulang."
        )
        _load_failed = True
        return False

    if _MANIFEST_PATH.exists():
        manifest: dict[str, Any] | None
        try:
            with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Gagal membaca manifest.json vector store, lanjut tanpa validasi: {e}")
            manifest = None

        if manifest is not None and (
            manifest.get("embedding_model") != EMBEDDING_MODEL
            or manifest.get("embedding_dimension") != EMBEDDING_DIMENSION
        ):
            logger.error(
                "Vector store lokal di-generate dengan embedding BERBEDA dari config.py saat ini "
                f"(manifest: model={manifest.get('embedding_model')!r}, "
                f"dim={manifest.get('embedding_dimension')!r}; config: model={EMBEDDING_MODEL!r}, "
                f"dim={EMBEDDING_DIMENSION!r}). Similarity search tidak bisa dipercaya walau dimensi "
                "kebetulan cocok — jalankan ulang `python scripts/reindex.py`. SEMUA pertanyaan akan "
                "dijawab 'informasi tidak ditemukan' sampai diperbaiki."
            )
            _load_failed = True
            return False
    else:
        logger.warning(
            "manifest.json tidak ditemukan di vector store lokal (kemungkinan dibuat sebelum fitur "
            "ini ada) — tidak bisa memvalidasi kecocokan model embedding, melanjutkan tanpa validasi."
        )

    _cached_vectors = vectors
    _cached_metadata = metadata
    _loaded = True
    return True


def query(vector: list[float], top_k: int) -> list[dict[str, Any]]:
    """
    Cari top_k chunk paling mirip secara cosine similarity terhadap `vector`, dihitung
    brute-force dengan numpy atas seluruh vector store lokal.

    Returns:
        List dict {"text", "file", "page", "score"}, terurut menurun berdasarkan score,
        panjang <= top_k. Score dinormalisasi ke [0,1] via (1 + cosine_similarity) / 2 —
        sama persis dengan semantik skor Upstash lama, supaya SIMILARITY_THRESHOLD di
        config.py tetap berarti sama. List kosong kalau vector store belum ada/kosong/rusak.
    """
    if not _load() or _cached_vectors.shape[0] == 0:
        return []

    q = np.asarray(vector, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    v_norms = _cached_vectors / (np.linalg.norm(_cached_vectors, axis=1, keepdims=True) + 1e-10)
    raw_cosine = v_norms @ q_norm
    scores = (1.0 + raw_cosine) / 2.0

    k = min(top_k, scores.shape[0])
    top_indices = np.argsort(-scores)[:k]

    return [
        {
            "text": _cached_metadata[i].get("text", ""),
            "file": _cached_metadata[i].get("file", "unknown"),
            "page": _cached_metadata[i].get("page", 0),
            "score": float(scores[i]),
        }
        for i in top_indices
    ]
