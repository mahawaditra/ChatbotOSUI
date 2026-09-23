"""
Vector store lokal: menyimpan embeddings sebagai file (vectors.npy + metadata.json) di
data/vector_store/, di-commit ke git alih-alih disimpan di database cloud pihak ketiga.

Cocok untuk korpus dokumen kecil (ratusan chunk) — cosine similarity dihitung exact
(brute-force) dengan numpy, bukan approximate nearest-neighbor, jadi hasil pencarian
selalu benar-benar yang paling mirip tanpa perlu over-fetch besar seperti era Upstash.
"""

import json
import logging
from typing import Any

import numpy as np

from app.config import VECTOR_STORE_DIR, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

_VECTORS_PATH = VECTOR_STORE_DIR / "vectors.npy"
_METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

_cached_vectors: np.ndarray | None = None
_cached_metadata: list[dict[str, Any]] | None = None
_loaded: bool = False


def save(vectors: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
    """
    Menulis ulang seluruh vector store lokal, menggantikan isi lama sepenuhnya (full
    overwrite, dipanggil sekali di akhir run_indexing() setelah semua chunk di-embed —
    tidak ada langkah "delete" terpisah karena ini selalu menimpa total).

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

    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_VECTORS_PATH, array)
    with open(_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, ensure_ascii=False, indent=2)

    global _cached_vectors, _cached_metadata, _loaded
    _cached_vectors = array
    _cached_metadata = metadatas
    _loaded = True

    logger.info(f"Vector store lokal disimpan: {len(vectors)} vektor di {VECTOR_STORE_DIR}")


def _load() -> bool:
    """Lazy-load vectors.npy + metadata.json ke cache level-modul (sekali per proses)."""
    global _cached_vectors, _cached_metadata, _loaded

    if _loaded:
        return True

    if not _VECTORS_PATH.exists() or not _METADATA_PATH.exists():
        logger.warning(
            f"Vector store lokal belum ada di {VECTOR_STORE_DIR}. "
            "Jalankan `python scripts/reindex.py` terlebih dahulu."
        )
        return False

    try:
        vectors = np.load(_VECTORS_PATH)
        with open(_METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError, EOFError) as e:
        logger.warning(f"Gagal membaca vector store lokal di {VECTOR_STORE_DIR}: {e}")
        return False

    if vectors.shape[0] != len(metadata):
        logger.warning(
            f"Vector store lokal tidak konsisten: {vectors.shape[0]} vektor vs "
            f"{len(metadata)} metadata. Jalankan ulang scripts/reindex.py."
        )
        return False

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
        config.py tetap berarti sama. List kosong kalau vector store belum ada/kosong.
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
