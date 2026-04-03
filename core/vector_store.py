"""
core/vector_store.py
────────────────────
Manages a persistent ChromaDB collection.

Responsibilities:
  - Chunk transcript text into overlapping word windows
  - Embed with sentence-transformers
  - Store in Chroma with rich metadata (chapter, timestamps, chapter_index)
  - Retrieve top-K chunks for a query (used by the RAG Q&A pipeline)
  - Reset collection between lectures (one Space = one lecture at a time)
"""

from __future__ import annotations

import uuid
from typing import List, Dict, Any, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

import config
from core.segmenter import Chapter
from utils.logger import get_logger

log = get_logger(__name__)

_embed_model: SentenceTransformer | None = None
_client:      chromadb.Client | None     = None
_collection                              = None


# ── Initialisation ────────────────────────────────────────────────────────────

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
    return _embed_model


def _get_collection():
    global _client, _collection
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(
    text:     str,
    size:     int = config.CHUNK_SIZE_WORDS,
    overlap:  int = config.CHUNK_OVERLAP_WORDS,
) -> List[str]:
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks


# ── Public API ────────────────────────────────────────────────────────────────

def reset_collection() -> None:
    """Drop and re-create the collection (call before each new lecture)."""
    global _client, _collection
    col = _get_collection()
    try:
        _get_collection()  # ensure client is initialised
        _client.delete_collection(config.COLLECTION_NAME)
        log.info("Chroma collection reset.")
    except Exception:
        pass
    _collection = None
    _get_collection()   # re-create


def ingest_chapters(chapters: List[Chapter]) -> int:
    """
    Chunk each chapter's text and upsert into Chroma.
    Returns total number of chunks stored.
    """
    col   = _get_collection()
    model = _get_embed_model()
    total = 0

    for ch in chapters:
        chunks = _chunk_text(ch.text)
        if not chunks:
            continue

        embeddings = model.encode(chunks, show_progress_bar=False).tolist()

        col.upsert(
            ids        = [str(uuid.uuid4()) for _ in chunks],
            embeddings = embeddings,
            documents  = chunks,
            metadatas  = [
                {
                    "chapter_index": ch.index,
                    "chapter_title": ch.title,
                    "start_sec":     ch.start_sec,
                    "end_sec":       ch.end_sec,
                    "summary":       ch.summary,
                }
                for _ in chunks
            ],
        )
        total += len(chunks)
        log.debug(f"Chapter {ch.index}: ingested {len(chunks)} chunks")

    log.info(f"Total chunks ingested: {total}")
    return total


def query(
    question: str,
    top_k:    int = config.TOP_K_CHUNKS,
) -> List[Dict[str, Any]]:
    """
    Retrieve the top-K most relevant chunks for a question.

    Returns list of dicts:
      {"text": str, "chapter_title": str, "chapter_index": int,
       "start_sec": float, "end_sec": float, "distance": float}
    """
    col   = _get_collection()
    model = _get_embed_model()

    q_embed = model.encode([question], show_progress_bar=False).tolist()
    results = col.query(
        query_embeddings = q_embed,
        n_results        = min(top_k, col.count()),
        include          = ["documents", "metadatas", "distances"],
    )

    output = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        output.append({
            "text":          doc,
            "chapter_title": meta.get("chapter_title", ""),
            "chapter_index": meta.get("chapter_index", 0),
            "start_sec":     meta.get("start_sec", 0.0),
            "end_sec":       meta.get("end_sec", 0.0),
            "distance":      round(dist, 4),
        })

    return output


def collection_size() -> int:
    try:
        return _get_collection().count()
    except Exception:
        return 0
