"""
core/segmenter.py
─────────────────
Detects topic-shift boundaries in a transcript using cosine similarity
between adjacent sentence-embedding windows.

Algorithm:
  1. Sentence-tokenise the transcript.
  2. For each position i, compute the average embedding of a window of
     sentences before and after i.
  3. Cosine distance between the two windows → boundary score.
  4. Positions where the score exceeds SEGMENT_THRESHOLD become chapter
     boundaries.
  5. Short segments are merged into their neighbour.

Returns a list of Chapter objects with timestamps and sentence spans.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from utils.logger import get_logger

log = get_logger(__name__)

_embed_model = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info(f"Loading embedding model: {config.EMBED_MODEL}")
        _embed_model = SentenceTransformer(config.EMBED_MODEL)
        log.info("Embedding model loaded.")
    return _embed_model


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Chapter:
    index:      int
    title:      str          # auto-generated placeholder; replaced by summariser
    start_sec:  float
    end_sec:    float
    sentences:  List[str]    = field(default_factory=list)
    summary:    str          = ""

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def start_ts(self) -> str:
        return _fmt_ts(self.start_sec)

    def end_ts(self) -> str:
        return _fmt_ts(self.end_sec)


def _fmt_ts(sec: float) -> str:
    sec = int(sec)
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


# ── Sentence tokeniser (no NLTK dependency) ───────────────────────────────────

_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ── Boundary detection ────────────────────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def _window_embed(embeddings: np.ndarray, idx: int, w: int) -> np.ndarray:
    start = max(0, idx - w)
    end   = min(len(embeddings), idx + w)
    return embeddings[start:idx].mean(axis=0), embeddings[idx:end].mean(axis=0)


def detect_boundaries(sentences: List[str]) -> List[int]:
    """Return indices where a new topic begins."""
    if len(sentences) < config.WINDOW_SIZE * 2 + 1:
        return []

    model = _get_embed_model()
    log.info(f"Embedding {len(sentences)} sentences for boundary detection…")
    embeddings = model.encode(sentences, batch_size=32, show_progress_bar=False)

    scores = []
    w = config.WINDOW_SIZE
    for i in range(w, len(sentences) - w):
        before, after = _window_embed(embeddings, i, w)
        scores.append((i, _cosine_distance(before, after)))

    if not scores:
        return []

    # Smooth scores with a 3-point moving average
    raw = [s for _, s in scores]
    smoothed = np.convolve(raw, np.ones(3) / 3, mode="same")

    boundaries = [
        scores[i][0]
        for i, score in enumerate(smoothed)
        if score > config.SEGMENT_THRESHOLD
    ]

    # Suppress boundaries that are too close together (< 5 sentences apart)
    filtered = []
    prev = -10
    for b in boundaries:
        if b - prev >= 5:
            filtered.append(b)
            prev = b

    log.info(f"Detected {len(filtered)} topic boundaries")
    return filtered


# ── Map sentences back to timestamps ─────────────────────────────────────────

def _sentence_timestamps(
    sentences: List[str],
    segments:  List[Dict[str, Any]],
) -> List[float]:
    """
    Approximate start timestamp for each sentence by aligning
    character offsets into the flat transcript against Whisper segments.
    """
    flat = " ".join(s["text"] for s in segments)
    # Build a char-offset → timestamp map from Whisper segments
    ts_map: List[tuple] = []
    pos = 0
    for seg in segments:
        ts_map.append((pos, seg["start"]))
        pos += len(seg["text"]) + 1

    def char_to_ts(char_pos: int) -> float:
        for i in range(len(ts_map) - 1):
            if ts_map[i][0] <= char_pos < ts_map[i + 1][0]:
                return ts_map[i][1]
        return ts_map[-1][1] if ts_map else 0.0

    result = []
    cursor = 0
    for sent in sentences:
        idx = flat.find(sent[:30], cursor)   # anchor on first 30 chars
        result.append(char_to_ts(max(idx, cursor)))
        cursor = max(idx + 1, cursor)
    return result


# ── Main entry ────────────────────────────────────────────────────────────────

def segment(segments: List[Dict[str, Any]]) -> List[Chapter]:
    """
    Takes Whisper segments → returns Chapter list with timestamps.
    """
    full_text = " ".join(s["text"] for s in segments)
    sentences = _split_sentences(full_text)

    if not sentences:
        return []

    boundaries  = detect_boundaries(sentences)
    sent_times  = _sentence_timestamps(sentences, segments)
    total_end   = segments[-1]["end"] if segments else 0.0

    # Build chapter slices
    split_points = [0] + boundaries + [len(sentences)]
    chapters: List[Chapter] = []

    for i in range(len(split_points) - 1):
        s_start = split_points[i]
        s_end   = split_points[i + 1]
        sents   = sentences[s_start:s_end]

        if len(" ".join(sents).split()) < config.MIN_SEGMENT_WORDS and i > 0:
            # Merge into previous chapter
            chapters[-1].sentences.extend(sents)
            chapters[-1].end_sec = (
                sent_times[s_end - 1] if s_end < len(sent_times) else total_end
            )
            continue

        chapters.append(Chapter(
            index     = len(chapters) + 1,
            title     = f"Chapter {len(chapters) + 1}",
            start_sec = sent_times[s_start] if s_start < len(sent_times) else 0.0,
            end_sec   = sent_times[s_end - 1] if s_end - 1 < len(sent_times) else total_end,
            sentences = sents,
        ))

    # Fix last chapter end time
    if chapters:
        chapters[-1].end_sec = total_end

    log.info(f"Segmented into {len(chapters)} chapters")
    return chapters
