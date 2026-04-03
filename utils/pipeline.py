"""
utils/pipeline.py
─────────────────
Orchestrates the full processing pipeline:
  audio/video → transcribe → segment → summarise → ingest → done

Exposes a single callable used by the Gradio UI.
"""

from __future__ import annotations

import os
import tempfile
from typing import Callable, List, Tuple

from core.transcriber  import transcribe, extract_audio_from_video
from core.segmenter    import segment, Chapter
from core.summariser   import summarise_all, overall_summary
from core.vector_store import reset_collection, ingest_chapters
from utils.exporter    import to_json, to_markdown, to_srt
from utils.logger      import get_logger

import config

log = get_logger(__name__)


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}


def run_pipeline(
    media_path:    str,
    progress_cb:   Callable[[str], None] | None = None,
) -> Tuple[List[Chapter], str, str, str, str]:
    """
    Full pipeline.

    Returns
    -------
    chapters       : List[Chapter]
    overall        : str   — overall lecture summary
    markdown       : str   — formatted markdown string
    json_path      : str   — path to exported JSON file
    srt_content    : str   — SRT subtitle string
    """
    def _log(msg: str):
        log.info(msg)
        if progress_cb:
            progress_cb(msg)

    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Extract audio if video ───────────────────────────────────────
    audio_path = media_path
    if _is_video(media_path):
        _log("Extracting audio from video…")
        audio_path = extract_audio_from_video(media_path)

    # ── Step 2: Transcribe ───────────────────────────────────────────────────
    _log("Transcribing with Whisper… (this may take a minute)")
    segments = transcribe(audio_path)
    _log(f"Transcription done: {len(segments)} segments")

    # ── Step 3: Topic segmentation ───────────────────────────────────────────
    _log("Detecting topic boundaries…")
    chapters = segment(segments)
    _log(f"Segmented into {len(chapters)} chapters")

    # ── Step 4: Summarise chapters ───────────────────────────────────────────
    _log("Summarising chapters…")
    chapters  = summarise_all(chapters)
    overall   = overall_summary(chapters)
    _log("Summaries done")

    # ── Step 5: Ingest into ChromaDB ─────────────────────────────────────────
    _log("Ingesting into ChromaDB for Q&A…")
    reset_collection()
    n_chunks = ingest_chapters(chapters)
    _log(f"Ingested {n_chunks} chunks into vector store")

    # ── Step 6: Export ───────────────────────────────────────────────────────
    source_name = os.path.basename(media_path)
    json_path   = to_json(chapters, overall, source_name)
    markdown    = to_markdown(chapters, overall)
    srt_content = to_srt(chapters)

    _log("Pipeline complete.")

    # Clean up temp audio file if we extracted it
    if _is_video(media_path) and audio_path != media_path:
        try:
            os.unlink(audio_path)
        except OSError:
            pass

    return chapters, overall, markdown, json_path, srt_content
