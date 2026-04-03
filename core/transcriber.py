"""
core/transcriber.py
───────────────────
Transcribes audio/video using OpenAI Whisper via HuggingFace pipeline.
Returns a list of timed segments:
  [{"start": 0.0, "end": 4.2, "text": "Welcome to the lecture..."}]
"""

import os
import tempfile
from typing import List, Dict, Any

import torch
from transformers import pipeline

import config
from utils.logger import get_logger

log = get_logger(__name__)


def _get_device() -> int:
    """Return 0 for GPU, -1 for CPU."""
    return 0 if torch.cuda.is_available() else -1


_pipe = None  # lazy-load so import is fast


def _load_pipe():
    global _pipe
    if _pipe is None:
        log.info(f"Loading Whisper model: {config.WHISPER_MODEL}")
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=config.WHISPER_MODEL,
            chunk_length_s=config.WHISPER_CHUNK_SEC,
            batch_size=config.WHISPER_BATCH_SIZE,
            return_timestamps=True,
            device=_get_device(),
        )
        log.info("Whisper model loaded.")
    return _pipe


def transcribe(audio_path: str) -> List[Dict[str, Any]]:
    """
    Transcribe an audio or video file.

    Parameters
    ----------
    audio_path : str
        Path to audio/video file (mp3, wav, mp4, etc.)

    Returns
    -------
    List of dicts with keys: start (float), end (float), text (str)
    """
    pipe = _load_pipe()
    log.info(f"Transcribing: {audio_path}")

    result = pipe(
        audio_path,
        generate_kwargs={"language": "english", "task": "transcribe"},
    )

    # Whisper pipeline returns {"text": "...", "chunks": [{"timestamp": (s,e), "text":"..."}]}
    chunks = result.get("chunks", [])

    if not chunks:
        # Fallback: wrap full text as single segment
        return [{"start": 0.0, "end": 0.0, "text": result.get("text", "").strip()}]

    segments = []
    for chunk in chunks:
        ts = chunk.get("timestamp", (0.0, 0.0))
        segments.append({
            "start": float(ts[0] or 0.0),
            "end":   float(ts[1] or 0.0),
            "text":  chunk["text"].strip(),
        })

    log.info(f"Transcription complete: {len(segments)} segments")
    return segments


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio track from a video file using ffmpeg.
    Returns path to a temp .wav file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    cmd = (
        f'ffmpeg -y -i "{video_path}" '
        f'-ar 16000 -ac 1 -vn "{tmp.name}" -loglevel error'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with code {ret}. Is ffmpeg installed?")
    return tmp.name


def full_text(segments: List[Dict]) -> str:
    """Flatten segment list to a single string."""
    return " ".join(s["text"] for s in segments)
