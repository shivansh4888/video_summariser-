"""
core/summariser.py
──────────────────
Generates:
  - A 2–3 sentence summary for each chapter (BART)
  - A short descriptive title for each chapter (extractive, then NLP cleanup)
  - An overall lecture summary (combining chapter summaries)
"""

from __future__ import annotations

import re
from typing import List

from transformers import pipeline

import config
from core.segmenter import Chapter
from utils.logger import get_logger

log = get_logger(__name__)

_summarizer = None


def _get_summarizer():
    global _summarizer
    if _summarizer is None:
        log.info(f"Loading summariser: {config.SUMMARIZER_MODEL}")
        _summarizer = pipeline(
            "summarization",
            model=config.SUMMARIZER_MODEL,
            device=-1,          # CPU; swap to 0 for GPU
        )
        log.info("Summariser loaded.")
    return _summarizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _truncate(text: str, max_words: int = 700) -> str:
    """BART has a 1024-token input limit; truncate to be safe."""
    words = text.split()
    return " ".join(words[:max_words])


def _clean_summary(text: str) -> str:
    text = text.strip()
    if not text.endswith((".", "!", "?")):
        text += "."
    return text


def _extract_title(summary: str, chapter_text: str) -> str:
    """
    Derive a 4–7 word title from the summary.
    Strategy: take the first sentence, strip filler words, trim to ~6 words.
    """
    first_sent = re.split(r'[.!?]', summary)[0].strip()
    stopwords  = {"the","a","an","is","are","was","were","this","that",
                  "these","those","in","on","at","to","for","of","and",
                  "or","but","it","its","by","with","as","be","been"}
    words      = [w for w in first_sent.split() if w.lower() not in stopwords]
    title      = " ".join(words[:6])
    return title.capitalize() if title else "Overview"


# ── Public API ────────────────────────────────────────────────────────────────

def summarise_chapter(chapter: Chapter) -> Chapter:
    """In-place: adds summary and title to a Chapter. Returns it."""
    summarizer = _get_summarizer()
    truncated  = _truncate(chapter.text)

    if len(truncated.split()) < 30:
        chapter.summary = truncated
        chapter.title   = _extract_title(truncated, chapter.text)
        return chapter

    result = summarizer(
        truncated,
        max_length = config.SUMMARY_MAX_TOKENS,
        min_length = config.SUMMARY_MIN_TOKENS,
        do_sample  = False,
    )
    chapter.summary = _clean_summary(result[0]["summary_text"])
    chapter.title   = _extract_title(chapter.summary, chapter.text)
    log.debug(f"Chapter {chapter.index} summary: {chapter.summary[:80]}…")
    return chapter


def summarise_all(chapters: List[Chapter]) -> List[Chapter]:
    """Summarise every chapter and return the updated list."""
    log.info(f"Summarising {len(chapters)} chapters…")
    for ch in chapters:
        summarise_chapter(ch)
    return chapters


def overall_summary(chapters: List[Chapter]) -> str:
    """
    Produce an overall lecture summary by summarising the chapter summaries.
    """
    combined = " ".join(ch.summary for ch in chapters if ch.summary)
    if not combined.strip():
        return "No summary available."

    summarizer = _get_summarizer()
    truncated  = _truncate(combined, max_words=600)

    if len(truncated.split()) < 30:
        return _clean_summary(combined)

    result = summarizer(
        truncated,
        max_length = 250,
        min_length = 80,
        do_sample  = False,
    )
    return _clean_summary(result[0]["summary_text"])
