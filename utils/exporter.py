"""
utils/exporter.py
─────────────────
Export processed lecture data to various formats.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List

from core.segmenter import Chapter, _fmt_ts
import config


def _ensure_output_dir():
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)


def to_json(chapters: List[Chapter], overall: str, source_name: str = "") -> str:
    """Export full structured output as JSON. Returns file path."""
    _ensure_output_dir()
    data = {
        "source":       source_name,
        "processed_at": datetime.utcnow().isoformat(),
        "overall_summary": overall,
        "chapters": [
            {
                "index":      ch.index,
                "title":      ch.title,
                "start":      ch.start_ts(),
                "end":        ch.end_ts(),
                "summary":    ch.summary,
                "word_count": ch.word_count,
                "transcript": ch.text,
            }
            for ch in chapters
        ],
    }
    path = os.path.join(config.OUTPUT_DIR, "lecture_output.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def to_markdown(chapters: List[Chapter], overall: str) -> str:
    """Export as a readable Markdown string (returned, not saved)."""
    lines = ["# Lecture Summary\n", f"> {overall}\n", "---\n"]
    for ch in chapters:
        lines.append(f"## {ch.index}. {ch.title}  `{ch.start_ts()} → {ch.end_ts()}`\n")
        lines.append(f"{ch.summary}\n")
    return "\n".join(lines)


def to_srt(chapters: List[Chapter]) -> str:
    """
    Export chapter markers as an SRT subtitle file string.
    Each chapter becomes a single subtitle entry.
    """
    lines = []
    for i, ch in enumerate(chapters, 1):
        lines.append(str(i))
        lines.append(f"{_srt_ts(ch.start_sec)} --> {_srt_ts(ch.end_sec)}")
        lines.append(f"[{ch.title}]")
        lines.append("")
    return "\n".join(lines)


def _srt_ts(sec: float) -> str:
    sec   = max(0.0, sec)
    h     = int(sec // 3600)
    m     = int((sec % 3600) // 60)
    s     = int(sec % 60)
    ms    = int((sec - int(sec)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
