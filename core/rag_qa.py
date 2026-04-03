"""
core/rag_qa.py
──────────────
RAG (Retrieval-Augmented Generation) Q&A pipeline.

Flow:
  1. Embed the user's question.
  2. Retrieve top-K relevant chunks from ChromaDB.
  3. Build a context string from those chunks.
  4. Pass [context + question] to a generative model for the final answer.
     - On CPU / free HF Spaces: uses extractive fallback (no heavy LLM needed)
     - On GPU / with HF_TOKEN: can swap to a real generative model

The extractive fallback scores each retrieved sentence by keyword overlap
with the question and returns the top-scoring sentences — good enough for
demo purposes and runs in milliseconds on CPU.
"""

from __future__ import annotations

import os
import re
from typing import List, Dict, Any, Tuple

from core.vector_store import query as chroma_query
from core.segmenter import _fmt_ts
from utils.logger import get_logger

import config

log = get_logger(__name__)

# Optional: set HF_TOKEN env var to enable the generative path
HF_TOKEN = os.getenv("HF_TOKEN", "")


# ── Extractive fallback (always available, no GPU needed) ────────────────────

def _keyword_score(sentence: str, question: str) -> float:
    q_words = set(re.findall(r'\w+', question.lower()))
    s_words = set(re.findall(r'\w+', sentence.lower()))
    if not q_words:
        return 0.0
    return len(q_words & s_words) / len(q_words)


def _extractive_answer(chunks: List[Dict], question: str, top_n: int = 3) -> str:
    """
    Score every sentence in the retrieved chunks by keyword overlap
    with the question. Return the top_n highest-scoring sentences.
    """
    scored: List[Tuple[float, str, str]] = []
    for chunk in chunks:
        sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
        for sent in sentences:
            if len(sent.split()) < 5:
                continue
            score = _keyword_score(sent, question)
            ts    = _fmt_ts(chunk["start_sec"])
            scored.append((score, sent.strip(), chunk["chapter_title"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    if not top or top[0][0] == 0.0:
        return "I couldn't find a specific answer to that in the lecture. Try rephrasing or asking about a broader topic."

    parts = []
    for _, sent, ch_title in top:
        parts.append(f"• {sent}  *(from: {ch_title})*")
    return "\n\n".join(parts)


# ── Generative path (optional, requires HF_TOKEN + internet on HF Spaces) ────

def _generative_answer(context: str, question: str) -> str:
    """
    Call HF Inference API with a small instruction-tuned model.
    Only used when HF_TOKEN is set.
    """
    try:
        import requests
        prompt = (
            f"Context from a lecture transcript:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely based only on the context above:"
        )
        resp = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 200}},
            timeout=30,
        )
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").split("Answer concisely")[-1].strip()
    except Exception as e:
        log.warning(f"Generative answer failed: {e}. Falling back to extractive.")
    return ""


# ── Public API ────────────────────────────────────────────────────────────────

def answer(question: str) -> Dict[str, Any]:
    """
    Answer a question about the lecture.

    Returns:
        {
          "answer":   str,
          "sources":  [{"chapter": str, "timestamp": str, "excerpt": str}],
          "mode":     "generative" | "extractive",
        }
    """
    if not question.strip():
        return {"answer": "Please enter a question.", "sources": [], "mode": "none"}

    chunks = chroma_query(question)
    if not chunks:
        return {
            "answer":  "The lecture hasn't been processed yet, or no relevant content was found.",
            "sources": [],
            "mode":    "none",
        }

    # Build sources list for UI display
    sources = []
    seen    = set()
    for ch in chunks:
        key = ch["chapter_title"]
        if key not in seen:
            seen.add(key)
            sources.append({
                "chapter":   ch["chapter_title"],
                "timestamp": _fmt_ts(ch["start_sec"]),
                "excerpt":   ch["text"][:140] + "…",
            })

    # Try generative if token available
    ans_text = ""
    mode     = "extractive"
    if HF_TOKEN:
        context  = "\n\n".join(c["text"] for c in chunks)
        ans_text = _generative_answer(context, question)
        if ans_text:
            mode = "generative"

    if not ans_text:
        ans_text = _extractive_answer(chunks, question)

    return {"answer": ans_text, "sources": sources, "mode": mode}
