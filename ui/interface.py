"""
ui/interface.py
───────────────
Builds the Gradio interface. Four tabs:
  1. Process   — upload + run pipeline, progress log
  2. Chapters  — timestamped chapter cards with summaries
  3. Q&A       — RAG-powered question answering
  4. Export    — download JSON, Markdown, SRT
"""

from __future__ import annotations

import os
import time
from typing import List

import gradio as gr

from core.rag_qa    import answer as rag_answer
from core.segmenter import Chapter
from utils.pipeline import run_pipeline
from utils.logger   import get_logger

log = get_logger(__name__)

# ── Module-level state (per-session in HF Spaces) ────────────────────────────
_state: dict = {
    "chapters":   [],
    "overall":    "",
    "markdown":   "",
    "json_path":  "",
    "srt":        "",
    "ready":      False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chapters_to_html(chapters: List[Chapter]) -> str:
    if not chapters:
        return "<p style='color:gray'>No chapters yet. Upload and process a lecture first.</p>"

    cards = []
    for ch in chapters:
        cards.append(f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:16px;margin-bottom:14px;background:#fafafa">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
    <span style="font-weight:600;font-size:15px">{ch.index}. {ch.title}</span>
    <span style="font-size:12px;color:#888;background:#efefef;padding:2px 8px;border-radius:12px">
      {ch.start_ts()} → {ch.end_ts()}
    </span>
  </div>
  <p style="margin:0;font-size:14px;color:#444;line-height:1.6">{ch.summary}</p>
</div>""")
    return "\n".join(cards)


def _sources_to_md(sources: list) -> str:
    if not sources:
        return ""
    lines = ["\n**Sources retrieved:**"]
    for s in sources:
        lines.append(f"- **{s['chapter']}** `{s['timestamp']}` — _{s['excerpt']}_")
    return "\n".join(lines)


# ── Pipeline runner (called from Gradio) ─────────────────────────────────────

def process_media(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return (
            "Please upload a file first.",
            "<p style='color:gray'>No chapters yet.</p>",
            "", "", None, gr.update(interactive=False),
        )

    log_lines = []

    def cb(msg):
        log_lines.append(msg)
        progress(0, desc=msg)

    try:
        chapters, overall, markdown, json_path, srt = run_pipeline(
            file_obj.name, progress_cb=cb
        )
        _state.update({
            "chapters":  chapters,
            "overall":   overall,
            "markdown":  markdown,
            "json_path": json_path,
            "srt":       srt,
            "ready":     True,
        })

        # Write SRT to a temp file for download
        srt_path = os.path.join("data/outputs", "chapters.srt")
        with open(srt_path, "w") as f:
            f.write(srt)

        md_path = os.path.join("data/outputs", "summary.md")
        with open(md_path, "w") as f:
            f.write(markdown)

        log_text  = "\n".join(log_lines)
        status    = f"Done! {len(chapters)} chapters detected."
        ch_html   = _chapters_to_html(chapters)

        return (
            status,
            ch_html,
            overall,
            markdown,
            json_path,
            gr.update(interactive=True),
        )

    except Exception as e:
        log.exception("Pipeline error")
        return (
            f"Error: {e}",
            "<p style='color:red'>Processing failed.</p>",
            "", "", None,
            gr.update(interactive=False),
        )


def ask_question(question: str):
    if not _state["ready"]:
        return "Please process a lecture first.", ""
    if not question.strip():
        return "Please enter a question.", ""

    result  = rag_answer(question)
    answer  = result["answer"]
    sources = _sources_to_md(result["sources"])
    mode    = result["mode"]
    footer  = f"\n\n*Answer mode: {mode}*"
    return answer + footer, sources


# ── UI builder ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Lecture Summarizer",
        theme=gr.themes.Soft(primary_hue="violet"),
        css="""
        .tab-nav button { font-size: 14px !important; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# Lecture Summarizer\n"
            "Upload a lecture audio or video → get timestamped chapter summaries "
            "and ask questions with RAG-powered Q&A."
        )

        with gr.Tabs():

            # ── Tab 1: Process ────────────────────────────────────────────────
            with gr.Tab("Process"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_input = gr.File(
                            label="Upload lecture (MP3, WAV, MP4, MKV…)",
                            file_types=[".mp3", ".wav", ".mp4", ".mkv", ".m4a",
                                        ".avi", ".mov", ".webm", ".ogg"],
                        )
                        run_btn = gr.Button("Summarise lecture", variant="primary")
                        status  = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        overall_box = gr.Textbox(
                            label="Overall summary",
                            lines=5,
                            interactive=False,
                            placeholder="Overall summary will appear here…",
                        )

            # ── Tab 2: Chapters ───────────────────────────────────────────────
            with gr.Tab("Chapters"):
                ch_html = gr.HTML(
                    value="<p style='color:gray'>Process a lecture to see chapters.</p>"
                )

            # ── Tab 3: Q&A ────────────────────────────────────────────────────
            with gr.Tab("Q&A"):
                gr.Markdown(
                    "Ask any question about the lecture. "
                    "Answers are retrieved from the transcript via ChromaDB."
                )
                question_box = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is backpropagation? When was gradient descent discussed?",
                    lines=2,
                )
                ask_btn   = gr.Button("Ask", variant="primary", interactive=False)
                ans_box   = gr.Markdown(label="Answer")
                src_box   = gr.Markdown(label="Sources")

            # ── Tab 4: Export ─────────────────────────────────────────────────
            with gr.Tab("Export"):
                gr.Markdown("Download processed outputs:")
                with gr.Row():
                    json_dl = gr.File(label="JSON (full structured output)")
                    md_box  = gr.Textbox(
                        label="Markdown summary (copy or download)",
                        lines=20,
                        interactive=False,
                    )

        # ── Wiring ────────────────────────────────────────────────────────────

        run_btn.click(
            fn=process_media,
            inputs=[file_input],
            outputs=[status, ch_html, overall_box, md_box, json_dl, ask_btn],
        )

        ask_btn.click(
            fn=ask_question,
            inputs=[question_box],
            outputs=[ans_box, src_box],
        )

        question_box.submit(
            fn=ask_question,
            inputs=[question_box],
            outputs=[ans_box, src_box],
        )

    return demo
