---
title: Lecture Summarizer
emoji: 🎓

---

# Lecture Summarizer

Upload any lecture audio or video and get:

- **Timestamped chapter summaries** — automatic topic segmentation using sentence-transformer cosine similarity
- **RAG-powered Q&A** — ask questions about the lecture, answered from a ChromaDB vector store
- **Exportable outputs** — download structured JSON, Markdown summary, or SRT chapter markers

## Tech stack

| Component | Library |
|---|---|
| Transcription | OpenAI Whisper (`whisper-base`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Topic segmentation | Cosine distance windowing |
| Vector store | ChromaDB (persistent) |
| Summarisation | `facebook/bart-large-cnn` |
| RAG Q&A | ChromaDB retrieval + extractive / generative answer |
| UI | Gradio 4.x |

## Running locally

```bash
# 1. Clone
git clone https://huggingface.co/spaces/YOUR_USERNAME/lecture-summarizer
cd lecture-summarizer

# 2. Install ffmpeg (needed for video → audio extraction)
# Ubuntu/Debian:  sudo apt install ffmpeg
# macOS:          brew install ffmpeg

# 3. Install Python deps
pip install -r requirements.txt

# 4. Run
python app.py
```

## Optional: enable generative Q&A

Set `HF_TOKEN` environment variable to your Hugging Face token to enable
the Mistral-7B generative answering path. Without it, the app uses fast
extractive answering (no GPU needed).

```bash
export HF_TOKEN=hf_your_token_here
python app.py
```

## Project structure

```
lecture-summarizer/
├── app.py                  # Entry point
├── config.py               # All tunable constants
├── requirements.txt
├── README.md
│
├── core/
│   ├── transcriber.py      # Whisper transcription
│   ├── segmenter.py        # Topic boundary detection
│   ├── summariser.py       # BART per-chapter summaries
│   ├── vector_store.py     # ChromaDB ingest & retrieval
│   └── rag_qa.py           # RAG Q&A pipeline
│
├── ui/
│   └── interface.py        # Gradio UI
│
├── utils/
│   ├── pipeline.py         # Orchestrator
│   ├── exporter.py         # JSON / Markdown / SRT export
│   └── logger.py           # Logging helper
│
└── data/
    ├── chroma_store/       # Persistent ChromaDB files
    ├── uploads/            # Temp uploaded files
    └── outputs/            # Generated JSON, SRT, MD
```
