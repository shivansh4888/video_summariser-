import os

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_MODEL       = os.getenv("WHISPER_MODEL", "openai/whisper-base")   # swap to "small" for better accuracy
WHISPER_CHUNK_SEC   = 30        # audio chunk length fed to Whisper
WHISPER_BATCH_SIZE  = 8

# ── Sentence-Transformers ──────────────────────────────────────────────────────
EMBED_MODEL         = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM           = 384

# ── Topic segmentation ─────────────────────────────────────────────────────────
# Cosine distance above this threshold → new chapter boundary
SEGMENT_THRESHOLD   = 0.35
MIN_SEGMENT_WORDS   = 80        # ignore segments shorter than this
WINDOW_SIZE         = 3         # sentences averaged for boundary detection

# ── ChromaDB ───────────────────────────────────────────────────────────────────
CHROMA_PATH         = os.getenv("CHROMA_PATH", "data/chroma_store")
COLLECTION_NAME     = "lecture_chunks"

# ── RAG / Q&A ─────────────────────────────────────────────────────────────────
TOP_K_CHUNKS        = 5         # chunks retrieved per query
CHUNK_SIZE_WORDS    = 120       # target words per chunk stored in Chroma
CHUNK_OVERLAP_WORDS = 20

# ── Summarisation (via HF Inference API or local) ─────────────────────────────
SUMMARIZER_MODEL    = "facebook/bart-large-cnn"
SUMMARY_MAX_TOKENS  = 180
SUMMARY_MIN_TOKENS  = 60

# ── Paths ──────────────────────────────────────────────────────────────────────
UPLOAD_DIR          = "data/uploads"
OUTPUT_DIR          = "data/outputs"
