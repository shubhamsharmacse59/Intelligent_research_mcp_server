"""All settings in one place."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"        # uploaded PDFs go here
ANALYSES_DIR = BASE_DIR / "analyses"    # finished reports (cache)
INDEX_DIR = BASE_DIR / "index"          # FAISS index per paper
for d in (PAPERS_DIR, ANALYSES_DIR, INDEX_DIR):
    d.mkdir(exist_ok=True)

# ---- LLM: Ollama running Gemma 4 on the college server ----
# Reached through an SSH tunnel, so it looks local:
#   ssh -N -L 11434:localhost:11434 user@10.6.7.107
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma4:31b"     # use the exact name from `ollama list` on the server, e.g. "gemma4:31"
CONTEXT_WINDOW = 8192       # tokens Ollama lets the model read at once
TEMPERATURE = 0.2           # low = factual, less creative

# ---- Analysis ----
SECTION_CHARS = 9000        # max characters sent to the LLM per section (~2,300 tokens)

# ---- Retrieval (for questions) ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384-dim, runs on laptop CPU
CHUNK_WORDS = 200
OVERLAP_WORDS = 40
TOP_K = 5
