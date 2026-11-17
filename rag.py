"""
Step 3: semantic retrieval for follow-up questions (RAG).

paper text -> 200-word chunks -> embeddings (384-dim) -> FAISS index
question   -> embedding -> top-5 most similar chunks -> LLM answers from them
"""
import json
from pathlib import Path

import faiss
import numpy as np

import config
from llm import ask_llm
from pdf_utils import body_and_references, chunk_words, extract_text, split_sections

_model = None


def embed(texts: list[str]) -> np.ndarray:
    """Text -> normalised vectors, so dot product = cosine similarity."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model.encode(texts, normalize_embeddings=True).astype("float32")


def _build_index(pdf_path: Path):
    index_file = config.INDEX_DIR / f"{pdf_path.stem}.faiss"
    chunks_file = config.INDEX_DIR / f"{pdf_path.stem}.json"
    if index_file.exists():
        return faiss.read_index(str(index_file)), json.loads(chunks_file.read_text())

    body, _ = body_and_references(split_sections(extract_text(pdf_path)))
    chunks = []
    for section, text in body.items():
        for piece in chunk_words(text, config.CHUNK_WORDS, config.OVERLAP_WORDS):
            chunks.append({"section": section, "text": piece})

    vectors = embed([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])   # inner product on unit vectors = cosine
    index.add(vectors)
    faiss.write_index(index, str(index_file))
    chunks_file.write_text(json.dumps(chunks))
    return index, chunks


def retrieve(pdf_path: Path, question: str, top_k: int = config.TOP_K) -> list[dict]:
    index, chunks = _build_index(pdf_path)
    scores, ids = index.search(embed([question]), min(top_k, index.ntotal))
    return [{**chunks[i], "score": round(float(s), 3)} for s, i in zip(scores[0], ids[0]) if i != -1]


def answer(pdf_path: Path, question: str) -> dict:
    passages = retrieve(pdf_path, question)
    context = "\n\n".join(f"[{n}] ({p['section']}) {p['text']}" for n, p in enumerate(passages, 1))
    reply = ask_llm(
        "You answer questions about a research paper using ONLY the numbered passages. "
        "Cite passages like [1]. If the passages do not contain the answer, say so.",
        f"PASSAGES:\n{context}\n\nQUESTION: {question}",
    )
    return {"answer": reply, "sources": [
        {"n": n, "section": p["section"], "score": p["score"], "preview": p["text"][:200]}
        for n, p in enumerate(passages, 1)
    ]}
