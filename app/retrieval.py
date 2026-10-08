"""Split evidence into passages and rank them by semantic similarity."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str  # article title
    url: str
    text: str


def chunk_text(text: str, source: str, url: str, chunk_words: int = 80, overlap: int = 20) -> list[Chunk]:
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words")
    words = re.sub(r"\s+", " ", text).strip().split(" ")
    if words == [""]:
        return []
    chunks = []
    for start in range(0, len(words), chunk_words - overlap):
        piece = " ".join(words[start:start + chunk_words])
        chunk_id = hashlib.sha1(f"{source}|{start}|{piece}".encode()).hexdigest()[:16]
        chunks.append(Chunk(chunk_id, source, url, piece))
        if start + chunk_words >= len(words):
            break
    return chunks


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 vectors, one row per text."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return np.asarray(vectors, dtype=np.float32)


class FastEmbedEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_name)
        embeddings = list(self._model.embed(texts))
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms



def top_k(query_vector: np.ndarray, vectors: np.ndarray, k: int) -> list[tuple[int, float]]:
    """Indices and cosine similarities of the k nearest vectors. Uses FAISS when installed, NumPy otherwise."""
    if len(vectors) == 0:
        return []
    k = min(k, len(vectors))
    query = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    try:
        import faiss
    except ImportError:
        sims = vectors @ query[0]
        return [(int(i), float(sims[i])) for i in np.argsort(-sims)[:k]]
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    scores, ids = index.search(query, k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
