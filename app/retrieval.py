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


_STOPWORDS = frozenset("""a an and are as at be been but by did do does for from had has have he her his i in is it
its of on or she that the their them they this to was were what when where which who whom why will with you your
how into than then there these those also not no about after before over under can could would should may might
""".split())
_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    """A very light stemmer: enough to match 'landed'/'landing'/'lands' and plurals."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


class LexicalEmbedder:
    """Hashed bag-of-words vectors (unigrams + bigrams, stopwords removed, sublinear term frequency).

    Needs no model and almost no CPU or memory, which suits tiny servers (e.g. Render's free plan). It only matches
    shared words, not meaning; the verifier still does the actual fact-checking, so this only has to put the
    relevant passages near the top. Its cosine scores run lower than a neural model's, so the pipeline uses
    lexical thresholds with it (see Settings.thresholds_for_embedder)."""

    def __init__(self, dim: int = 2 ** 15):
        self.dim = dim

    def _features(self, text: str) -> list[str]:
        words = [_stem(w) for w in _TOKEN.findall(text.lower()) if w not in _STOPWORDS]
        return words + [f"{a} {b}" for a, b in zip(words, words[1:])]

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            counts: dict[int, int] = {}
            for feature in self._features(text):
                index = int.from_bytes(hashlib.md5(feature.encode()).digest()[:4], "little") % self.dim
                counts[index] = counts.get(index, 0) + 1
            for index, count in counts.items():
                out[row, index] = 1.0 + np.log(count)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


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
