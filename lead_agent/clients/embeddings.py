"""Text similarity for ranking (Stage 7).

With an OpenAI key, texts are compared with OpenAI embeddings. Without one, a simple
word-count vector is used, so the pipeline still runs (with rougher similarity).
"""
import math
import re
import zlib

from lead_agent import settings
from lead_agent.clients.http import request_json

WORD_VECTOR_SIZE = 1024


def embed(texts: list[str]) -> list[list[float]]:
    """One vector per text."""
    texts = [text.strip() or "-" for text in texts]
    if not texts:
        return []
    if settings.OPENAI_API_KEY:
        return _openai_embeddings(texts)
    return [_word_count_vector(text) for text in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _openai_embeddings(texts: list[str]) -> list[list[float]]:
    answer = request_json(
        "POST",
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
        body={"model": settings.EMBEDDING_MODEL, "input": texts},
    )
    return [item["embedding"] for item in sorted(answer["data"], key=lambda item: item["index"])]


def _word_count_vector(text: str) -> list[float]:
    """Count words into a fixed-size vector (a stable hash picks each word's slot)."""
    vector = [0.0] * WORD_VECTOR_SIZE
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vector[zlib.crc32(word.encode()) % WORD_VECTOR_SIZE] += 1.0
    return vector
