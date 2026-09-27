"""임베딩 프로바이더. 기본은 CPU에서 도는 fastembed(ONNX) 다국어 모델이다."""

import hashlib
from functools import lru_cache

import numpy as np

from app.config import get_settings


@lru_cache
def _fastembed():
    from fastembed import TextEmbedding

    s = get_settings()
    return TextEmbedding(s.embedding_model, cache_dir=s.embedding_cache_dir)


def _mock_embed(text: str, dim: int) -> list[float]:
    """테스트용 가짜 임베딩. 글자 3-gram 해시를 벡터에 더해 비슷한 문장이 비슷한 벡터가 되게 한다."""
    v = np.zeros(dim)
    t = text.replace(" ", "")
    for i in range(max(len(t) - 2, 1)):
        h = int(hashlib.md5(t[i : i + 3].encode()).hexdigest(), 16)
        v[h % dim] += 1
    n = np.linalg.norm(v)
    return (v / n if n else v).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    s = get_settings()
    if not texts:
        return []
    if s.embedding_provider == "mock":
        return [_mock_embed(t, s.embedding_dim) for t in texts]
    vecs = list(_fastembed().embed(texts))
    return [(v / np.linalg.norm(v)).tolist() for v in vecs]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    d = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(va @ vb / d) if d else 0.0
