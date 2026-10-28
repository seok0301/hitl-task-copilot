"""글자 2-gram BM25. 한국어는 조사와 어미가 붙어 단어 단위 일치가 어려워서 글자 2-gram으로 자른다."""

import math
from collections import Counter

import numpy as np


def bigrams(text: str) -> list[str]:
    t = "".join(text.split())
    return [t[i : i + 2] for i in range(len(t) - 1)]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.toks = [Counter(bigrams(t)) for t in texts]
        self.lens = [sum(c.values()) for c in self.toks]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1.0
        df = Counter(w for c in self.toks for w in c)
        n = len(texts)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> np.ndarray:
        q = set(bigrams(query))
        out = []
        for c, ln in zip(self.toks, self.lens, strict=True):
            s = 0.0
            for w in q:
                f = c.get(w, 0)
                if f:
                    s += (
                        self.idf[w]
                        * f
                        * (self.k1 + 1)
                        / (f + self.k1 * (1 - self.b + self.b * ln / self.avg))
                    )
            out.append(s)
        return np.array(out)


def rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """여러 순위를 Reciprocal Rank Fusion으로 합친다. (항목, 점수)를 점수 순서로 돌려준다."""
    score: dict[int, float] = {}
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            score[i] = score.get(i, 0) + 1 / (k + r + 1)
    return sorted(score.items(), key=lambda x: -x[1])
