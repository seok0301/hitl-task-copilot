"""검색 방식 비교 실험(LLM 없이). 실행: `uv run python -m eval.retrieval_lab [--sets hard] [--models ...]`

교훈 태스크마다 근거 문서가 검색 결과 문서 안에 들어왔는지(Recall)를 본다.
DB 없이 seed 데이터로 문서와 청크를 메모리에 만들어, 임베딩 모델과 검색 단위를 바꿔 가며 비교한다.
"""

import argparse
import math
from collections import Counter

import numpy as np
from fastembed import TextEmbedding

from app.config import get_settings
from app.rag.ingest import chunk_text
from app.seed.company import DOCUMENTS, PAST_PROJECTS, USERS
from app.seed.seed import retrospective
from eval.plan_sets import plan_sets


def corpus(max_level: int = 2) -> list[tuple[str, str]]:
    names = {u[0]: u[1] for u in USERS}
    docs = [
        (f"[회고] {title}", retrospective(title, goal, year, tasks, lessons, names))
        for _pid, year, title, goal, tasks, lessons in PAST_PROJECTS
    ]
    docs += [(t, c) for t, lv, c in DOCUMENTS if lv <= max_level]
    return docs


def bigrams(text: str) -> list[str]:
    t = "".join(text.split())
    return [t[i : i + 2] for i in range(len(t) - 1)]


class BM25:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75):
        self.toks = [Counter(bigrams(t)) for t in texts]
        self.lens = [sum(c.values()) for c in self.toks]
        self.avg = sum(self.lens) / len(self.lens)
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


def embedder(model: str):
    m = TextEmbedding(model, cache_dir=get_settings().embedding_cache_dir)
    e5 = "e5" in model

    def enc(texts: list[str], query: bool) -> np.ndarray:
        if e5:
            texts = [("query: " if query else "passage: ") + t for t in texts]
        v = np.array(list(m.embed(texts)))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    return enc


def rrf(*rank_lists: list[int], k: int = 60) -> list[int]:
    score: dict[int, float] = {}
    for ranks in rank_lists:
        for r, i in enumerate(ranks):
            score[i] = score.get(i, 0) + 1 / (k + r + 1)
    return sorted(score, key=lambda i: -score[i])


def draft_queries(projects: list[dict]) -> list[list[str]]:
    """RAG 없이 계획 초안을 만들어, 초안의 태스크 문장을 추가 검색어로 쓴다."""
    from concurrent.futures import ThreadPoolExecutor

    from app.agents.planner import plan_tasks
    from app.db import SessionLocal
    from app.models import Role

    def one(p: dict) -> list[str]:
        with SessionLocal() as db:
            tasks, _ = plan_tasks(db, p["title"], p["goal"], p["days"], Role.MANAGER, use_rag=False)
        return [f"{t['title']}. {t['description']}" for t in tasks]

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, projects))


def run(sets: list[str], models: list[str], expand: bool = False) -> None:
    docs = corpus()
    titles = [t for t, _ in docs]
    chunks = [(di, f"{docs[di][0]}\n{p}") for di in range(len(docs)) for p in chunk_text(docs[di][1])]
    bm_doc = BM25([f"{t}\n{c}" for t, c in docs])
    projects = [p for s in sets for p in plan_sets()[s]]
    queries = [f"{p['title']}. {p['goal']}" for p in projects]
    lessons = [(qi, r) for qi, p in enumerate(projects) for r in p["ref"] if r["lesson"] and r["gold_docs"]]
    drafts = draft_queries(projects) if expand else [[] for _ in projects]
    print(
        f"문서 {len(docs)}개, 청크 {len(chunks)}개, 프로젝트 {len(projects)}개, 교훈 태스크 {len(lessons)}개"
    )

    for model in models:
        enc = embedder(model)
        cv = enc([c for _, c in chunks], query=False)
        dv = enc([f"{t}\n{c}" for t, c in docs], query=False)
        qv = enc(queries, query=True)
        results: dict[str, list[list[int]]] = {}
        for qi, q in enumerate(queries):
            cs = cv @ qv[qi]
            corder = np.argsort(-cs)
            # 현재 방식: 청크 상위 6개가 속한 문서
            results.setdefault("chunk6", []).append(list(dict.fromkeys(chunks[i][0] for i in corder[:6])))
            # 문서 단위: 문서별 최고 청크 점수
            best: dict[int, float] = {}
            for i in corder:
                best.setdefault(chunks[i][0], float(cs[i]))
            dense_docs = sorted(best, key=lambda d: -best[d])
            whole = list(np.argsort(-(dv @ qv[qi])))
            lex = list(np.argsort(-bm_doc.scores(q)))
            for k in (3, 4, 6):
                results.setdefault(f"doc_maxchunk@{k}", []).append(dense_docs[:k])
                results.setdefault(f"doc_whole@{k}", []).append([int(x) for x in whole[:k]])
                results.setdefault(f"bm25@{k}", []).append([int(x) for x in lex[:k]])
                results.setdefault(f"hybrid_rrf@{k}", []).append(rrf(dense_docs, [int(x) for x in lex])[:k])
            if expand:
                # 초안 태스크마다 검색한 순위를 RRF로 합친다. 프로젝트 질의도 순위 하나로 넣는다.
                sub = [q, *drafts[qi]]
                sv = enc(sub, query=True)
                d_ranks = [[int(x) for x in np.argsort(-(dv @ v))] for v in sv]
                l_ranks = [[int(x) for x in np.argsort(-bm_doc.scores(t))] for t in sub]
                for k in (3, 4, 6):
                    results.setdefault(f"multi_dense@{k}", []).append(rrf(*d_ranks)[:k])
                    results.setdefault(f"multi_bm25@{k}", []).append(rrf(*l_ranks)[:k])
                    results.setdefault(f"multi_hybrid@{k}", []).append(rrf(*d_ranks, *l_ranks)[:k])
        print(f"\n## {model}")
        for name, per_q in results.items():
            hit = sum(1 for qi, r in lessons if set(r["gold_docs"]) & {titles[d] for d in per_q[qi]})
            ndocs = np.mean([len(x) for x in per_q])
            print(
                f"{name:20s} recall={hit / len(lessons):.3f} ({hit}/{len(lessons)})  평균 문서 수 {ndocs:.1f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=["hard"])
    ap.add_argument(
        "--models",
        nargs="*",
        default=[
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        ],
    )
    ap.add_argument("--expand", action="store_true")
    a = ap.parse_args()
    run(a.sets, a.models, a.expand)
