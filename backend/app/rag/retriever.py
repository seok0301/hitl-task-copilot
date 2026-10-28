"""접근 등급으로 거른 사내 문서 검색.

- search(): 청크 단위 벡터 검색(초기 방식).
- search_docs(): 문서 단위 글자 2-gram BM25. 질의를 여러 개 받아 순위를 RRF로 합친다.
  계획 에이전트는 프로젝트 목표와 초안 태스크들을 질의로 넘긴다.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.embeddings import embed_text
from app.models import ACCESS_LEVEL_BY_ROLE, DocChunk, Document, Role
from app.rag.lexical import BM25, rrf


@dataclass
class Hit:
    document_id: int
    title: str
    content: str
    access_level: int
    score: float


def search(db: Session, query: str, role: Role, k: int = 6) -> list[Hit]:
    """역할이 볼 수 있는 등급 이하의 청크만 검색한다."""
    qv = embed_text(query)
    dist = DocChunk.embedding.cosine_distance(qv)
    rows = db.execute(
        select(DocChunk, Document.title, dist.label("dist"))
        .join(Document)
        .where(DocChunk.access_level <= ACCESS_LEVEL_BY_ROLE[role])
        .order_by(dist)
        .limit(k)
    ).all()
    return [Hit(c.document_id, title, c.content, c.access_level, round(1 - d, 4)) for c, title, d in rows]


def visible_documents(db: Session, role: Role) -> list[Document]:
    return list(
        db.scalars(
            select(Document).where(Document.access_level <= ACCESS_LEVEL_BY_ROLE[role]).order_by(Document.id)
        )
    )


def search_docs(db: Session, queries: list[str], role: Role, k: int = 6) -> list[Hit]:
    """문서 단위로 검색한다. 질의마다 BM25 순위를 매기고 RRF로 합쳐 상위 k개 문서를 통째로 돌려준다.

    접근 등급 필터는 색인을 만들기 전에 적용하므로, 볼 수 없는 문서는 점수 계산에도 들어가지 않는다.
    문서가 수천 개를 넘으면 매번 색인을 만드는 대신 DB 전문 검색으로 옮겨야 한다.
    """
    docs = visible_documents(db, role)
    if not docs or not queries:
        return []
    bm = BM25([f"{d.title}\n{d.content}" for d in docs])
    ranks = []
    for q in queries:
        s = bm.scores(q)
        ranks.append([int(i) for i in sorted(range(len(docs)), key=lambda i: -s[i]) if s[i] > 0])
    return [
        Hit(docs[i].id, docs[i].title, docs[i].content, docs[i].access_level, round(score, 4))
        for i, score in rrf(ranks)[:k]
    ]
