"""접근 등급으로 거른 벡터 검색."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.embeddings import embed_text
from app.models import ACCESS_LEVEL_BY_ROLE, DocChunk, Document, Role


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
