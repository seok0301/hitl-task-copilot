"""문서를 청크로 나누고 임베딩해서 doc_chunks에 적재한다."""

from sqlalchemy.orm import Session

from app.llm.embeddings import embed_texts
from app.models import DocChunk, Document


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """줄 단위로 모아 max_chars를 넘지 않는 청크를 만든다."""
    chunks: list[str] = []
    buf = ""
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        if buf and len(buf) + len(line) + 1 > max_chars:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n{line}" if buf else line
    if buf:
        chunks.append(buf)
    return chunks


def ingest_document(db: Session, doc: Document) -> int:
    doc.chunks.clear()
    pieces = chunk_text(doc.content)
    # 청크마다 문서 제목을 앞에 붙여 검색할 때 문맥을 잃지 않게 한다.
    vectors = embed_texts([f"{doc.title}\n{p}" for p in pieces])
    for i, (piece, vec) in enumerate(zip(pieces, vectors, strict=True)):
        doc.chunks.append(DocChunk(seq=i, content=piece, access_level=doc.access_level, embedding=vec))
    db.flush()
    return len(pieces)
