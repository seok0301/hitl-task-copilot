import pytest

from app.db import SessionLocal
from app.models import Role
from app.rag.ingest import chunk_text
from app.rag.retriever import search


def test_chunk_text_respects_limit():
    text = "\n".join(f"{i}번째 줄입니다." for i in range(100))
    chunks = chunk_text(text, max_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


@pytest.mark.db
def test_search_filters_by_access_level():
    with SessionLocal() as db:
        emp = search(db, "인사 평가 보상 조정", Role.EMPLOYEE, k=20)
        exe = search(db, "인사 평가 보상 조정", Role.EXECUTIVE, k=20)
    assert all(h.access_level <= 1 for h in emp)
    assert any(h.access_level == 3 for h in exe)


@pytest.mark.db
def test_search_finds_related_retrospective():
    with SessionLocal() as db:
        hits = search(db, "앱 정기구독 결제", Role.MANAGER, k=3)
    assert any("정기구독" in h.title for h in hits)
