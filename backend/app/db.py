from collections.abc import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

# 평가 스크립트가 LLM 호출을 병렬로 보내므로 연결 풀을 넉넉히 둔다.
engine = create_engine(get_settings().database_url, pool_pre_ping=True, pool_size=20, max_overflow=30)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db(drop: bool = False) -> None:
    from app import models  # noqa: F401  테이블 등록

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    if drop:
        Base.metadata.drop_all(engine)
        with engine.begin() as conn:
            for t in ("checkpoint_writes", "checkpoint_blobs", "checkpoints", "checkpoint_migrations"):
                conn.execute(text(f"DROP TABLE IF EXISTS {t}"))
    Base.metadata.create_all(engine)


def get_db() -> Iterator[Session]:
    with SessionLocal() as db:
        yield db
