"""테스트는 개발 DB 대신 `hitl_test` DB를 쓰고, mock LLM으로 돌린다."""

import os

os.environ["DATABASE_URL"] = "postgresql+psycopg://hitl:hitl@localhost:5442/hitl_test"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["JWT_SECRET"] = "test-secret-" + "x" * 32

import psycopg  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seeded_test_db():
    with psycopg.connect("postgresql://hitl:hitl@localhost:5442/hitl", autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = 'hitl_test'").fetchone()
        if not exists:
            conn.execute("CREATE DATABASE hitl_test")
    from app.seed.seed import run

    run()
