from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import get_graph
from app.routers import auth, dashboard, projects, tasks, users


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 체크포인터 테이블은 요청을 받기 전에 만든다. 첫 요청 중에 만들면 CREATE INDEX CONCURRENTLY가
    # 요청이 열어 둔 트랜잭션을 기다리고, 그 요청은 계획 작업을 기다려 서로 멈춘다.
    get_graph()
    yield


app = FastAPI(title="HITL Task Copilot", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)
for r in (auth.router, users.router, projects.router, tasks.router, dashboard.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True}
