"""① 계획 에이전트: 과거 유사 프로젝트를 검색해 하위 태스크 초안을 만든다."""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.llm.provider import call_json
from app.models import Role, User
from app.rag.retriever import Hit, search

SYSTEM = """너는 IT 스타트업의 프로젝트 계획 담당 에이전트다.
관리자가 준 목표를 실행 가능한 하위 태스크로 나눈다.
- 태스크는 4~8개로 만들고, 한 사람이 맡을 수 있는 크기로 쪼갠다.
- 기획, 디자인, 개발, QA, 마케팅 중 목표에 필요한 역할만 포함한다.
- 참고 자료가 있으면 과거 프로젝트의 태스크 구성과 교훈을 반영한다.
- required_skills는 반드시 아래 "사내 스킬 목록"에 있는 이름 그대로 1~3개 고른다.
- estimate_hours는 한 사람 기준 작업 시간(4~120)이다.
출력 형식:
{"tasks": [{"title": "...", "description": "...", "required_skills": ["..."], "estimate_hours": 16}]}"""

GENERIC_TASKS = [
    ("요구사항 정의", "목표를 달성하는 데 필요한 기능과 범위를 정의한다.", ["서비스 기획"], 16),
    ("화면 디자인", "필요한 화면을 디자인한다.", ["UI 디자인"], 24),
    ("백엔드 개발", "필요한 API와 데이터 처리를 개발한다.", ["REST API"], 40),
    ("프론트엔드 개발", "사용자 화면을 개발한다.", ["React"], 32),
    ("QA", "기능을 테스트하고 결함을 수정한다.", ["QA 시나리오"], 16),
]

RETRO_LINE = re.compile(r"^- (.+?) \(담당 .+?, (\d+)시간\): (.+)$")


def _mock_plan(goal: str, hits: list[Hit]) -> dict:
    """mock 프로바이더용 계획. 가장 관련 높은 회고 문서의 태스크 구성을 그대로 빌려 온다."""
    for h in hits:
        if not h.title.startswith("[회고]"):
            continue
        tasks = [
            {"title": m[1], "description": m[3], "required_skills": [], "estimate_hours": float(m[2])}
            for m in (RETRO_LINE.match(line) for line in h.content.splitlines())
            if m
        ]
        if tasks:
            return {"tasks": tasks}
    return {
        "tasks": [
            {"title": t, "description": d, "required_skills": s, "estimate_hours": h}
            for t, d, s, h in GENERIC_TASKS
        ]
    }


def skill_vocabulary(db: Session) -> list[str]:
    return sorted({sk for skills in db.scalars(select(User.skills)) for sk in skills})


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{i + 1}] {h.title}\n{h.content}" for i, h in enumerate(hits))


def plan_tasks(
    db: Session,
    title: str,
    goal: str,
    days: int,
    role: Role = Role.MANAGER,
    use_rag: bool = True,
    feedback: str | None = None,
    project_id: int | None = None,
) -> tuple[list[dict], list[Hit]]:
    hits = search(db, f"{title}. {goal}", role, k=6) if use_rag else []
    user = f"프로젝트: {title}\n목표: {goal}\n기간: {days}일"
    user += f"\n\n사내 스킬 목록: {', '.join(skill_vocabulary(db))}"
    if hits:
        user += f"\n\n참고 자료(사내 문서 검색 결과):\n{format_hits(hits)}"
    if feedback:
        user += f"\n\n관리자 요청 사항(이전 초안에 대한 피드백): {feedback}"

    out = call_json("planner", SYSTEM, user, mock=lambda: _mock_plan(goal, hits), project_id=project_id)
    tasks = []
    for t in out.get("tasks", []):
        if not t.get("title"):
            continue
        tasks.append(
            {
                "title": str(t["title"]).strip(),
                "description": str(t.get("description", "")).strip(),
                "required_skills": [str(s) for s in t.get("required_skills", [])],
                "estimate_hours": float(t.get("estimate_hours") or 8),
            }
        )
    return tasks, hits
