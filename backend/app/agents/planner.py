"""① 계획 에이전트: 과거 유사 프로젝트를 검색해 하위 태스크 초안을 만든다."""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.llm.provider import call_json
from app.models import Role, User
from app.rag.retriever import Hit, search, search_docs

SYSTEM_BASE = """너는 IT 스타트업의 프로젝트 계획 담당 에이전트다.
관리자가 준 목표를 실행 가능한 하위 태스크로 나눈다.
- 태스크는 4~8개로 만들고, 한 사람이 맡을 수 있는 크기로 쪼갠다.
- 기획, 디자인, 개발, QA, 마케팅 중 목표에 필요한 역할만 포함한다.
- 참고 자료가 있으면 과거 프로젝트의 태스크 구성과 교훈을 반영한다.
- required_skills는 반드시 아래 "사내 스킬 목록"에 있는 이름 그대로 1~3개 고른다.
- estimate_hours는 한 사람 기준 작업 시간(4~120)이다.
- 이전 초안과 관리자 요청 사항이 함께 주어지면, 요청한 부분만 고치고 나머지 태스크는 그대로 둔다.
  관리자가 부른 이름이 초안의 태스크 제목과 달라도, 요청이 가리키는 태스크를 찾아 요청대로 추가·삭제·수정한다.
  관리자 요청은 위의 태스크 개수 규칙보다 우선한다.
출력 형식:
{"tasks": [{"title": "...", "description": "...", "required_skills": ["..."], "estimate_hours": 16}]}"""

# 교훈 반영 강화판: 참고 자료의 교훈·규정을 먼저 골라 태스크로 옮기고, 태스크마다 근거를 남긴다.
# 근거(basis)는 관리자가 초안을 검토할 때 왜 이 태스크가 들어갔는지 확인하는 데 쓴다.
SYSTEM_LESSONS = """너는 IT 스타트업의 프로젝트 계획 담당 에이전트다.
관리자가 준 목표를 실행 가능한 하위 태스크로 나눈다.
- 태스크는 4~8개로 만들고, 한 사람이 맡을 수 있는 크기로 쪼갠다.
- 기획, 디자인, 개발, QA, 마케팅 중 목표에 필요한 역할만 포함한다.
- 참고 자료가 있으면, 먼저 참고 자료에 적힌 교훈·재발 방지책·사내 규정 중 이번 프로젝트에 해당하는 것을 모두 골라
  lessons에 적는다. 이번 프로젝트와 관계없는 자료는 무시한다.
- lessons의 각 항목은 별도 태스크로 만들거나, 관련 태스크의 설명에 구체적으로 적어 계획에 반드시 반영한다.
- 태스크마다 basis에 근거를 적는다. 참고 자료에서 온 태스크는 "[자료 번호] 교훈 요약", 일반적인 작업은 빈 문자열로 둔다.
- required_skills는 반드시 아래 "사내 스킬 목록"에 있는 이름 그대로 1~3개 고른다.
- estimate_hours는 한 사람 기준 작업 시간(4~120)이다.
- 이전 초안과 관리자 요청 사항이 함께 주어지면, 요청한 부분만 고치고 나머지 태스크는 그대로 둔다.
  관리자가 부른 이름이 초안의 태스크 제목과 달라도, 요청이 가리키는 태스크를 찾아 요청대로 추가·삭제·수정한다.
  관리자 요청은 위의 태스크 개수 규칙보다 우선한다.
출력 형식:
{"lessons": [{"source": 1, "lesson": "..."}],
 "tasks": [{"title": "...", "description": "...", "required_skills": ["..."], "estimate_hours": 16, "basis": "..."}]}"""

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


def retrieve(
    db: Session,
    title: str,
    goal: str,
    days: int,
    role: Role,
    mode: str,
    draft_tasks: list[dict] | None = None,
    project_id: int | None = None,
) -> list[Hit]:
    """검색 방식(mode)에 따라 참고 문서를 찾는다.

    - chunk: 프로젝트 제목과 목표로 청크 단위 벡터 검색
    - lexical: 프로젝트 제목과 목표로 문서 단위 BM25 검색
    - draft: RAG 없이 먼저 만든 초안의 태스크 문장들을 질의로 더해 문서 단위 BM25 검색
      (재생성할 때는 이전 초안이 있으므로 초안을 다시 만들지 않는다)
    """
    query = f"{title}. {goal}"
    if mode == "chunk":
        return search(db, query, role, k=6)
    if mode == "lexical":
        return search_docs(db, [query], role, k=6)
    if mode == "draft":
        if draft_tasks is None:
            draft_tasks = _generate(db, title, goal, days, [], None, None, "planner_draft", project_id)
        return search_docs(
            db, [query, *(f"{t['title']}. {t['description']}" for t in draft_tasks)], role, k=6
        )
    raise ValueError(f"알 수 없는 검색 방식: {mode}")


def _generate(
    db: Session,
    title: str,
    goal: str,
    days: int,
    hits: list[Hit],
    feedback: str | None,
    previous: list[dict] | None,
    agent: str,
    project_id: int | None,
) -> list[dict]:
    user = f"프로젝트: {title}\n목표: {goal}\n기간: {days}일"
    user += f"\n\n사내 스킬 목록: {', '.join(skill_vocabulary(db))}"
    if hits:
        user += f"\n\n참고 자료(사내 문서 검색 결과):\n{format_hits(hits)}"
    if previous:
        user += "\n\n이전 초안:\n" + "\n".join(
            f"- {t['title']}: {t.get('description', '')} (필요 스킬: {', '.join(t.get('required_skills') or [])}, "
            f"{t.get('estimate_hours', 8):.0f}시간)"
            for t in previous
        )
    if feedback:
        user += f"\n\n관리자 요청 사항(이전 초안에 대한 피드백): {feedback}"

    system = SYSTEM_LESSONS if get_settings().plan_extract_lessons else SYSTEM_BASE
    out = call_json(agent, system, user, mock=lambda: _mock_plan(goal, hits), project_id=project_id)
    return [_clean(t) for t in out.get("tasks", []) if isinstance(t, dict) and t.get("title")]


REVISE_SYSTEM = """너는 IT 스타트업의 프로젝트 계획 담당 에이전트다.
관리자가 이전 초안을 보고 수정을 요청했다. 초안을 새로 쓰지 말고, 요청을 반영하는 편집 명령만 낸다.
- 관리자가 부른 이름이 초안의 태스크 제목과 달라도, 요청이 가리키는 태스크를 찾아 그 번호로 편집한다.
- 요청과 관계없는 태스크에는 명령을 내지 않는다. 명령이 없는 태스크는 그대로 남는다.
- op는 remove(번호의 태스크 삭제), update(번호의 태스크를 task 내용으로 바꿈), add(task를 새로 추가) 중 하나다.
- 태스크 개수를 줄이라는 요청이면 관련 태스크를 update로 합치고 남는 것을 remove한다.
- 새 태스크와 바뀐 태스크의 required_skills는 "사내 스킬 목록"에 있는 이름 그대로 1~3개 고른다.
출력 형식:
{"edits": [{"op": "remove", "index": 3},
           {"op": "update", "index": 1, "task": {"title": "...", "description": "...", "required_skills": ["..."], "estimate_hours": 16, "basis": "..."}},
           {"op": "add", "task": {"title": "...", "description": "...", "required_skills": ["..."], "estimate_hours": 16, "basis": "..."}}]}"""


def _clean(t: dict) -> dict:
    return {
        "title": str(t["title"]).strip(),
        "description": str(t.get("description", "")).strip(),
        "required_skills": [str(s) for s in t.get("required_skills", [])],
        "estimate_hours": float(t.get("estimate_hours") or 8),
        "basis": str(t.get("basis") or "").strip(),
    }


def apply_edits(previous: list[dict], edits: list[dict]) -> list[dict]:
    """편집 명령을 코드로 적용한다. 명령이 없는 태스크는 그대로 남고, 잘못된 번호의 명령은 무시한다."""
    tasks: list[dict | None] = [dict(t) for t in previous]
    added = []
    for e in edits:
        op, idx, t = e.get("op"), e.get("index"), e.get("task")
        valid = isinstance(idx, int) and 0 <= idx < len(previous)
        if op == "remove" and valid:
            tasks[idx] = None
        elif op == "update" and valid and isinstance(t, dict) and t.get("title"):
            tasks[idx] = _clean(t)
        elif op == "add" and isinstance(t, dict) and t.get("title"):
            added.append(_clean(t))
    return [t for t in tasks if t is not None] + added


def revise_tasks(
    db: Session,
    title: str,
    goal: str,
    days: int,
    hits: list[Hit],
    feedback: str,
    previous: list[dict],
    project_id: int | None = None,
) -> list[dict]:
    user = f"프로젝트: {title}\n목표: {goal}\n기간: {days}일"
    user += f"\n\n사내 스킬 목록: {', '.join(skill_vocabulary(db))}"
    if hits:
        user += f"\n\n참고 자료(사내 문서 검색 결과):\n{format_hits(hits)}"
    user += "\n\n이전 초안:\n" + "\n".join(
        f"[{i}] {t['title']}: {t.get('description', '')} (필요 스킬: {', '.join(t.get('required_skills') or [])}, "
        f"{t.get('estimate_hours', 8):.0f}시간)"
        for i, t in enumerate(previous)
    )
    user += f"\n\n관리자 요청 사항: {feedback}"
    out = call_json("planner_revise", REVISE_SYSTEM, user, mock=lambda: {"edits": []}, project_id=project_id)
    edits = out.get("edits", []) if isinstance(out, dict) else []
    return apply_edits(previous, [e for e in edits if isinstance(e, dict)])


def plan_tasks(
    db: Session,
    title: str,
    goal: str,
    days: int,
    role: Role = Role.MANAGER,
    use_rag: bool = True,
    feedback: str | None = None,
    project_id: int | None = None,
    previous: list[dict] | None = None,
    hits: list[Hit] | None = None,
    rag_mode: str | None = None,
    revision: str | None = None,
) -> tuple[list[dict], list[Hit]]:
    """하위 태스크 초안을 만든다.

    previous: 재생성할 때 관리자가 본 이전 초안. feedback과 함께 주면 revision 방식으로 고친다.
      - edit: LLM은 편집 명령만 내고 코드가 적용한다(기본값). 언급하지 않은 태스크는 반드시 남는다.
      - rewrite: 이전 초안과 피드백을 주고 LLM이 목록 전체를 다시 쓴다.
      - regenerate: 이전 초안 없이 피드백만 주고 다시 계획한다.
    hits: 검색을 건너뛰고 이 문서들을 참고 자료로 쓴다(평가의 대조군과 상한 조건용).
    """
    revision = revision or get_settings().plan_revision
    if previous and feedback and revision == "edit":
        if hits is None:
            hits = (
                retrieve(
                    db, title, goal, days, role, rag_mode or get_settings().rag_mode, previous, project_id
                )
                if use_rag
                else []
            )
        return revise_tasks(db, title, goal, days, hits, feedback, previous, project_id), hits
    if revision == "regenerate":
        previous = None
    if hits is None:
        mode = rag_mode or get_settings().rag_mode
        hits = retrieve(db, title, goal, days, role, mode, previous, project_id) if use_rag else []
    tasks = _generate(db, title, goal, days, hits, feedback, previous, "planner", project_id)
    return tasks, hits
