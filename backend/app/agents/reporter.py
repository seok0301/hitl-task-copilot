"""③ 취합·보고 에이전트.

- review_submission: 실무자 산출물을 태스크 명세와 대조한다.
- 보고 그래프: collect → progress_metrics → executive_brief
"""

import json
from datetime import date
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.db import SessionLocal
from app.llm.provider import call_json
from app.models import Project, ProjectStatus, Task, TaskStatus

REVIEW_SYSTEM = """너는 실무자가 제출한 산출물을 태스크 명세와 대조하는 검토 에이전트다.
명세의 요구 사항 중 산출물이 충족한 것과 빠진 것을 나누고, 충족도를 0~100으로 매긴다.
산출물에 없는 내용을 있다고 판단하지 않는다.
출력 형식: {"coverage": 80, "met": ["..."], "missing": ["..."], "summary": "한두 문장 요약"}"""


def _mock_review(task: Task, content: str) -> dict:
    words = {w for w in (task.title + " " + task.description).split() if len(w) >= 2}
    met = sorted(w for w in words if w in content)
    cov = round(100 * len(met) / len(words)) if words else 100
    return {
        "coverage": cov,
        "met": met[:5],
        "missing": sorted(words - set(met))[:5],
        "summary": f"명세 핵심어 {len(words)}개 중 {len(met)}개가 산출물에 나타납니다.",
    }


def review_submission(task: Task, content: str) -> dict:
    user = f"태스크: {task.title}\n명세: {task.description}\n\n제출된 산출물:\n{content}"
    out = call_json(
        "reporter_review",
        REVIEW_SYSTEM,
        user,
        mock=lambda: _mock_review(task, content),
        project_id=task.project_id,
    )
    try:
        out["coverage"] = max(0, min(100, int(out.get("coverage", 0))))
    except (TypeError, ValueError):
        out["coverage"] = 0
    return out


def project_metrics(p: Project, today: date | None = None) -> dict:
    """진척도는 완료 태스크의 예상 시간 합 / 전체 예상 시간 합이다. 제출됐지만 승인 전인 태스크는 절반으로 센다."""
    today = today or date.today()
    total = sum(t.estimate_hours for t in p.tasks) or 1
    weight = {TaskStatus.DONE: 1.0, TaskStatus.SUBMITTED: 0.5}
    done = sum(t.estimate_hours * weight.get(t.status, 0) for t in p.tasks)
    overdue = [t.title for t in p.tasks if t.deadline and t.deadline < today and t.status != TaskStatus.DONE]
    reviews = [s.ai_review["coverage"] for t in p.tasks for s in t.submissions[-1:] if s.ai_review]
    per_person: dict[str, float] = {}
    for t in p.tasks:
        if t.assignee and t.status != TaskStatus.DONE:
            per_person[t.assignee.name] = per_person.get(t.assignee.name, 0) + t.estimate_hours
    return {
        "project_id": p.id,
        "title": p.title,
        "deadline": p.deadline.isoformat(),
        "days_left": (p.deadline - today).days,
        "task_count": len(p.tasks),
        "status_counts": {s.value: sum(1 for t in p.tasks if t.status == s) for s in TaskStatus},
        "progress": round(100 * done / total, 1),
        "overdue_tasks": overdue,
        "avg_review_coverage": round(sum(reviews) / len(reviews), 1) if reviews else None,
        "remaining_hours_by_person": per_person,
    }


class ReportState(TypedDict, total=False):
    metrics: list[dict]
    brief: dict


BRIEF_SYSTEM = """너는 경영진에게 프로젝트 현황을 보고하는 에이전트다.
주어진 지표만 근거로 짧고 명확하게 보고한다. 수치를 지어내지 않는다.
출력 형식: {"headline": "한 문장 요약", "highlights": ["..."], "risks": ["..."], "next_actions": ["..."]}"""


def collect_and_measure(state: ReportState) -> ReportState:
    with SessionLocal() as db:
        return {"metrics": all_metrics(db)}


def all_metrics(db: Session) -> list[dict]:
    projects = db.scalars(
        select(Project)
        .where(Project.status == ProjectStatus.ACTIVE)
        .options(
            selectinload(Project.tasks).selectinload(Task.submissions),
            selectinload(Project.tasks).selectinload(Task.assignee),
        )
        .order_by(Project.id)
    ).all()
    return [project_metrics(p) for p in projects]


def _mock_brief(metrics: list[dict]) -> dict:
    risks = [f"{m['title']}: 기한 초과 {len(m['overdue_tasks'])}건" for m in metrics if m["overdue_tasks"]]
    return {
        "headline": f"진행 중인 프로젝트 {len(metrics)}개, 평균 진척도 "
        f"{round(sum(m['progress'] for m in metrics) / max(len(metrics), 1), 1)}%입니다.",
        "highlights": [f"{m['title']}: 진척도 {m['progress']}%" for m in metrics],
        "risks": risks,
        "next_actions": ["기한 초과 태스크의 담당자와 일정 재조정"] if risks else [],
    }


def executive_brief(state: ReportState) -> ReportState:
    metrics = state["metrics"]
    user = "프로젝트 지표(JSON):\n" + json.dumps(metrics, ensure_ascii=False, indent=1)
    return {"brief": call_json("reporter_brief", BRIEF_SYSTEM, user, mock=lambda: _mock_brief(metrics))}


def build_report_graph():
    g = StateGraph(ReportState)
    g.add_node("collect_and_measure", collect_and_measure)
    g.add_node("executive_brief", executive_brief)
    g.add_edge(START, "collect_and_measure")
    g.add_edge("collect_and_measure", "executive_brief")
    g.add_edge("executive_brief", END)
    return g.compile()


def run_report() -> dict:
    return build_report_graph().invoke({})
