"""계획 그래프: planner → resource_matcher → human_review(interrupt) → commit.

human_review에서 그래프가 멈추고, 관리자의 결정이 `Command(resume=...)`로 들어오면 이어서 실행된다.
그래프 상태는 Postgres 체크포인터에 저장되므로 서버를 재시작해도 승인 대기 상태가 유지된다.
"""

import time
from datetime import date
from functools import lru_cache
from typing import TypedDict

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.agents.edits import edit_stats
from app.agents.planner import plan_tasks
from app.agents.resource import match_tasks
from app.config import get_settings
from app.db import SessionLocal
from app.models import ApprovalLog, Project, ProjectStatus, Role, Task


class PlanState(TypedDict, total=False):
    project_id: int
    title: str
    goal: str
    days: int
    use_rag: bool
    feedback: str | None
    round: int
    tasks: list[dict]
    assignments: list[dict]
    sources: list[dict]
    draft_ready_at: float
    decision: dict


def planner_node(state: PlanState) -> PlanState:
    with SessionLocal() as db:
        tasks, hits = plan_tasks(
            db,
            state["title"],
            state["goal"],
            state["days"],
            Role.MANAGER,
            use_rag=state.get("use_rag", True),
            feedback=state.get("feedback"),
            project_id=state["project_id"],
        )
    sources = [{"title": h.title, "score": h.score, "content": h.content} for h in hits]
    return {"tasks": tasks, "sources": sources, "round": state.get("round", 0) + 1}


def matcher_node(state: PlanState) -> PlanState:
    with SessionLocal() as db:
        assignments = match_tasks(db, state["tasks"], days=state["days"], project_id=state["project_id"])
        project = db.get(Project, state["project_id"])
        project.status = ProjectStatus.REVIEW
        db.commit()
    return {"assignments": assignments, "draft_ready_at": time.time()}


def review_node(state: PlanState) -> Command:
    decision = interrupt(
        {"tasks": state["tasks"], "assignments": state["assignments"], "sources": state.get("sources", [])}
    )
    if decision.get("action") == "regenerate":
        _log(state, decision, final=None)
        with SessionLocal() as db:
            db.get(Project, state["project_id"]).status = ProjectStatus.PLANNING
            db.commit()
        return Command(goto="planner", update={"feedback": decision.get("feedback") or None})
    return Command(goto="commit", update={"decision": decision})


def commit_node(state: PlanState) -> PlanState:
    decision = state["decision"]
    final = decision["tasks"]
    _log(state, decision, final=final)
    with SessionLocal() as db:
        project = db.get(Project, state["project_id"])
        for t in final:
            di = t.get("draft_index")
            db.add(
                Task(
                    project_id=project.id,
                    title=t["title"],
                    description=t.get("description", ""),
                    required_skills=t.get("required_skills", []),
                    estimate_hours=float(t.get("estimate_hours") or 8),
                    deadline=date.fromisoformat(t["deadline"]) if t.get("deadline") else project.deadline,
                    assignee_id=t.get("assignee_id"),
                    ai_draft=_draft_item(state, di) if di is not None else None,
                )
            )
        project.status = ProjectStatus.ACTIVE
        db.commit()
    return {}


def _draft(state: PlanState) -> list[dict]:
    return [_draft_item(state, i) for i in range(len(state["tasks"]))]


def _draft_item(state: PlanState, i: int) -> dict:
    return {**state["tasks"][i], **state["assignments"][i]}


def _log(state: PlanState, decision: dict, final: list[dict] | None) -> None:
    draft = _draft(state)
    with SessionLocal() as db:
        db.add(
            ApprovalLog(
                project_id=state["project_id"],
                reviewer_id=decision["reviewer_id"],
                action=decision.get("action", "approve"),
                draft={"round": state.get("round", 1), "tasks": draft, "feedback": decision.get("feedback")},
                final={"tasks": final} if final is not None else None,
                edit_stats=edit_stats(draft, final) if final is not None else None,
                review_seconds=round(time.time() - state["draft_ready_at"], 1),
            )
        )
        db.commit()


def build_graph(checkpointer=None):
    g = StateGraph(PlanState)
    g.add_node("planner", planner_node)
    g.add_node("resource_matcher", matcher_node)
    g.add_node("human_review", review_node, destinations=("planner", "commit"))
    g.add_node("commit", commit_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "resource_matcher")
    g.add_edge("resource_matcher", "human_review")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache
def get_graph():
    url = get_settings().database_url.replace("postgresql+psycopg://", "postgresql://")
    pool = ConnectionPool(
        url,
        max_size=5,
        open=True,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    saver = PostgresSaver(pool)
    saver.setup()
    return build_graph(saver)


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def pending_review(thread_id: str) -> dict | None:
    """승인 대기 중이면 interrupt에 담긴 초안을 돌려준다."""
    snap = get_graph().get_state(thread_config(thread_id))
    for task in snap.tasks:
        for it in task.interrupts:
            return it.value
    return None
