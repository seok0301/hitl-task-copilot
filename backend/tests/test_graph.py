"""mock 프로바이더로 계획 → 추천 → 승인 대기 → 승인 → 커밋까지 한 사이클을 확인한다."""

import uuid
from datetime import date, timedelta

import pytest
from langgraph.types import Command

from app.agents.edits import edit_stats
from app.agents.graph import get_graph, pending_review, thread_config
from app.db import SessionLocal
from app.models import ApprovalLog, Project, ProjectStatus


def test_edit_stats():
    draft = [{"title": "a", "assignee_id": 1}, {"title": "b", "assignee_id": 2}, {"title": "c", "assignee_id": 3}]
    final = [
        {"draft_index": 0, "title": "a", "assignee_id": 1},
        {"draft_index": 1, "title": "b", "assignee_id": 5},
        {"draft_index": None, "title": "d", "assignee_id": 4},
    ]
    s = edit_stats(draft, final)
    assert (s["modified"], s["deleted"], s["added"], s["unchanged"]) == (1, 1, 1, 1)
    assert s["edit_rate"] == 0.75
    assert s["assignee_change_rate"] == 0.5


@pytest.mark.db
def test_plan_review_approve_cycle(monkeypatch):
    with SessionLocal() as db:
        p = Project(title="앱 정기구독 결제", goal="모바일 앱에 정기구독 결제를 추가한다.",
                    deadline=date.today() + timedelta(days=30), owner_id=3, graph_thread_id=str(uuid.uuid4()))
        db.add(p)
        db.commit()
        pid, tid = p.id, p.graph_thread_id

    graph, cfg = get_graph(), thread_config(tid)
    graph.invoke({"project_id": pid, "title": "앱 정기구독 결제", "goal": "모바일 앱에 정기구독 결제를 추가한다.",
                  "days": 30, "use_rag": True}, cfg)
    draft = pending_review(tid)
    assert draft and draft["tasks"] and len(draft["assignments"]) == len(draft["tasks"])

    # 재생성 요청 후 다시 승인 대기
    graph.invoke(Command(resume={"action": "regenerate", "feedback": "QA를 꼭 넣어 주세요", "reviewer_id": 3}), cfg)
    draft = pending_review(tid)
    assert draft is not None

    final = [{**t, "draft_index": i, "assignee_id": a["assignee_id"]}
             for i, (t, a) in enumerate(zip(draft["tasks"], draft["assignments"], strict=True))]
    final[0]["assignee_id"] = 16
    graph.invoke(Command(resume={"action": "approve", "tasks": final[:-1], "reviewer_id": 3}), cfg)
    assert pending_review(tid) is None

    with SessionLocal() as db:
        p = db.get(Project, pid)
        assert p.status == ProjectStatus.ACTIVE
        assert len(p.tasks) == len(final) - 1
        logs = db.query(ApprovalLog).filter_by(project_id=pid).order_by(ApprovalLog.id).all()
        assert [log.action for log in logs] == ["regenerate", "approve"]
        assert logs[-1].edit_stats["deleted"] == 1
