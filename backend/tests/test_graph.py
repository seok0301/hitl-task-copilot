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
    draft = [
        {"title": "a", "assignee_id": 1},
        {"title": "b", "assignee_id": 2},
        {"title": "c", "assignee_id": 3},
    ]
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
        p = Project(
            title="앱 정기구독 결제",
            goal="모바일 앱에 정기구독 결제를 추가한다.",
            deadline=date.today() + timedelta(days=30),
            owner_id=3,
            graph_thread_id=str(uuid.uuid4()),
        )
        db.add(p)
        db.commit()
        pid, tid = p.id, p.graph_thread_id

    graph, cfg = get_graph(), thread_config(tid)
    graph.invoke(
        {
            "project_id": pid,
            "title": "앱 정기구독 결제",
            "goal": "모바일 앱에 정기구독 결제를 추가한다.",
            "days": 30,
            "use_rag": True,
        },
        cfg,
    )
    draft = pending_review(tid)
    assert draft and draft["tasks"] and len(draft["assignments"]) == len(draft["tasks"])
    first_titles = [t["title"] for t in draft["tasks"]]

    # 승인 전에는 업무가 하나도 만들어지지 않는다.
    with SessionLocal() as db:
        assert db.get(Project, pid).tasks == []

    # 서버 재시작: 그래프와 체크포인터를 새로 만들어도 승인 대기 상태가 남아 있다.
    get_graph.cache_clear()
    graph = get_graph()
    assert pending_review(tid)["tasks"] == draft["tasks"]

    # 재생성 요청 후 다시 승인 대기. mock 편집 명령은 비어 있으므로 이전 초안이 그대로 남아야 한다.
    graph.invoke(
        Command(resume={"action": "regenerate", "feedback": "QA를 꼭 넣어 주세요", "reviewer_id": 3}), cfg
    )
    draft = pending_review(tid)
    assert draft is not None
    assert [t["title"] for t in draft["tasks"]] == first_titles

    final = [
        {**t, "draft_index": i, "assignee_id": a["assignee_id"]}
        for i, (t, a) in enumerate(zip(draft["tasks"], draft["assignments"], strict=True))
    ]
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


def test_apply_edits_keeps_untouched_tasks():
    from app.agents.planner import apply_edits

    prev = [{"title": t} for t in ("기획", "디자인", "개발", "QA")]
    edits = [
        {"op": "remove", "index": 1},
        {"op": "update", "index": 2, "task": {"title": "백엔드 개발", "estimate_hours": 40}},
        {"op": "add", "task": {"title": "법무 검토"}},
        {"op": "remove", "index": 9},  # 없는 번호는 무시한다
        {"op": "update", "index": 0, "task": {}},  # 제목 없는 수정은 무시한다
    ]
    assert [t["title"] for t in apply_edits(prev, edits)] == ["기획", "백엔드 개발", "QA", "법무 검토"]


def test_tier_prefers_same_title_replacement():
    import numpy as np

    from app.agents.resource import Candidate, score_task

    v = lambda *x: np.array(x) / np.linalg.norm(x)
    cands = [
        Candidate(1, "적임자", "개발", "프론트엔드 개발자", ["React"], v(1, 0, 0), 40, load_hours=200),
        Candidate(2, "같은 직함", "개발", "프론트엔드 개발자", ["Vue"], v(0.6, 0.8, 0), 40),
        Candidate(3, "같은 팀 다른 직함", "개발", "백엔드 개발자", ["Python"], v(0.8, 0.6, 0), 40),
    ]
    ranked = score_task(
        v(1, 0, 0),
        cands,
        required_skills=["React"],
        weeks=3,
        task_hours=24,
        capacity_constraint=True,
        role_filter=True,
    )
    assert ranked[0]["name"] == "같은 직함"
    assert ranked[-1]["over_capacity"] is True


@pytest.mark.db
def test_capacity_constraint_skips_overloaded():
    from app.agents.resource import match_tasks

    task = {
        "title": "해지 사유 수집 API",
        "description": "사유를 저장하는 API",
        "required_skills": ["FastAPI"],
        "estimate_hours": 24,
    }
    with SessionLocal() as db:
        free = match_tasks(db, [task], days=21, with_reasons=False)[0]["assignee_id"]
        busy = match_tasks(db, [task], days=21, with_reasons=False, busy={free: 500})[0]
    assert busy["assignee_id"] != free
    assert busy["candidates"][0]["over_capacity"] is False
