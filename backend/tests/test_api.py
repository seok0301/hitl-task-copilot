"""mock 프로바이더로 로그인부터 경영진 브리핑까지 API 전체 사이클을 확인한다."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app

pytestmark = pytest.mark.db
client = TestClient(app)


def login(email: str) -> dict:
    r = client.post("/api/auth/login", json={"email": email, "password": "password"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_login_rejects_wrong_password():
    r = client.post("/api/auth/login", json={"email": "ceo@example.com", "password": "x"})
    assert r.status_code == 401


def test_employee_cannot_create_project():
    r = client.post(
        "/api/projects",
        headers=login("seoyeon@example.com"),
        json={"title": "x", "goal": "x", "deadline": "2030-01-01"},
    )
    assert r.status_code == 403


def test_full_cycle():
    mgr = login("dev.lead@example.com")
    deadline = (date.today() + timedelta(days=30)).isoformat()
    r = client.post(
        "/api/projects",
        headers=mgr,
        json={
            "title": "앱 푸시 재방문 캠페인",
            "goal": "휴면 고객에게 앱 푸시를 보내 재방문을 늘린다.",
            "deadline": deadline,
        },
    )
    assert r.status_code == 200, r.text
    pid = r.json()["id"]

    d = client.get(f"/api/projects/{pid}/draft", headers=mgr).json()
    assert d["status"] == "REVIEW" and d["draft"], d
    draft = d["draft"]
    tasks = [
        {**t, "draft_index": i, "assignee_id": a["assignee_id"]}
        for i, (t, a) in enumerate(zip(draft["tasks"], draft["assignments"], strict=True))
    ]
    r = client.post(f"/api/projects/{pid}/review", headers=mgr, json={"action": "approve", "tasks": tasks})
    assert r.status_code == 200 and r.json()["status"] == "ACTIVE", r.text

    detail = client.get(f"/api/projects/{pid}", headers=mgr).json()
    task = detail["tasks"][0]
    users = {u["id"]: u for u in client.get("/api/users", headers=mgr).json()}
    worker = login(users[task["assignee_id"]]["email"])

    assert any(t["id"] == task["id"] for t in client.get("/api/tasks/mine", headers=worker).json())
    r = client.post(
        f"/api/tasks/{task['id']}/submissions", headers=worker, json={"content": task["description"]}
    )
    assert r.status_code == 200 and r.json()["ai_review"]["coverage"] > 0, r.text
    assert client.post(f"/api/tasks/{task['id']}/accept", headers=mgr).json()["status"] == "DONE"

    exe = login("ceo@example.com")
    m = next(x for x in client.get("/api/dashboard/metrics", headers=exe).json() if x["project_id"] == pid)
    assert m["progress"] > 0
    brief = client.post("/api/dashboard/brief", headers=exe).json()["brief"]
    assert brief["headline"]
