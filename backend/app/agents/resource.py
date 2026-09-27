"""② 리소스 에이전트: 하이브리드 점수로 담당자를 고르고, LLM은 추천 사유만 쓴다.

score = α·스킬 적합도 + β·과거 유사 업무 이력 − γ·업무량
- 스킬 적합도: 태스크와 팀원 스킬의 임베딩 유사도와 필요 스킬 태그 일치율의 평균
- 이력: 팀원이 과거에 맡은 태스크 중 가장 비슷한 것과의 임베딩 유사도
- 업무량: (진행 중인 업무 + 이번 계획에서 먼저 배정된 업무) / 프로젝트 기간 동안의 수용 시간
스킬 적합도와 이력 점수는 태스크마다 후보 전체에서 0~1로 정규화한다.
"""

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.llm.embeddings import embed_texts
from app.llm.provider import call_json
from app.models import PastTask, Role, Task, TaskStatus, User


@dataclass
class Candidate:
    user_id: int
    name: str
    team: str
    title: str
    skills: list[str]
    skill_vec: np.ndarray
    capacity: float
    load_hours: float = 0.0
    history: list[tuple[str, np.ndarray]] = field(default_factory=list)


def task_text(t: dict) -> str:
    skills = ", ".join(t.get("required_skills") or [])
    return f"{t['title']}. {t.get('description', '')} {skills}".strip()


def load_candidates(db: Session) -> list[Candidate]:
    users = db.scalars(select(User).where(User.role != Role.EXECUTIVE).order_by(User.id)).all()
    loads = dict(
        db.execute(
            select(Task.assignee_id, func.sum(Task.estimate_hours))
            .where(Task.status.in_([TaskStatus.TODO, TaskStatus.IN_PROGRESS]), Task.assignee_id.isnot(None))
            .group_by(Task.assignee_id)
        ).all()
    )
    cands = {
        u.id: Candidate(
            u.id,
            u.name,
            u.team,
            u.title,
            list(u.skills),
            np.asarray(u.skill_embedding),
            u.weekly_capacity_hours,
            float(loads.get(u.id, 0.0)),
        )
        for u in users
    }
    for pt in db.scalars(select(PastTask)):
        if pt.assignee_id in cands and pt.embedding is not None:
            cands[pt.assignee_id].history.append((pt.title, np.asarray(pt.embedding)))
    return list(cands.values())


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 1e-9 else np.zeros_like(x)


def score_task(
    task_vec: np.ndarray,
    cands: list[Candidate],
    extra_load: dict[int, float] | None = None,
    required_skills: list[str] | None = None,
    weeks: float = 1.0,
) -> list[dict]:
    s = get_settings()
    extra_load = extra_load or {}
    req = set(required_skills or [])
    emb = np.array([float(task_vec @ c.skill_vec) for c in cands])
    tag = np.array([len(req & set(c.skills)) / len(req) if req else 0.0 for c in cands])
    skill = (emb + tag) / 2 if req else emb
    hist, hist_title = [], []
    for c in cands:
        if c.history:
            sims = [float(task_vec @ v) for _, v in c.history]
            j = int(np.argmax(sims))
            hist.append(sims[j])
            hist_title.append(c.history[j][0])
        else:
            hist.append(0.0)
            hist_title.append(None)
    hist = np.array(hist)
    load = np.array(
        [min((c.load_hours + extra_load.get(c.user_id, 0)) / (c.capacity * weeks), 1.0) for c in cands]
    )
    skill_n, hist_n = _minmax(skill), _minmax(hist)
    total = s.match_alpha * skill_n + s.match_beta * hist_n - s.match_gamma * load
    ranked = []
    for i in np.argsort(-total):
        c = cands[i]
        ranked.append(
            {
                "user_id": c.user_id,
                "name": c.name,
                "team": c.team,
                "score": round(float(total[i]), 4),
                "skill": round(float(skill_n[i]), 4),
                "history": round(float(hist_n[i]), 4),
                "load": round(float(load[i]), 4),
                "similar_past_task": hist_title[i],
            }
        )
    return ranked


REASON_SYSTEM = """너는 담당자 추천 사유를 쓰는 에이전트다.
담당자는 이미 점수로 정해졌다. 담당자를 바꾸지 말고, 주어진 근거(스킬, 과거 유사 업무, 현재 업무량)만으로
태스크마다 한두 문장의 추천 사유를 한국어로 쓴다. 근거에 없는 사실을 지어내지 않는다.
출력 형식: {"reasons": [{"index": 0, "reason": "..."}]}"""


def _mock_reason(c: dict) -> str:
    parts = [f"{c['team']}팀 {c['name']}의 스킬이 태스크와 잘 맞습니다"]
    if c.get("similar_past_task"):
        parts.append(f"과거 '{c['similar_past_task']}' 업무를 수행했습니다")
    parts.append(f"업무량은 프로젝트 기간 수용량의 {round(c['load'] * 100)}%입니다")
    return ". ".join(parts) + "."


def match_tasks(
    db: Session,
    tasks: list[dict],
    days: int = 30,
    project_id: int | None = None,
    with_reasons: bool = True,
) -> list[dict]:
    """태스크마다 상위 3명 후보와 추천 담당자를 돌려준다.

    같은 계획 안에서 먼저 배정된 태스크의 시간을 업무량에 더해, 한 사람에게 몰리지 않게 한다.
    """
    cands = load_candidates(db)
    cand_by_id = {c.user_id: c for c in cands}
    vecs = [np.asarray(v) for v in embed_texts([task_text(t) for t in tasks])]
    weeks = max(days / 7, 1.0)
    extra: dict[int, float] = {}
    results = []
    for t, v in zip(tasks, vecs, strict=True):
        ranked = score_task(v, cands, extra, t.get("required_skills"), weeks)
        top = ranked[0]
        extra[top["user_id"]] = extra.get(top["user_id"], 0) + float(t.get("estimate_hours") or 8)
        results.append({"assignee_id": top["user_id"], "candidates": ranked[:3], "reason": ""})

    if with_reasons and results:
        lines = []
        for i, (t, r) in enumerate(zip(tasks, results, strict=True)):
            c = r["candidates"][0]
            u = cand_by_id[c["user_id"]]
            lines.append(
                f"[{i}] 태스크: {t['title']} ({t.get('description', '')})\n"
                f"    담당자: {u.name} ({u.team}팀 {u.title}), 스킬: {', '.join(u.skills)}\n"
                f"    과거 유사 업무: {c['similar_past_task'] or '없음'}, "
                f"업무량: 프로젝트 기간 수용량의 {round(c['load'] * 100)}%"
            )
        out = call_json(
            "resource",
            REASON_SYSTEM,
            "\n".join(lines),
            mock=lambda: {
                "reasons": [
                    {"index": i, "reason": _mock_reason(r["candidates"][0])} for i, r in enumerate(results)
                ]
            },
            project_id=project_id,
        )
        for item in out.get("reasons", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(results):
                results[idx]["reason"] = str(item.get("reason", ""))
    return results


LLM_ONLY_SYSTEM = """너는 IT 스타트업의 담당자 배정 에이전트다.
팀원 목록(스킬, 현재 업무량)을 보고 각 태스크에 가장 적합한 담당자를 3순위까지 고른다.
출력 형식: {"assignments": [{"index": 0, "ranking": [팀원 id, 팀원 id, 팀원 id], "reason": "..."}]}"""


def match_tasks_llm_only(db: Session, tasks: list[dict]) -> list[list[int]]:
    """비교 실험용: 점수 없이 LLM이 담당자를 직접 고른다."""
    cands = load_candidates(db)
    roster = "\n".join(
        f"id={c.user_id} {c.name} ({c.team}팀 {c.title}) 스킬: {', '.join(c.skills)} "
        f"/ 현재 업무 {c.load_hours:.0f}시간"
        for c in cands
    )
    body = "\n".join(f"[{i}] {t['title']}: {t.get('description', '')}" for i, t in enumerate(tasks))

    def mock():
        # mock에서는 스킬 유사도만으로 고른다.
        vecs = [np.asarray(v) for v in embed_texts([task_text(t) for t in tasks])]
        return {
            "assignments": [
                {
                    "index": i,
                    "ranking": [
                        cands[j].user_id for j in np.argsort([-(v @ c.skill_vec) for c in cands])[:3]
                    ],
                }
                for i, v in enumerate(vecs)
            ]
        }

    out = call_json("resource_llm_only", LLM_ONLY_SYSTEM, f"팀원:\n{roster}\n\n태스크:\n{body}", mock=mock)
    rankings: list[list[int]] = [[] for _ in tasks]
    for a in out.get("assignments", []):
        idx = a.get("index")
        if isinstance(idx, int) and 0 <= idx < len(tasks):
            rankings[idx] = [int(x) for x in a.get("ranking", []) if str(x).isdigit()]
    return rankings
