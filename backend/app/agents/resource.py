"""② 리소스 에이전트: LLM이 담당자 순위를 매기고, 규칙이 수용 시간을 지키게 한다.

기본 방식(llm_rank)은 LLM이 팀원 목록을 보고 매긴 순위에서 수용 시간을 넘는 사람을 규칙으로 건너뛴다.
LLM에 "수용 시간을 넘기지 말라"고 지시만 하면 지켜지지 않았기 때문이다(eval/PROTOCOL.md H1).
아래 점수 방식은 초기 설계로, 지금은 후보별 근거(스킬·이력·업무량) 표시와 LLM이 실패했을 때의 예비 경로로 쓴다.

score = α·스킬 적합도 + β·과거 유사 업무 이력 − γ·업무량
- 스킬 적합도: 태스크와 팀원 스킬의 임베딩 유사도와 필요 스킬 태그 일치율의 평균
- 이력: 팀원이 과거에 맡은 태스크 중 가장 비슷한 것과의 임베딩 유사도
- 업무량: (진행 중인 업무 + 이번 계획에서 먼저 배정된 업무) / 프로젝트 기간 동안의 수용 시간
스킬 적합도와 이력 점수는 태스크마다 후보 전체에서 0~1로 정규화한다.
수용 시간 제약: 이 태스크를 맡으면 기간 수용 시간을 넘는 후보는 점수와 관계없이 순위를 뒤로 미룬다.
(모든 후보가 넘으면 점수 순서를 그대로 쓴다.) 넘지 않는 후보는 다시 점수 1위와 같은 직함, 필요 스킬 보유자,
점수 1위와 같은 팀, 나머지 순서의 계층으로 나눠, 적임자가 빠졌을 때 직무가 다른 사람이 대체자로 올라오지 않게 한다.
"""

from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.llm.embeddings import embed_texts
from app.llm.provider import LLMError, call_json
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


def load_candidates(db: Session, busy: dict[int, float] | None = None) -> list[Candidate]:
    """후보 목록. `busy`는 평가에서 특정 인원에게 이미 잡힌 업무 시간을 가정할 때 쓴다."""
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
            float(loads.get(u.id, 0.0)) + (busy or {}).get(u.id, 0.0),
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
    task_hours: float = 0.0,
    capacity_constraint: bool | None = None,
    role_filter: bool | None = None,
) -> list[dict]:
    s = get_settings()
    if capacity_constraint is None:
        capacity_constraint = s.match_capacity_constraint
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
    over = np.array(
        [c.load_hours + extra_load.get(c.user_id, 0) + task_hours > c.capacity * weeks for c in cands]
    )
    order = [int(i) for i in np.argsort(-total)]
    tier = [0] * len(cands)
    if capacity_constraint and not over.all():
        if role_filter is None:
            role_filter = s.match_role_filter
        # 후보 계층: 과부하 적임자를 뺄 때, 직무가 맞는 사람부터 대체자로 찾는다.
        # 점수 1위(제약을 적용하기 전)의 직함을 이 태스크에 필요한 직무로 본다.
        # 0: 점수 1위와 같은 직함  1: 필요 스킬 보유  2: 점수 1위와 같은 팀  3: 나머지  4: 수용 시간 초과
        top = cands[order[0]]
        for i, c in enumerate(cands):
            if over[i]:
                tier[i] = 4
            elif not role_filter or c.title == top.title:
                tier[i] = 0
            elif req and req & set(c.skills):
                tier[i] = 1
            elif c.team == top.team:
                tier[i] = 2
            else:
                tier[i] = 3
        order = sorted(order, key=lambda i: tier[i])  # 안정 정렬이라 같은 계층 안에서는 점수 순서가 유지된다
    ranked = []
    for i in order:
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
                "over_capacity": bool(over[i]),
                "tier": tier[i],
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
    busy: dict[int, float] | None = None,
    capacity_constraint: bool | None = None,
    role_filter: bool | None = None,
    strategy: str | None = None,
) -> list[dict]:
    """태스크마다 추천 담당자와 상위 3명 후보를 돌려준다.

    strategy(기본값은 설정의 match_strategy)
    - llm_rank: LLM이 팀원 목록을 보고 순위를 매기고, 규칙이 수용 시간을 넘는 사람을 순위에서 건너뛴다.
      LLM 순위에 수용 시간 안의 사람이 없거나 LLM 호출이 실패하면 점수 방식으로 고른다.
    - score: 하이브리드 점수와 규칙으로 고른다(match_tasks_score).
    """
    strategy = strategy or get_settings().match_strategy
    if strategy == "score":
        return match_tasks_score(
            db, tasks, days, project_id, with_reasons, busy, capacity_constraint, role_filter
        )
    try:
        rankings, reasons = llm_rankings(db, tasks, days, busy, "resource", project_id)
    except LLMError:
        return match_tasks_score(db, tasks, days, project_id, with_reasons, busy)

    cands = load_candidates(db, busy)
    vecs = [np.asarray(v) for v in embed_texts([task_text(t) for t in tasks])]
    weeks = max(days / 7, 1.0)
    extra: dict[int, float] = {}
    results = []
    for i, (t, v) in enumerate(zip(tasks, vecs, strict=True)):
        hours = float(t.get("estimate_hours") or 8)
        # 점수 순위는 후보별 근거(스킬·이력·업무량)를 보여 주고, LLM 순위가 모두 막혔을 때 쓴다.
        ranked = score_task(v, cands, extra, t.get("required_skills"), weeks, hours, True, True)
        by_id = {r["user_id"]: r for r in ranked}
        llm_order = [u for u in dict.fromkeys(rankings[i]) if u in by_id]
        feasible = [u for u in llm_order if not by_id[u]["over_capacity"]]
        if feasible:
            chosen, source = feasible[0], "llm"
        else:
            chosen, source = ranked[0]["user_id"], "score"
        skipped = (
            [by_id[u]["name"] for u in llm_order[: llm_order.index(chosen)]] if chosen in llm_order else []
        )
        order = [chosen, *feasible, *(r["user_id"] for r in ranked)]
        candidates = [by_id[u] for u in dict.fromkeys(order)][:3]
        if source == "llm" and not skipped:
            reason = reasons[i]
        elif source == "llm":
            reason = (
                f"AI가 먼저 추천한 {', '.join(skipped)}은(는) 이 태스크를 맡으면 프로젝트 기간의 수용 시간을 넘어 "
                f"규칙에 따라 제외했고, 다음 순위인 {by_id[chosen]['name']}을(를) 배정했습니다."
            )
        else:
            reason = (
                _mock_reason(by_id[chosen])
                + " (AI 추천 후보가 모두 수용 시간을 넘어 점수 순위로 골랐습니다.)"
            )
        extra[chosen] = extra.get(chosen, 0) + hours
        results.append(
            {
                "assignee_id": chosen,
                "candidates": candidates,
                "reason": reason if with_reasons else "",
                "source": source,
                "skipped": skipped,
            }
        )
    return results


def match_tasks_score(
    db: Session,
    tasks: list[dict],
    days: int = 30,
    project_id: int | None = None,
    with_reasons: bool = True,
    busy: dict[int, float] | None = None,
    capacity_constraint: bool | None = None,
    role_filter: bool | None = None,
) -> list[dict]:
    """점수 방식: 하이브리드 점수와 규칙(수용 시간 제약, 후보 계층)으로 고르고, LLM은 추천 사유만 쓴다.

    같은 계획 안에서 먼저 배정된 태스크의 시간을 업무량에 더해, 한 사람에게 몰리지 않게 한다.
    LLM 순위 방식의 예비 경로이자 비교 실험의 기준선이다.
    """
    cands = load_candidates(db, busy)
    cand_by_id = {c.user_id: c for c in cands}
    vecs = [np.asarray(v) for v in embed_texts([task_text(t) for t in tasks])]
    weeks = max(days / 7, 1.0)
    extra: dict[int, float] = {}
    results = []
    for t, v in zip(tasks, vecs, strict=True):
        hours = float(t.get("estimate_hours") or 8)
        ranked = score_task(
            v, cands, extra, t.get("required_skills"), weeks, hours, capacity_constraint, role_filter
        )
        top = ranked[0]
        extra[top["user_id"]] = extra.get(top["user_id"], 0) + hours
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


# 담당자 순위 프롬프트. 기본 방식(llm_rank)과 비교 실험의 LLM 단독 방식이 같은 프롬프트를 쓴다.
LLM_ONLY_SYSTEM = """너는 IT 스타트업의 담당자 배정 에이전트다.
팀원 목록(스킬, 현재 업무량)을 보고 각 태스크에 가장 적합한 담당자를 5순위까지 고른다.
ranking의 첫 번째 사람이 실제로 배정된다. 태스크들은 이번 프로젝트에서 한꺼번에 배정되므로,
한 사람의 (현재 업무 + 이번에 앞서 배정한 태스크 시간 + 이 태스크 시간)이 그 사람의 수용 가능 시간을 넘으면
그 사람을 첫 번째로 두지 않는다.
reason에는 첫 번째 사람을 고른 이유를 스킬, 비슷한 경험, 업무량을 근거로 한두 문장 한국어로 쓴다.
출력 형식: {"assignments": [{"index": 0, "ranking": [팀원 id, 팀원 id, 팀원 id, 팀원 id, 팀원 id], "reason": "..."}]}"""


def llm_rankings(
    db: Session,
    tasks: list[dict],
    days: int = 30,
    busy: dict[int, float] | None = None,
    agent: str = "resource_llm_only",
    project_id: int | None = None,
) -> tuple[list[list[int]], list[str]]:
    """LLM이 팀원 목록(스킬, 현재 업무, 수용 시간)을 보고 태스크마다 5순위까지 매긴 순위와 사유."""
    cands = load_candidates(db, busy)
    roster = "\n".join(
        f"id={c.user_id} {c.name} ({c.team}팀 {c.title}) 스킬: {', '.join(c.skills)} "
        f"/ 현재 업무 {c.load_hours:.0f}시간, 프로젝트 기간 {days}일 동안 수용 가능 "
        f"{c.capacity * max(days / 7, 1):.0f}시간"
        for c in cands
    )

    def mock():
        # mock에서는 스킬 유사도만으로 고른다.
        vecs = [np.asarray(v) for v in embed_texts([task_text(t) for t in tasks])]
        return {
            "assignments": [
                {
                    "index": i,
                    "ranking": [
                        cands[j].user_id for j in np.argsort([-(v @ c.skill_vec) for c in cands])[:5]
                    ],
                    "reason": "스킬 유사도가 가장 높습니다.",
                }
                for i, v in enumerate(vecs)
            ]
        }

    body = "\n".join(
        f"[{i}] {t['title']}: {t.get('description', '')} (예상 {t.get('estimate_hours', 8):.0f}시간)"
        for i, t in enumerate(tasks)
    )
    out = call_json(
        agent, LLM_ONLY_SYSTEM, f"팀원:\n{roster}\n\n태스크:\n{body}", mock=mock, project_id=project_id
    )
    rankings: list[list[int]] = [[] for _ in tasks]
    reasons = ["" for _ in tasks]
    for a in out.get("assignments", []) if isinstance(out, dict) else []:
        idx = a.get("index") if isinstance(a, dict) else None
        if isinstance(idx, int) and 0 <= idx < len(tasks):
            rankings[idx] = [int(x) for x in a.get("ranking", []) if str(x).isdigit()]
            reasons[idx] = str(a.get("reason", ""))
    return rankings, reasons


def match_tasks_llm_only(
    db: Session, tasks: list[dict], days: int = 30, busy: dict[int, float] | None = None
) -> list[list[int]]:
    """비교 실험용: 규칙 없이 LLM 순위를 그대로 쓴다."""
    return llm_rankings(db, tasks, days, busy)[0]
