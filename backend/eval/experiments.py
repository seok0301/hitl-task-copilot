"""최종 평가 실험. 가설과 판정 기준은 eval/PROTOCOL.md에 있다.

실행 예: `uv run python -m eval.experiments --sets hard --only matching planning --repeats 5`
- 평가 DB(EVAL_DATABASE_URL, 기본 hitl_eval)를 seed로 초기화한 뒤 돌린다.
- LLM 호출은 스레드로 병렬 실행한다.
- 결과는 docs/report/results/exp-<시각>.json, .md로 저장한다.
"""

import os

os.environ["DATABASE_URL"] = os.environ.get(
    "EVAL_DATABASE_URL", "postgresql+psycopg://hitl:hitl@localhost:5442/hitl_eval"
)

import argparse
import json
import random
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.agents.planner import plan_tasks
from app.agents.resource import load_candidates, match_tasks, match_tasks_llm_only, task_text
from app.config import ROOT_DIR, get_settings
from app.db import SessionLocal
from app.llm.provider import call_json, model_name
from app.models import Role
from app.rag.retriever import Hit, visible_documents
from app.seed.heldout_set import HELDOUT_FEEDBACK, HELDOUT_MATCH_PROJECTS
from eval.plan_sets import DEV_FEEDBACK, keyword_hit, plan_sets
from eval.run_eval import covered_by_judge, llm_skill_tags, match_sets

WORKERS = 16
REVISIONS = ("regenerate", "rewrite", "edit")


def pmap(fn, items):
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def cluster_bootstrap(diffs: dict[str, list[float]], n: int = 10000, seed: int = 0) -> dict:
    """프로젝트 단위로 다시 뽑는 부트스트랩. diffs: 프로젝트 → 그 프로젝트 태스크들의 차이 값."""
    rng = np.random.default_rng(seed)
    keys = [k for k, v in diffs.items() if v]
    flat = [x for k in keys for x in diffs[k]]
    if not flat:
        return {"mean": None, "lo": None, "hi": None}
    sums = np.array([sum(diffs[k]) for k in keys])
    cnts = np.array([len(diffs[k]) for k in keys])
    idx = rng.integers(0, len(keys), size=(n, len(keys)))
    boot = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return {
        "mean": round(float(np.mean(flat)), 4),
        "lo": round(float(np.percentile(boot, 2.5)), 4),
        "hi": round(float(np.percentile(boot, 97.5)), 4),
    }


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    if not a:
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else 1.0


# ── 담당자 추천 ──
def matching_projects(name: str) -> list[dict]:
    if name == "heldout":
        return [
            {
                "id": pid,
                "days": days,
                "busy": busy,
                "tasks": [
                    {"title": t, "description": d, "hours": float(h), "gt": gt, "ok": ok, "kind": kind}
                    for t, d, h, gt, ok, kind in tasks
                ],
            }
            for pid, _title, days, busy, tasks in HELDOUT_MATCH_PROJECTS
        ]
    projects = match_sets()[name]
    for p in projects:
        for t in p["tasks"]:
            t["kind"] = "-"
    return projects


def _match(db, ts, p, weights, cap, tier) -> list[int]:
    s = get_settings()
    saved = s.match_alpha, s.match_beta, s.match_gamma
    s.match_alpha, s.match_beta, s.match_gamma = weights
    try:
        res = match_tasks(
            db,
            ts,
            days=p["days"],
            with_reasons=False,
            busy=p["busy"],
            capacity_constraint=cap,
            role_filter=tier,
            strategy="score",
        )
    finally:
        s.match_alpha, s.match_beta, s.match_gamma = saved
    return [[c["user_id"] for c in r["candidates"]] for r in res]


def capacity_filter(db, p: dict, rankings: list[list[int]]) -> list[list[int]]:
    """LLM 단독 + 규칙 후처리: LLM 순위에서 배정하면 수용 시간을 넘는 사람을 건너뛴다. 모두 넘으면 1순위를 쓴다."""
    cands = {c.user_id: c for c in load_candidates(db, p["busy"])}
    weeks = max(p["days"] / 7, 1.0)
    assigned: dict[int, float] = {}
    out = []
    for t, rank in zip(p["tasks"], rankings, strict=True):
        ok = [
            u
            for u in rank
            if u in cands
            and cands[u].load_hours + assigned.get(u, 0) + t["hours"] <= cands[u].capacity * weeks
        ]
        chosen = ok[0] if ok else (rank[0] if rank else -1)
        assigned[chosen] = assigned.get(chosen, 0) + t["hours"]
        out.append([chosen, *[u for u in rank if u != chosen]])
    return out


def overload_count(db, p: dict, firsts: list[int]) -> int:
    cands = {c.user_id: c for c in load_candidates(db, p["busy"])}
    weeks = max(p["days"] / 7, 1.0)
    assigned: dict[int, float] = {}
    n = 0
    for uid, t in zip(firsts, p["tasks"], strict=True):
        if uid not in cands:
            continue
        assigned[uid] = assigned.get(uid, 0) + t["hours"]
        if cands[uid].load_hours + assigned[uid] > cands[uid].capacity * weeks:
            n += 1
    return n


def run_matching(name: str, repeats: int) -> dict:
    projects = matching_projects(name)
    s = get_settings()
    default = (s.match_alpha, s.match_beta, s.match_gamma)
    skill = (1.0, 0.0, 0.0)
    variants = {
        # 점수 방식(초기 설계): 점수 + 수용 시간 규칙 + 후보 계층, 계층 없음, 규칙 없음, 스킬만
        "score_tier": (default, True, True),
        "score_cap": (default, True, False),
        "score_only": (default, False, False),
        "skill_only": (skill, False, False),
    }
    jobs = [(p, r) for r in range(repeats) for p in projects]

    def llm_part(job):
        p, _r = job
        base = [{"title": t["title"], "description": t["description"]} for t in p["tasks"]]
        with SessionLocal() as db:
            tags = llm_skill_tags(db, base)
            ts = [{**b, "estimate_hours": t["hours"]} for b, t in zip(base, p["tasks"], strict=True)]
            llm = match_tasks_llm_only(db, ts, days=p["days"], busy=p["busy"])
            # 제안 방식: 시스템이 실제로 쓰는 match_tasks(LLM 순위 + 수용 시간 규칙)를 그대로 부른다.
            sys_ts = [{**x, "required_skills": tg} for x, tg in zip(ts, tags, strict=True)]
            res = match_tasks(db, sys_ts, days=p["days"], busy=p["busy"], strategy="llm_rank")
            proposed = [[c["user_id"] for c in r["candidates"]] for r in res]
        return tags, llm, proposed

    llm_out = pmap(llm_part, jobs)
    rows = []
    with SessionLocal() as db:
        for (p, r), (tags, llm, proposed) in zip(jobs, llm_out, strict=True):
            preds = {"proposed": proposed}
            for m, (w, cap, tier) in variants.items():
                ts = [
                    {
                        "title": t["title"],
                        "description": t["description"],
                        "required_skills": tg,
                        "estimate_hours": t["hours"],
                    }
                    for t, tg in zip(p["tasks"], tags, strict=True)
                ]
                preds[m] = _match(db, ts, p, w, cap, tier)
            preds["llm_only"] = llm
            overloads = {m: overload_count(db, p, [x[0] if x else -1 for x in v]) for m, v in preds.items()}
            for i, t in enumerate(p["tasks"]):
                rows.append(
                    {
                        "project": p["id"],
                        "repeat": r,
                        "task": t["title"],
                        "kind": t["kind"],
                        "gt": t["gt"],
                        "ok": t["ok"],
                        "tags": tags[i],
                        "pred": {m: v[i][:5] for m, v in preds.items()},
                    }
                )
            for m, n in overloads.items():
                rows.append({"project": p["id"], "repeat": r, "overload": True, "method": m, "count": n})

    task_rows = [x for x in rows if "task" in x]
    over_rows = [x for x in rows if x.get("overload")]
    methods = list(task_rows[0]["pred"])

    def correct(x, m, strict=False):
        pr = x["pred"][m]
        if not pr:
            return 0.0
        return float(pr[0] == x["gt"] if strict else pr[0] in {x["gt"], *x["ok"]})

    summary = {}
    for m in methods:
        per_rep = []
        for r in range(repeats):
            xs = [x for x in task_rows if x["repeat"] == r]
            per_rep.append(
                {
                    "top1": mean([correct(x, m) for x in xs]),
                    "top1_strict": mean([correct(x, m, True) for x in xs]),
                    "top3": mean([float(bool({x["gt"], *x["ok"]} & set(x["pred"][m][:3]))) for x in xs]),
                    "overload": sum(o["count"] for o in over_rows if o["method"] == m and o["repeat"] == r),
                }
            )
        # 반복 일치율: 모든 반복에서 1순위가 같은 태스크의 비율
        by_task: dict[tuple, set] = {}
        for x in task_rows:
            by_task.setdefault((x["project"], x["task"]), set()).add(x["pred"][m][0] if x["pred"][m] else -1)
        kinds = {}
        for k in sorted({x["kind"] for x in task_rows}):
            xs = [x for x in task_rows if x["kind"] == k]
            kinds[k] = {"n": len(xs) // repeats, "top1": round(mean([correct(x, m) for x in xs]), 4)}
        summary[m] = {
            **{
                k: {
                    "mean": round(mean([pr[k] for pr in per_rep]), 4),
                    "sd": round(statistics.pstdev([pr[k] for pr in per_rep]), 4),
                    "min": min(pr[k] for pr in per_rep),
                    "max": max(pr[k] for pr in per_rep),
                }
                for k in ("top1", "top1_strict", "top3", "overload")
            },
            "consistency": round(sum(1 for v in by_task.values() if len(v) == 1) / len(by_task), 4),
            "by_kind": kinds,
        }

    # 짝지은 차이: 태스크마다 반복 평균 정답률의 차이, 프로젝트 단위 부트스트랩
    def task_means(m, strict=False):
        acc: dict[tuple, list[float]] = {}
        for x in task_rows:
            acc.setdefault((x["project"], x["task"]), []).append(correct(x, m, strict))
        return {k: statistics.mean(v) for k, v in acc.items()}

    comparisons = {}
    for a, b in [("proposed", "llm_only"), ("proposed", "score_tier"), ("score_tier", "score_cap")]:
        for strict in (False, True):
            ta, tb = task_means(a, strict), task_means(b, strict)
            diffs: dict[str, list[float]] = {}
            for (proj, task), va in ta.items():
                diffs.setdefault(proj, []).append(va - tb[(proj, task)])
            comparisons[f"{a} - {b}{' (정답만)' if strict else ''}"] = cluster_bootstrap(diffs)
    return {
        "summary": summary,
        "comparisons": comparisons,
        "n_tasks": len(task_rows) // repeats,
        "rows": rows,
    }


# ── 계획 ──
def random_hits(db, project: dict, k: int, seed: int) -> list[Hit]:
    """대조군: 근거 문서를 뺀 문서 중에서 무작위로 k개를 고른다."""
    gold = {g for r in project["ref"] for g in r["gold_docs"]}
    docs = [d for d in visible_documents(db, Role.MANAGER) if d.title not in gold]
    pick = random.Random(seed).sample(docs, k)
    return [Hit(d.id, d.title, d.content, d.access_level, 0.0) for d in pick]


def oracle_hits(db, project: dict) -> list[Hit]:
    """상한: 교훈 태스크의 근거 문서만 준다."""
    gold = {g for r in project["ref"] for g in r["gold_docs"]}
    return [
        Hit(d.id, d.title, d.content, d.access_level, 1.0)
        for d in visible_documents(db, Role.MANAGER)
        if d.title in gold
    ]


def run_planning(name: str, conditions: list[str], repeats: int) -> dict:
    projects = plan_sets()[name]
    has_lesson = any(r["lesson"] for p in projects for r in p["ref"])
    conds = [c for c in conditions if has_lesson or c in ("no_rag", "chunk", "lexical", "draft")]
    jobs = [(p, c, r) for p in projects for c in conds for r in range(repeats)]

    def one(job):
        p, c, r = job
        with SessionLocal() as db:
            if c == "no_rag":
                gen, hits = plan_tasks(db, p["title"], p["goal"], p["days"], Role.MANAGER, use_rag=False)
            elif c == "random":
                gen, hits = plan_tasks(db, p["title"], p["goal"], p["days"], hits=random_hits(db, p, 6, r))
            elif c == "oracle":
                gen, hits = plan_tasks(db, p["title"], p["goal"], p["days"], hits=oracle_hits(db, p))
            else:
                gen, hits = plan_tasks(db, p["title"], p["goal"], p["days"], rag_mode=c)
        gen_text = [task_text(g) for g in gen]
        ref = [x["text"] for x in p["ref"]]
        judge = covered_by_judge(ref, gen_text)
        plan_text = "\n".join(gen_text)
        sources = [h.title for h in hits]
        return {
            "project": p["id"],
            "cond": c,
            "repeat": r,
            "tasks": [g["title"] for g in gen],
            "sources": sources,
            "judge": judge,
            "keyword": [keyword_hit(x["keywords"], plan_text) if x["keywords"] else None for x in p["ref"]],
            "gold_found": [
                bool(set(x["gold_docs"]) & set(sources)) if x["gold_docs"] else None for x in p["ref"]
            ],
        }

    rows = pmap(one, jobs)
    ref_of = {p["id"]: p["ref"] for p in projects}

    def flags(row, which, lesson: bool | None):
        return [
            f
            for f, x in zip(row[which], ref_of[row["project"]], strict=True)
            if (lesson is None or x["lesson"] == lesson) and f is not None
        ]

    summary = {}
    for c in conds:
        rs = [x for x in rows if x["cond"] == c]
        cov = lambda which, lesson, rs=rs: mean(
            [mean(flags(x, which, lesson)) for x in rs if flags(x, which, lesson)]
        )
        summary[c] = {
            "coverage": cov("judge", None),
            "lesson_judge": cov("judge", True),
            "general_judge": cov("judge", False),
            "lesson_keyword": cov("keyword", True),
            "gold_recall": cov("gold_found", True),
            "tasks": mean([len(x["tasks"]) for x in rs]),
        }
        summary[c] = {k: (round(v, 4) if v is not None else None) for k, v in summary[c].items()}

    # 판정 일치도(교훈 태스크에서 LLM 심사 대 키워드)
    ja, kb = [], []
    for x in rows:
        for j, k, ref in zip(x["judge"], x["keyword"], ref_of[x["project"]], strict=True):
            if ref["lesson"] and k is not None:
                ja.append(j)
                kb.append(k)
    agreement = {
        "kappa": cohen_kappa(ja, kb),
        "n": len(ja),
        "raw": round(mean([a == b for a, b in zip(ja, kb, strict=True)]) or 0, 4),
    }

    # 근거 문서를 찾았을 때와 못 찾았을 때의 교훈 커버리지(검색을 한 조건만)
    found, missed = [], []
    for x in rows:
        if x["cond"] in ("no_rag", "random", "oracle"):
            continue
        for j, g, ref in zip(x["judge"], x["gold_found"], ref_of[x["project"]], strict=True):
            if ref["lesson"] and g is not None:
                (found if g else missed).append(j)
    by_retrieval = {
        "found": [round(mean(found) or 0, 4), len(found)],
        "missed": [round(mean(missed) or 0, 4), len(missed)],
    }

    comparisons = {}
    if has_lesson:

        def lesson_task_means(c, which):
            acc: dict[tuple, list[float]] = {}
            for x in rows:
                if x["cond"] != c:
                    continue
                for i, (f, ref) in enumerate(zip(x[which], ref_of[x["project"]], strict=True)):
                    if ref["lesson"] and f is not None:
                        acc.setdefault((x["project"], i), []).append(float(f))
            return {k: statistics.mean(v) for k, v in acc.items()}

        def general_task_means(c):
            acc: dict[tuple, list[float]] = {}
            for x in rows:
                if x["cond"] != c:
                    continue
                for i, (f, ref) in enumerate(zip(x["judge"], ref_of[x["project"]], strict=True)):
                    if not ref["lesson"]:
                        acc.setdefault((x["project"], i), []).append(float(f))
            return {k: statistics.mean(v) for k, v in acc.items()}

        rag_conds = [c for c in conds if c not in ("no_rag", "random", "oracle")]
        for a in [*rag_conds, "oracle"]:
            for b in ("no_rag", "random"):
                if a not in conds or b not in conds:
                    continue
                for which in ("judge", "keyword"):
                    ta, tb = lesson_task_means(a, which), lesson_task_means(b, which)
                    diffs: dict[str, list[float]] = {}
                    for k, v in ta.items():
                        diffs.setdefault(k[0], []).append(v - tb[k])
                    comparisons[f"교훈({which}) {a} - {b}"] = cluster_bootstrap(diffs)
            if "no_rag" in conds:
                ga, gb = general_task_means(a), general_task_means("no_rag")
                diffs = {}
                for k, v in ga.items():
                    diffs.setdefault(k[0], []).append(v - gb[k])
                comparisons[f"일반(judge) {a} - no_rag"] = cluster_bootstrap(diffs)
    return {
        "summary": summary,
        "agreement": agreement,
        "by_retrieval": by_retrieval,
        "comparisons": comparisons,
        "rows": rows,
    }


# ── 관리자 피드백(HITL) ──
SEPARATE_SYSTEM = """너는 프로젝트 계획을 평가하는 심사자다.
생성된 계획에 기준 작업을 수행하는 별도의 태스크가 있는지 판단한다.
다른 태스크의 설명 안에서 언급만 되거나 일부로 포함된 것은 별도 태스크로 보지 않는다.
출력 형식: {"separate": true}"""


def has_separate_task(ref: str, gen: list[str]) -> bool:
    """삭제 요청 채점용. covered_by_judge는 다른 태스크 설명 속 언급도 포함으로 보므로 삭제 판정에는 맞지 않는다."""
    user = f"기준 작업: {ref}\n\n생성된 계획:\n" + "\n".join(f"- {g}" for g in gen)
    out = call_json("eval_judge_separate", SEPARATE_SYSTEM, user, mock=lambda: {"separate": False})
    return bool(out.get("separate")) if isinstance(out, dict) else False


def run_feedback(name: str, repeats: int) -> dict:
    """관리자 피드백 반영 실험. 같은 초안에 대해 두 재생성 방식을 비교한다.

    - regenerate: 피드백만 주고 다시 계획한다(초기 구현).
    - rewrite: 이전 초안과 피드백을 주고 LLM이 목록 전체를 다시 쓴다.
    - edit: LLM은 편집 명령만 내고 코드가 적용한다.
    두 방식 모두 초안과 같은 참고 문서를 쓴다.
    """
    sets = plan_sets()
    if name == "heldout":
        projects = {p["id"]: p for p in sets["heldout"]}
        items = HELDOUT_FEEDBACK
    else:
        projects = {p["id"]: p for p in sets["basic"] + sets["hard"]}
        items = DEV_FEEDBACK
    jobs = [(fb, r) for fb in items for r in range(repeats)]

    def one(job):
        (pid, kind, text, target), r = job
        p = projects[pid]
        tref = next((x["text"] for x in p["ref"] if x["title"] == target), None)
        with SessionLocal() as db:
            draft, hits = plan_tasks(db, p["title"], p["goal"], p["days"])
            draft_text = [task_text(d) for d in draft]
            # 보존 대상: 요청과 무관한 초안 태스크. remove면 빼라고 한 작업에 해당하는 초안 태스크는 뺀다.
            keep = draft_text
            if kind == "remove":
                same = covered_by_judge(draft_text, [tref])
                keep = [t for t, sm in zip(draft_text, same, strict=True) if not sm]
            out = {}
            for mode in REVISIONS:
                regen, _ = plan_tasks(
                    db,
                    p["title"],
                    p["goal"],
                    p["days"],
                    feedback=text,
                    previous=draft,
                    hits=hits,
                    revision=mode,
                )
                regen_text = [task_text(g) for g in regen]
                if kind == "limit":
                    reflected = len(regen) <= int(target)
                elif kind == "add":
                    reflected = covered_by_judge([tref], regen_text)[0]
                else:
                    reflected = not has_separate_task(tref, regen_text)
                preserved = mean(covered_by_judge(keep, regen_text)) if keep and kind != "limit" else None
                out[mode] = {
                    "reflected": reflected,
                    "preserved": preserved,
                    "tasks": [g["title"] for g in regen],
                }
        return {"project": pid, "kind": kind, "repeat": r, "draft": [d["title"] for d in draft], **out}

    rows = pmap(one, jobs)
    summary = {}
    for mode in REVISIONS:
        summary[mode] = {
            "reflected": round(mean([float(x[mode]["reflected"]) for x in rows]), 4),
            "preserved": round(mean([x[mode]["preserved"] for x in rows]) or 0, 4),
            "by_kind": {
                k: round(mean([float(x[mode]["reflected"]) for x in rows if x["kind"] == k]), 4)
                for k in ("add", "remove", "limit")
            },
        }
    # 짝지은 차이(보존률, 프로젝트 단위 부트스트랩)
    comps = {}
    for a, b in [("edit", "regenerate"), ("edit", "rewrite"), ("rewrite", "regenerate")]:
        diffs: dict[str, list[float]] = {}
        for x in rows:
            va, vb = x[a]["preserved"], x[b]["preserved"]
            if va is not None and vb is not None:
                diffs.setdefault(x["project"], []).append(va - vb)
        comps[f"보존률 {a} - {b}"] = cluster_bootstrap(diffs)
    return {"summary": summary, "comparisons": comps, "n": len(items), "rows": rows}


# ── 보고 에이전트 ──
NUM = re.compile(r"\d+(?:\.\d+)?")


def _numbers(obj) -> set[float]:
    """지표 JSON에 나오는 숫자와, 목록 길이·합계·평균처럼 그 숫자에서 바로 나오는 값."""
    vals: set[float] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, int | float):
            vals.add(float(o))
        elif isinstance(o, str):
            vals.update(float(x) for x in NUM.findall(o))
        elif isinstance(o, dict):
            vals.add(float(len(o)))
            vals.update(
                float(v) for v in o.values() if isinstance(v, int | float) and not isinstance(v, bool)
            )
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            vals.add(float(len(o)))
            for v in o:
                walk(v)

    walk(obj)
    if isinstance(obj, list) and obj:
        for key in ("progress", "task_count", "days_left", "avg_review_coverage"):
            xs = [m[key] for m in obj if m.get(key) is not None]
            if not xs:
                continue
            vals.update({float(sum(xs)), round(sum(xs) / len(xs), 1)})
        for st in ("TODO", "IN_PROGRESS", "SUBMITTED", "DONE"):
            vals.add(float(sum(m["status_counts"].get(st, 0) for m in obj)))
        vals.add(float(sum(len(m["overdue_tasks"]) for m in obj)))
        for m in obj:
            n, c = m["task_count"], m["status_counts"]
            vals.add(float(n - c.get("DONE", 0)))
            vals.add(float(sum(m["remaining_hours_by_person"].values())))
            # 상태별 비율(예: 6개 중 4개 완료 → 66.7%)과 "50%대" 같은 구간 표현
            for k in range(n + 1):
                vals.add(round(100 * k / n, 1))
            vals.add(float(int(m["progress"] // 10 * 10)))
    return vals


def synthetic_metrics(seed: int) -> list[dict]:
    rng = random.Random(seed)
    titles = [
        "정기구독 일시정지",
        "회원 등급제",
        "검색 자동완성",
        "앱 2.4 업데이트",
        "파트너 정산 API",
        "블랙프라이데이 준비",
    ]
    people = [
        "이서연",
        "최민준",
        "정하은",
        "강지우",
        "윤서준",
        "임채원",
        "송민서",
        "문지호",
        "류다은",
        "홍재원",
    ]
    out = []
    for pid, title in enumerate(rng.sample(titles, rng.randint(2, 4)), start=1):
        n = rng.randint(4, 9)
        counts = {"TODO": 0, "IN_PROGRESS": 0, "SUBMITTED": 0, "DONE": 0}
        for _ in range(n):
            counts[rng.choice(list(counts))] += 1
        overdue = [f"{title} 태스크 {i}" for i in range(rng.randint(0, 2))]
        out.append(
            {
                "project_id": pid,
                "title": title,
                "deadline": f"2026-{rng.randint(11, 12)}-{rng.randint(10, 28)}",
                "days_left": rng.randint(-3, 40),
                "task_count": n,
                "status_counts": counts,
                "progress": round(rng.uniform(5, 95), 1),
                "overdue_tasks": overdue,
                "avg_review_coverage": round(rng.uniform(55, 98), 1)
                if counts["SUBMITTED"] + counts["DONE"]
                else None,
                "remaining_hours_by_person": {
                    p: rng.choice([8, 16, 24, 32, 40]) for p in rng.sample(people, rng.randint(2, 5))
                },
            }
        )
    return out


def run_reporter(repeats: int) -> dict:
    from app.agents.reporter import executive_brief, review_submission
    from app.models import Task
    from eval.reporter_set import REVIEW_CASES

    # 1. 산출물 검토: full > partial > off 순서
    jobs = [(c, kind, r) for c in REVIEW_CASES for kind in ("full", "partial", "off") for r in range(repeats)]

    def review(job):
        (title, spec, full, partial, off), kind, r = job
        content = {"full": full, "partial": partial, "off": off}[kind]
        task = Task(title=title, description=spec, project_id=None)
        return {
            "task": title,
            "kind": kind,
            "repeat": r,
            "coverage": review_submission(task, content)["coverage"],
        }

    rrows = pmap(review, jobs)
    cov = {(x["task"], x["kind"], x["repeat"]): x["coverage"] for x in rrows}
    ordered = [
        cov[(c[0], "full", r)] > cov[(c[0], "partial", r)] > cov[(c[0], "off", r)]
        for c in REVIEW_CASES
        for r in range(repeats)
    ]
    review_summary = {
        "order_rate": round(mean([float(o) for o in ordered]), 4),
        "mean": {
            k: round(mean([x["coverage"] for x in rrows if x["kind"] == k]), 1)
            for k in ("full", "partial", "off")
        },
        "n": len(ordered),
    }

    # 2. 브리핑: 입력 지표에 없는 숫자의 비율
    scenarios = [synthetic_metrics(i) for i in range(15)]
    bjobs = [(i, r) for i in range(len(scenarios)) for r in range(repeats)]

    def brief(job):
        i, r = job
        b = executive_brief({"metrics": scenarios[i]})["brief"]
        text = json.dumps(b, ensure_ascii=False)
        nums = [float(x) for x in NUM.findall(text)]
        allowed = _numbers(scenarios[i])
        bad = [x for x in nums if not any(abs(x - a) <= 0.51 for a in allowed)]
        return {"scenario": i, "repeat": r, "brief": b, "numbers": len(nums), "unsupported": bad}

    brows = pmap(brief, bjobs)
    total = sum(x["numbers"] for x in brows)
    bad = sum(len(x["unsupported"]) for x in brows)
    brief_summary = {
        "numbers": total,
        "unsupported": bad,
        "unsupported_rate": round(bad / total, 4) if total else None,
        "briefs_with_unsupported": sum(1 for x in brows if x["unsupported"]),
        "briefs": len(brows),
    }
    return {"review": review_summary, "brief": brief_summary, "review_rows": rrows, "brief_rows": brows}


# ── 출력 ──
def to_markdown(res: dict) -> str:
    m = res["meta"]
    out = [
        f"# 실험 결과 ({m['started_at'][:19]} UTC)",
        "",
        f"- 모델 `{m['model']}`, 반복 {m['repeats']}회, 검색 방식 기본값 `{m['rag_mode']}`",
        "",
    ]
    for name, r in res.get("matching", {}).items():
        out += [
            f"## 담당자 추천 ({name}, 태스크 {r['n_tasks']}개)",
            "",
            "| 방법 | Top-1 | Top-1(정답만) | Top-3 | 과부하 배정(건, 반복 평균 [최소, 최대]) | 반복 일치율 |",
            "|---|---|---|---|---|---|",
        ]
        for meth, s in r["summary"].items():
            f = lambda k, s=s: f"{s[k]['mean']:.3f} ± {s[k]['sd']:.3f}"
            o = s["overload"]
            out.append(
                f"| {meth} | {f('top1')} | {f('top1_strict')} | {f('top3')} | {o['mean']:.1f} [{o['min']}, {o['max']}] | {s['consistency']:.3f} |"
            )
        kinds = sorted(next(iter(r["summary"].values()))["by_kind"])
        if kinds != ["-"]:
            out += [
                "",
                "유형별 Top-1",
                "",
                "| 방법 | "
                + " | ".join(f"{k}({next(iter(r['summary'].values()))['by_kind'][k]['n']})" for k in kinds)
                + " |",
                "|---|" + "---|" * len(kinds),
            ]
            for meth, s in r["summary"].items():
                out.append(
                    f"| {meth} | " + " | ".join(f"{s['by_kind'][k]['top1']:.3f}" for k in kinds) + " |"
                )
        out += ["", "짝지은 차이(프로젝트 단위 부트스트랩 95% 신뢰구간)", ""]
        for k, v in r["comparisons"].items():
            out.append(f"- {k}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
        out.append("")
    for name, r in res.get("planning", {}).items():
        out += [
            f"## 계획 ({name})",
            "",
            "| 조건 | 전체 | 교훈(LLM 심사) | 교훈(키워드) | 일반(LLM 심사) | 근거 문서 검색률 | 태스크 수 |",
            "|---|---|---|---|---|---|---|",
        ]
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        for c, s in r["summary"].items():
            out.append(
                f"| {c} | {fmt(s['coverage'])} | {fmt(s['lesson_judge'])} | {fmt(s['lesson_keyword'])} | {fmt(s['general_judge'])} | {fmt(s['gold_recall'])} | {s['tasks']:.1f} |"
            )
        a = r["agreement"]
        out += [
            "",
            f"- 교훈 태스크 판정 일치도(LLM 심사 대 키워드): kappa {a['kappa']}, 단순 일치 {a['raw']} (n={a['n']})",
        ]
        b = r["by_retrieval"]
        out.append(
            f"- 근거 문서를 찾았을 때 교훈 커버리지 {b['found'][0]} (n={b['found'][1]}), 못 찾았을 때 {b['missed'][0]} (n={b['missed'][1]})"
        )
        for k, v in r["comparisons"].items():
            out.append(f"- {k}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
        out.append("")
    for name, r in res.get("feedback", {}).items():
        out += [
            f"## 관리자 피드백 반영 ({name}, 요청 {r['n']}건)",
            "",
            "| 방식 | 반영률 | 보존률 | add | remove | limit |",
            "|---|---|---|---|---|---|",
        ]
        for mode, s in r["summary"].items():
            bk = s["by_kind"]
            out.append(
                f"| {mode} | {s['reflected']:.3f} | {s['preserved']:.3f} | {bk['add']:.3f} | {bk['remove']:.3f} | {bk['limit']:.3f} |"
            )
        out.append("")
        for k, d in r["comparisons"].items():
            out.append(f"- {k}: {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]")
        out.append("")
    if "reporter" in res:
        rv, br = res["reporter"]["review"], res["reporter"]["brief"]
        out += [
            "## 보고 에이전트",
            "",
            f"- 산출물 검토: 충족도가 완전 > 일부 > 무관 순서로 매겨진 비율 {rv['order_rate']:.3f} (n={rv['n']}), "
            f"평균 충족도 완전 {rv['mean']['full']}, 일부 {rv['mean']['partial']}, 무관 {rv['mean']['off']}",
            f"- 경영진 브리핑: 숫자 {br['numbers']}개 중 입력 지표에서 찾을 수 없는 숫자 {br['unsupported']}개 "
            f"({br['unsupported_rate']}), 그런 숫자가 있는 브리핑 {br['briefs_with_unsupported']}/{br['briefs']}",
            "",
        ]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=["matching", "planning", "feedback"])
    ap.add_argument("--sets", nargs="*", default=["basic", "hard"])
    ap.add_argument(
        "--conditions", nargs="*", default=["no_rag", "random", "chunk", "lexical", "draft", "oracle"]
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rag-mode", default=None)
    ap.add_argument("--feedback-sets", nargs="*", default=["dev"])
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--out", default=str(ROOT_DIR / "docs" / "report" / "results"))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not args.no_seed:
        from app.seed.seed import run as seed

        seed()
    s = get_settings()
    rag_mode = args.rag_mode or s.rag_mode
    started = datetime.now(UTC)
    res: dict = {
        "meta": {
            "model": model_name(),
            "repeats": args.repeats,
            "rag_mode": rag_mode,
            "weights": [s.match_alpha, s.match_beta, s.match_gamma],
            "sets": args.sets,
            "conditions": args.conditions,
            "started_at": started.isoformat(),
            "argv": args.__dict__,
        }
    }
    if "matching" in args.only:
        res["matching"] = {n: run_matching(n, args.repeats) for n in args.sets}
    if "planning" in args.only:
        res["planning"] = {n: run_planning(n, args.conditions, args.repeats) for n in args.sets}
    if "feedback" in args.only:
        res["feedback"] = {n: run_feedback(n, args.repeats) for n in args.feedback_sets}
    if "reporter" in args.only:
        res["reporter"] = run_reporter(args.repeats)
    res["meta"]["finished_at"] = datetime.now(UTC).isoformat()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    (out / f"exp-{stamp}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    md = to_markdown(res)
    (out / f"exp-{stamp}.md").write_text(md)
    print(md)
    print(f"저장: {out / f'exp-{stamp}.md'}")


if __name__ == "__main__":
    main()
