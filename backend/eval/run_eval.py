"""평가 스크립트. 실행: `uv run python -m eval.run_eval [--out DIR] [--repeats N] [--only ...]`

1. 담당자 추천: 하이브리드 점수 / 스킬 유사도 단독 / LLM 단독의 Top-1·Top-3 정확도, 업무 편중도, 과부하 배정
2. 계획 에이전트: RAG 사용 여부에 따른 기준 태스크 커버리지 (어려운 세트는 교훈 태스크 커버리지를 따로 본다)
3. 비용: 에이전트별 LLM 호출 소요 시간과 토큰
평가 세트는 기본 세트(basic)와 어려운 세트(hard) 두 가지다. 결과는 JSON과 마크다운 요약으로 저장한다.
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from app.agents.planner import plan_tasks, skill_vocabulary
from app.agents.resource import load_candidates, match_tasks, match_tasks_llm_only, task_text
from app.config import ROOT_DIR, get_settings
from app.db import SessionLocal
from app.llm.embeddings import embed_texts
from app.llm.provider import call_json, model_name
from app.models import LLMCall, Role
from app.seed.eval_set import EVAL_PROJECTS, HARD_MATCH_PROJECTS, HARD_PLAN_PROJECTS

COVERAGE_THRESHOLD = 0.6  # 임베딩 기반 커버리지에서 "같은 태스크"로 볼 최소 코사인 유사도


# ── 평가 세트를 공통 형식으로 맞춘다 ──
def match_sets() -> dict[str, list[dict]]:
    """태스크: {title, description, hours, gt, ok, gold_tags}. 기본 세트만 연구자가 적은 스킬 태그가 있다."""
    basic = [
        {
            "id": pid,
            "days": days,
            "busy": {},
            "tasks": [
                {"title": t, "description": d, "hours": 16.0, "gt": gt, "ok": ok, "gold_tags": sk}
                for t, d, sk, gt, ok in tasks
            ],
        }
        for pid, _title, _goal, days, tasks in EVAL_PROJECTS
    ]
    hard = [
        {
            "id": pid,
            "days": days,
            "busy": busy,
            "tasks": [
                {"title": t, "description": d, "hours": float(h), "gt": gt, "ok": ok, "gold_tags": None}
                for t, d, h, gt, ok in tasks
            ],
        }
        for pid, _title, days, busy, tasks in HARD_MATCH_PROJECTS
    ]
    return {"basic": basic, "hard": hard}


def plan_sets() -> dict[str, list[dict]]:
    basic = [
        {
            "id": pid,
            "title": title,
            "goal": goal,
            "days": days,
            "ref": [{"text": f"{t}: {d}", "lesson": False} for t, d, _, _, _ in tasks],
        }
        for pid, title, goal, days, tasks in EVAL_PROJECTS
    ]
    hard = [
        {
            "id": pid,
            "title": title,
            "goal": goal,
            "days": days,
            "ref": [{"text": f"{t}: {d}", "lesson": lesson} for t, d, lesson in tasks],
        }
        for pid, title, goal, days, tasks in HARD_PLAN_PROJECTS
    ]
    return {"basic": basic, "hard": hard}


# ── 1. 담당자 추천 ──
def gini(values: list[float]) -> float:
    """배정 시간의 지니 계수. 0이면 모두 같고, 1에 가까울수록 한 사람에게 몰린다."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


TAG_SYSTEM = """너는 태스크에 필요한 스킬을 고르는 에이전트다.
각 태스크에 필요한 스킬을 "사내 스킬 목록"에 있는 이름 그대로 1~3개 고른다.
출력 형식: {"tags": [{"index": 0, "skills": ["..."]}]}"""


def llm_skill_tags(db, tasks: list[dict]) -> list[list[str]]:
    """실제 파이프라인에서 계획 에이전트가 하는 것처럼 LLM이 스킬 태그를 붙인다."""
    vocab = skill_vocabulary(db)
    user = f"사내 스킬 목록: {', '.join(vocab)}\n\n태스크:\n"
    user += "\n".join(f"[{i}] {t['title']}: {t['description']}" for i, t in enumerate(tasks))
    out = call_json("eval_tagger", TAG_SYSTEM, user, mock=lambda: {"tags": []})
    tags: list[list[str]] = [[] for _ in tasks]
    for item in out.get("tags", []):
        idx = item.get("index")
        if isinstance(idx, int) and 0 <= idx < len(tasks):
            tags[idx] = [sk for sk in item.get("skills", []) if sk in vocab]
    return tags


def _match_with_weights(db, ts, days, busy, weights, cap: bool) -> list[list[int]]:
    s = get_settings()
    saved = s.match_alpha, s.match_beta, s.match_gamma
    s.match_alpha, s.match_beta, s.match_gamma = weights
    try:
        res = match_tasks(db, ts, days=days, with_reasons=False, busy=busy, capacity_constraint=cap)
    finally:
        s.match_alpha, s.match_beta, s.match_gamma = saved
    return [[c["user_id"] for c in r["candidates"]] for r in res]


def eval_matching(db, projects: list[dict]) -> dict:
    """태그 조건 세 가지로 나눠 본다.

    - text: 태그 없이 제목과 설명만 준다.
    - llmtags: 실제 파이프라인처럼 LLM이 사내 스킬 목록에서 태그를 고른다.
    - tags: 연구자가 적은 태그를 그대로 준다(기본 세트만). 정답 담당자의 스킬 이름으로 적었으므로 상한으로만 본다.
    `_nocap`은 수용 시간 제약 없이 업무량 감점만 쓰는 버전이다.
    LLM 단독은 제목, 설명, 예상 시간과 팀원별 업무량·수용 시간을 받는다.
    """
    s = get_settings()
    default = (s.match_alpha, s.match_beta, s.match_gamma)
    has_gold = all(t["gold_tags"] is not None for p in projects for t in p["tasks"])
    # (가중치, 태그 종류, 수용 시간 제약). 스킬 유사도 단독은 제약 없이 순수 기준선으로 둔다.
    skill = (1.0, 0.0, 0.0)
    variants = {
        "hybrid_text": (default, "none", True),
        "skill_only_text": (skill, "none", False),
        "hybrid_llmtags": (default, "llm", True),
        "hybrid_llmtags_nocap": (default, "llm", False),
        "skill_only_llmtags": (skill, "llm", False),
    }
    if has_gold:
        variants |= {"hybrid_tags": (default, "gold", True), "skill_only_tags": (skill, "gold", False)}

    methods = [*variants, "llm_only"]
    rows = []
    per_project_assign: dict[str, list[tuple[dict, list[int]]]] = {m: [] for m in methods}
    for p in projects:
        base = [{"title": t["title"], "description": t["description"]} for t in p["tasks"]]
        tag_sets = {"none": [[] for _ in base], "llm": llm_skill_tags(db, base)}
        if has_gold:
            tag_sets["gold"] = [t["gold_tags"] for t in p["tasks"]]
        preds: dict[str, list[list[int]]] = {}
        for m, (weights, kind, cap) in variants.items():
            ts = [
                {**b, "required_skills": tg, "estimate_hours": t["hours"]}
                for b, tg, t in zip(base, tag_sets[kind], p["tasks"], strict=True)
            ]
            preds[m] = _match_with_weights(db, ts, p["days"], p["busy"], weights, cap)
        ts = [{**b, "estimate_hours": t["hours"]} for b, t in zip(base, p["tasks"], strict=True)]
        preds["llm_only"] = [r[:3] for r in match_tasks_llm_only(db, ts, days=p["days"], busy=p["busy"])]

        for m in methods:
            per_project_assign[m].append((p, [pr[0] if pr else -1 for pr in preds[m]]))
        for i, t in enumerate(p["tasks"]):
            rows.append(
                {
                    "project": p["id"],
                    "task": t["title"],
                    "gt": t["gt"],
                    "ok": t["ok"],
                    "llm_tags": tag_sets["llm"][i],
                    **{m: preds[m][i] for m in methods},
                }
            )

    cands = {c.user_id: c for c in load_candidates(db)}
    summary = {}
    for m in methods:
        n = len(rows)
        top1 = sum(1 for r in rows if r[m] and r[m][0] in {r["gt"], *r["ok"]})
        strict = sum(1 for r in rows if r[m] and r[m][0] == r["gt"])
        top3 = sum(1 for r in rows if {r["gt"], *r["ok"]} & set(r[m][:3]))
        # 편중도와 과부하는 프로젝트 단위로 계산한다.
        ginis, shares, overload = [], [], 0
        for p, firsts in per_project_assign[m]:
            weeks = max(p["days"] / 7, 1.0)
            assigned = dict.fromkeys(cands, 0.0)
            for uid, t in zip(firsts, p["tasks"], strict=True):
                if uid not in cands:
                    continue
                assigned[uid] += t["hours"]
                c = cands[uid]
                # 배정 후 (기존 업무 + busy + 이번 배정)이 기간 수용 시간을 넘으면 과부하 배정이다.
                if c.load_hours + p["busy"].get(uid, 0) + assigned[uid] > c.capacity * weeks:
                    overload += 1
            vals = list(assigned.values())
            ginis.append(gini(vals))
            shares.append(max(vals) / (sum(vals) or 1))
        summary[m] = {
            "n": n,
            "top1": round(top1 / n, 4),
            "top1_strict": round(strict / n, 4),
            "top3": round(top3 / n, 4),
            "invalid": sum(1 for r in rows if not r[m]),
            "gini_mean": round(statistics.mean(ginis), 4),
            "max_share_mean": round(statistics.mean(shares), 4),
            "overload_assignments": overload,
        }
    return {"summary": summary, "rows": rows}


def eval_matching_repeated(db, projects: list[dict], repeats: int) -> dict:
    """LLM이 끼는 방법(태그 부착, LLM 단독)은 실행마다 달라지므로 반복해서 평균과 표준편차를 낸다."""
    runs = [eval_matching(db, projects) for _ in range(repeats)]
    summary = {}
    for m, first in runs[0]["summary"].items():
        summary[m] = {"n": first["n"]}
        for k in ("top1", "top1_strict", "top3", "gini_mean", "max_share_mean", "overload_assignments"):
            vals = [r["summary"][m][k] for r in runs]
            summary[m][k] = {
                "mean": round(statistics.mean(vals), 4),
                "stdev": round(statistics.pstdev(vals), 4),
            }
    rows = [{"repeat": i, **row} for i, r in enumerate(runs) for row in r["rows"]]
    return {"summary": summary, "rows": rows}


# ── 2. 계획 커버리지 ──
JUDGE_SYSTEM = """너는 프로젝트 계획을 평가하는 심사자다.
기준 태스크 목록의 각 항목이 생성된 계획 안에 같은 작업으로 포함되어 있는지 판단한다.
이름이 달라도 하는 일이 같으면 포함된 것으로 본다. 다른 태스크의 설명 안에 그 작업이 명시되어 있어도 포함된 것으로 본다.
출력 형식: {"covered": [true, false, ...]}  (기준 태스크 순서대로)"""


def covered_by_embedding(ref: list[str], gen: list[str]) -> list[bool]:
    if not gen:
        return [False] * len(ref)
    rv, gv = np.asarray(embed_texts(ref)), np.asarray(embed_texts(gen))
    return [bool(x) for x in (rv @ gv.T).max(axis=1) >= COVERAGE_THRESHOLD]


def covered_by_judge(ref: list[str], gen: list[str]) -> list[bool]:
    user = "기준 태스크:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(ref))
    user += "\n\n생성된 계획:\n" + "\n".join(f"- {g}" for g in gen)
    out = call_json(
        "eval_judge", JUDGE_SYSTEM, user, mock=lambda: {"covered": covered_by_embedding(ref, gen)}
    )
    flags = [bool(f) for f in list(out.get("covered", []))[: len(ref)]]
    return flags + [False] * (len(ref) - len(flags))


def _ratio(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def eval_planning(db, projects: list[dict], repeats: int) -> dict:
    rows = []
    for p in projects:
        ref = [r["text"] for r in p["ref"]]
        lesson = [r["lesson"] for r in p["ref"]]
        for use_rag in (False, True):
            for rep in range(repeats):
                started = time.perf_counter()
                gen, hits = plan_tasks(db, p["title"], p["goal"], p["days"], Role.MANAGER, use_rag=use_rag)
                gen_text = [task_text(g) for g in gen]
                judge = covered_by_judge(ref, gen_text)
                emb = covered_by_embedding(ref, gen_text)
                rows.append(
                    {
                        "project": p["id"],
                        "use_rag": use_rag,
                        "repeat": rep,
                        "task_count": len(gen),
                        "coverage_judge": _ratio(judge),
                        "coverage_embedding": _ratio(emb),
                        "lesson_coverage_judge": _ratio(
                            [f for f, ls in zip(judge, lesson, strict=True) if ls]
                        ),
                        "general_coverage_judge": _ratio(
                            [f for f, ls in zip(judge, lesson, strict=True) if not ls]
                        ),
                        "seconds": round(time.perf_counter() - started, 2),
                        "sources": [h.title for h in hits],
                        "tasks": [g["title"] for g in gen],
                        "judge": judge,
                    }
                )

    def agg(key: str, use_rag: bool) -> dict | None:
        vals = [r[key] for r in rows if r["use_rag"] == use_rag and r[key] is not None]
        if not vals:
            return None
        return {"mean": round(statistics.mean(vals), 4), "stdev": round(statistics.pstdev(vals), 4)}

    keys = [
        "coverage_judge",
        "coverage_embedding",
        "lesson_coverage_judge",
        "general_coverage_judge",
        "task_count",
    ]
    summary = {("rag" if u else "no_rag"): {k: agg(k, u) for k in keys} for u in (False, True)}
    return {"summary": summary, "rows": rows, "threshold": COVERAGE_THRESHOLD}


# ── 3. 비용 ──
def eval_cost(db, since: datetime) -> dict:
    """이번 평가 실행 중에 기록된 LLM 호출만 집계한다."""
    q = (
        select(
            LLMCall.agent,
            func.count(),
            func.avg(LLMCall.latency_ms),
            func.percentile_cont(0.5).within_group(LLMCall.latency_ms),
            func.avg(LLMCall.input_tokens),
            func.avg(LLMCall.output_tokens),
        )
        .where(LLMCall.created_at >= since, LLMCall.model == model_name())
        .group_by(LLMCall.agent)
        .order_by(LLMCall.agent)
    )
    return {
        agent: {
            "calls": n,
            "latency_ms_mean": round(avg, 1),
            "latency_ms_median": round(med, 1),
            "input_tokens_mean": round(float(i or 0), 1),
            "output_tokens_mean": round(float(o or 0), 1),
        }
        for agent, n, avg, med, i, o in db.execute(q).all()
    }


# ── 출력 ──
def _ms(x: dict | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x['mean']:.{digits}f} ± {x['stdev']:.{digits}f}"


def to_markdown(res: dict) -> str:
    m = res["meta"]
    lines = [
        f"# 평가 결과 ({m['finished_at'][:19]} UTC)",
        "",
        f"- 모델: `{m['model']}`, 임베딩: `{m['embedding']}`",
        f"- 추천 가중치: α={m['weights'][0]}, β={m['weights'][1]}, γ={m['weights'][2]}",
        f"- 반복 횟수(추천·계획 공통): {m['repeats']}",
        "",
    ]
    for set_name, r in res.get("matching", {}).items():
        lines += [
            f"## 담당자 추천 ({set_name}, 태스크 {next(iter(r['summary'].values()))['n']}개)",
            "",
            f"반복 {m['repeats']}회의 평균 ± 표준편차다.",
            "",
            "| 방법 | Top-1 | Top-1(정답만) | Top-3 | 지니 계수 | 최대 점유율 | 과부하 배정(건) |",
            "|---|---|---|---|---|---|---|",
        ]
        for meth, s in r["summary"].items():
            cells = [_ms(s[k]) for k in ("top1", "top1_strict", "top3", "gini_mean", "max_share_mean")]
            lines.append(f"| {meth} | {' | '.join(cells)} | {_ms(s['overload_assignments'], 1)} |")
        lines.append("")
    for set_name, r in res.get("planning", {}).items():
        lines += [
            f"## 계획 커버리지 ({set_name})",
            "",
            "| 조건 | 전체(LLM 심사) | 교훈 태스크(LLM 심사) | 일반 태스크(LLM 심사) | 전체(임베딩) | 태스크 수 |",
            "|---|---|---|---|---|---|",
        ]
        for k, s in r["summary"].items():
            lines.append(
                f"| {k} | {_ms(s['coverage_judge'])} | {_ms(s['lesson_coverage_judge'])} | "
                f"{_ms(s['general_coverage_judge'])} | {_ms(s['coverage_embedding'])} | "
                f"{s['task_count']['mean']:.1f} |"
            )
        lines.append("")
    if res.get("cost"):
        lines += [
            "## 에이전트별 LLM 호출 비용",
            "",
            "| 에이전트 | 호출 | 평균 지연(ms) | 중앙값(ms) | 입력 토큰 | 출력 토큰 |",
            "|---|---|---|---|---|---|",
        ]
        for a, c in res["cost"].items():
            lines.append(
                f"| {a} | {c['calls']} | {c['latency_ms_mean']} | {c['latency_ms_median']} | "
                f"{c['input_tokens_mean']} | {c['output_tokens_mean']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT_DIR / "docs" / "report" / "results"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=["matching", "planning", "cost"])
    ap.add_argument("--sets", nargs="*", default=["basic", "hard"])
    args = ap.parse_args()

    s = get_settings()
    started = datetime.now(UTC)
    res: dict = {
        "meta": {
            "model": model_name(),
            "embedding": s.embedding_model,
            "repeats": args.repeats,
            "weights": [s.match_alpha, s.match_beta, s.match_gamma],
            "sets": args.sets,
            "started_at": started.isoformat(),
        }
    }
    with SessionLocal() as db:
        if "matching" in args.only:
            res["matching"] = {
                k: eval_matching_repeated(db, v, args.repeats)
                for k, v in match_sets().items()
                if k in args.sets
            }
        if "planning" in args.only:
            res["planning"] = {
                k: eval_planning(db, v, args.repeats) for k, v in plan_sets().items() if k in args.sets
            }
        if "cost" in args.only:
            res["cost"] = eval_cost(db, started)
    res["meta"]["finished_at"] = datetime.now(UTC).isoformat()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    (out / f"eval-{stamp}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    (out / f"eval-{stamp}.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
