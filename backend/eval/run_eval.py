"""평가 스크립트. 실행: `uv run python -m eval.run_eval [--out DIR] [--repeats N]`

1. 담당자 추천: 하이브리드 점수 / 스킬 유사도 단독 / LLM 단독의 Top-1·Top-3 정확도와 업무 편중도
2. 계획 에이전트: RAG 사용 여부에 따른 기준 태스크 커버리지
3. HITL: approval_logs에 쌓인 관리자 수정률
4. 비용: 에이전트별 LLM 호출 소요 시간과 토큰
결과는 JSON과 마크다운 요약으로 저장한다.
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sqlalchemy import func, select

from app.agents.planner import plan_tasks
from app.agents.resource import load_candidates, match_tasks, match_tasks_llm_only, task_text
from app.config import ROOT_DIR, get_settings
from app.db import SessionLocal
from app.llm.embeddings import embed_texts
from app.llm.provider import call_json, model_name
from app.models import ApprovalLog, LLMCall, Role
from app.seed.eval_set import EVAL_PROJECTS

COVERAGE_THRESHOLD = 0.6  # 임베딩 기반 커버리지에서 "같은 태스크"로 볼 최소 코사인 유사도


def gini(values: list[float]) -> float:
    """배정 시간의 지니 계수. 0이면 모두 같고, 1에 가까울수록 한 사람에게 몰린다."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def workload_stats(assign: list[int], hours: list[float], pool: list[int]) -> dict:
    per = dict.fromkeys(pool, 0.0)
    for uid, h in zip(assign, hours, strict=True):
        if uid in per:
            per[uid] += h
    vals = list(per.values())
    total = sum(vals) or 1
    return {
        "gini": round(gini(vals), 4),
        "max_share": round(max(vals) / total, 4),
        "people_used": sum(1 for v in vals if v > 0),
    }


def eval_matching(db) -> dict:
    pool = [c.user_id for c in load_candidates(db)]
    methods = {"hybrid": [], "skill_only": [], "llm_only": []}
    s = get_settings()
    rows = []
    for pid, title, _goal, days, tasks in EVAL_PROJECTS:
        ts = [
            {"title": t, "description": d, "required_skills": sk, "estimate_hours": 16}
            for t, d, sk, _, _ in tasks
        ]
        hybrid = [r["candidates"] for r in match_tasks(db, ts, days=days, with_reasons=False)]

        # 스킬 유사도 단독: α=1, β=γ=0
        a, b, g = s.match_alpha, s.match_beta, s.match_gamma
        s.match_alpha, s.match_beta, s.match_gamma = 1.0, 0.0, 0.0
        try:
            skill_only = [r["candidates"] for r in match_tasks(db, ts, days=days, with_reasons=False)]
        finally:
            s.match_alpha, s.match_beta, s.match_gamma = a, b, g

        llm_only = match_tasks_llm_only(db, ts)
        for i, (t_title, _, _, gt, ok) in enumerate(tasks):
            acc = {gt, *ok}
            preds = {
                "hybrid": [c["user_id"] for c in hybrid[i]],
                "skill_only": [c["user_id"] for c in skill_only[i]],
                "llm_only": llm_only[i][:3],
            }
            row = {"project": pid, "task": t_title, "gt": gt, "ok": ok}
            for m, p in preds.items():
                methods[m].append((p, acc, 16.0))
                row[m] = p
            rows.append(row)

    summary = {}
    for m, items in methods.items():
        n = len(items)
        top1 = sum(1 for p, acc, _ in items if p and p[0] in acc)
        top3 = sum(1 for p, acc, _ in items if acc & set(p[:3]))
        strict1 = sum(1 for (p, _, _), r in zip(items, rows, strict=True) if p and p[0] == r["gt"])
        firsts = [p[0] if p else -1 for p, _, _ in items]
        summary[m] = {
            "n": n,
            "top1": round(top1 / n, 4),
            "top3": round(top3 / n, 4),
            "top1_strict": round(strict1 / n, 4),
            "invalid": sum(1 for p, _, _ in items if not p),
            **workload_stats(firsts, [h for _, _, h in items], pool),
        }
    return {"summary": summary, "rows": rows}


JUDGE_SYSTEM = """너는 프로젝트 계획을 평가하는 심사자다.
기준 태스크 목록의 각 항목이 생성된 계획 안에 같은 작업으로 포함되어 있는지 판단한다.
이름이 달라도 하는 일이 같으면 포함된 것으로 본다.
출력 형식: {"covered": [true, false, ...]}  (기준 태스크 순서대로)"""


def coverage_by_embedding(ref: list[str], gen: list[str]) -> float:
    if not gen:
        return 0.0
    rv, gv = np.asarray(embed_texts(ref)), np.asarray(embed_texts(gen))
    sims = rv @ gv.T
    return float((sims.max(axis=1) >= COVERAGE_THRESHOLD).mean())


def coverage_by_judge(ref: list[str], gen: list[str]) -> float:
    user = "기준 태스크:\n" + "\n".join(f"{i}. {r}" for i, r in enumerate(ref))
    user += "\n\n생성된 계획:\n" + "\n".join(f"- {g}" for g in gen)

    def mock():
        rv, gv = np.asarray(embed_texts(ref)), np.asarray(embed_texts(gen or ["-"]))
        return {"covered": [bool(x) for x in (rv @ gv.T).max(axis=1) >= COVERAGE_THRESHOLD]}

    out = call_json("eval_judge", JUDGE_SYSTEM, user, mock=mock)
    flags = list(out.get("covered", []))[: len(ref)]
    flags += [False] * (len(ref) - len(flags))
    return sum(bool(f) for f in flags) / len(ref)


def eval_rag(db, repeats: int) -> dict:
    rows = []
    for pid, title, goal, days, tasks in EVAL_PROJECTS:
        ref = [f"{t}: {d}" for t, d, _, _, _ in tasks]
        for use_rag in (False, True):
            for r in range(repeats):
                started = time.perf_counter()
                gen, hits = plan_tasks(db, title, goal, days, Role.MANAGER, use_rag=use_rag)
                gen_text = [task_text(g) for g in gen]
                rows.append(
                    {
                        "project": pid,
                        "use_rag": use_rag,
                        "repeat": r,
                        "task_count": len(gen),
                        "coverage_embedding": round(coverage_by_embedding(ref, gen_text), 4),
                        "coverage_judge": round(coverage_by_judge(ref, gen_text), 4),
                        "seconds": round(time.perf_counter() - started, 2),
                        "sources": [h.title for h in hits],
                        "tasks": [g["title"] for g in gen],
                    }
                )

    def agg(key: str, use_rag: bool) -> dict:
        vals = [row[key] for row in rows if row["use_rag"] == use_rag]
        return {"mean": round(statistics.mean(vals), 4), "stdev": round(statistics.pstdev(vals), 4)}

    summary = {
        ("rag" if u else "no_rag"): {
            "coverage_embedding": agg("coverage_embedding", u),
            "coverage_judge": agg("coverage_judge", u),
            "task_count": agg("task_count", u),
        }
        for u in (False, True)
    }
    return {"summary": summary, "rows": rows, "threshold": COVERAGE_THRESHOLD}


def eval_hitl(db) -> dict:
    logs = db.scalars(select(ApprovalLog).order_by(ApprovalLog.id)).all()
    approvals = [lg for lg in logs if lg.action == "approve" and lg.edit_stats]
    regen = [lg for lg in logs if lg.action == "regenerate"]
    if not approvals:
        return {"approvals": 0, "regenerations": len(regen)}
    es = [lg.edit_stats for lg in approvals]

    def mean(k):
        return round(statistics.mean(e[k] for e in es), 4)

    return {
        "approvals": len(approvals),
        "regenerations": len(regen),
        "edit_rate": mean("edit_rate"),
        "assignee_change_rate": mean("assignee_change_rate"),
        "unchanged_ratio": round(sum(e["unchanged"] for e in es) / sum(e["draft_count"] for e in es), 4),
        "added_total": sum(e["added"] for e in es),
        "deleted_total": sum(e["deleted"] for e in es),
        "field_changes_total": {k: sum(e["field_changes"][k] for e in es) for k in es[0]["field_changes"]},
        "review_seconds_median": round(statistics.median(lg.review_seconds or 0 for lg in approvals), 1),
        "per_project": [
            {"project_id": lg.project_id, **lg.edit_stats, "review_seconds": lg.review_seconds}
            for lg in approvals
        ],
    }


def eval_cost(db, since: datetime | None = None) -> dict:
    q = (
        select(
            LLMCall.agent,
            func.count(),
            func.avg(LLMCall.latency_ms),
            func.percentile_cont(0.5).within_group(LLMCall.latency_ms),
            func.avg(LLMCall.input_tokens),
            func.avg(LLMCall.output_tokens),
        )
        .group_by(LLMCall.agent)
        .order_by(LLMCall.agent)
    )
    if since:
        q = q.where(LLMCall.created_at >= since)
    q = q.where(LLMCall.model != "mock") if model_name() != "mock" else q
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


def to_markdown(res: dict) -> str:
    lines = [
        f"# 평가 결과 ({res['meta']['finished_at'][:19]})",
        "",
        f"- 모델: `{res['meta']['model']}`, 임베딩: `{res['meta']['embedding']}`",
        f"- 추천 가중치: α={res['meta']['weights'][0]}, β={res['meta']['weights'][1]}, γ={res['meta']['weights'][2]}",
        f"- 계획 반복 횟수: {res['meta']['repeats']}",
        "",
    ]
    if "matching" in res:
        lines += [
            "## 1. 담당자 추천",
            "",
            "| 방법 | Top-1 | Top-1(정답만) | Top-3 | 지니 계수 | 최대 점유율 | 배정 인원 |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, s in res["matching"]["summary"].items():
            lines.append(
                f"| {m} | {s['top1']:.3f} | {s['top1_strict']:.3f} | {s['top3']:.3f} | {s['gini']:.3f} | "
                f"{s['max_share']:.3f} | {s['people_used']} |"
            )
        lines.append("")
    if "rag" in res:
        lines += [
            "## 2. RAG 유무에 따른 계획 커버리지",
            "",
            "| 조건 | 커버리지(LLM 심사) | 커버리지(임베딩) | 태스크 수 |",
            "|---|---|---|---|",
        ]
        for k, s in res["rag"]["summary"].items():
            lines.append(
                f"| {k} | {s['coverage_judge']['mean']:.3f} ± {s['coverage_judge']['stdev']:.3f} | "
                f"{s['coverage_embedding']['mean']:.3f} ± {s['coverage_embedding']['stdev']:.3f} | "
                f"{s['task_count']['mean']:.1f} |"
            )
        lines.append("")
    if "hitl" in res:
        h = res["hitl"]
        lines += ["## 3. HITL 관리자 수정률", ""]
        if h.get("approvals"):
            lines += [
                f"- 승인 {h['approvals']}회, 재생성 요청 {h['regenerations']}회",
                f"- 평균 수정률 {h['edit_rate']:.3f}, 평균 담당자 변경률 {h['assignee_change_rate']:.3f}",
                f"- 그대로 승인된 초안 태스크 비율 {h['unchanged_ratio']:.3f}",
                f"- 검토 시간 중앙값 {h['review_seconds_median']}초",
                "",
            ]
        else:
            lines += ["- 아직 승인 기록이 없다.", ""]
    if "cost" in res:
        lines += [
            "## 4. 에이전트별 LLM 호출 비용",
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
    ap.add_argument("--only", nargs="*", default=["matching", "rag", "hitl", "cost"])
    args = ap.parse_args()

    s = get_settings()
    started = datetime.now(UTC)
    res: dict = {
        "meta": {
            "model": model_name(),
            "embedding": s.embedding_model,
            "repeats": args.repeats,
            "weights": [s.match_alpha, s.match_beta, s.match_gamma],
            "started_at": started.isoformat(),
        }
    }
    with SessionLocal() as db:
        if "matching" in args.only:
            res["matching"] = eval_matching(db)
        if "rag" in args.only:
            res["rag"] = eval_rag(db, args.repeats)
        if "hitl" in args.only:
            res["hitl"] = eval_hitl(db)
        if "cost" in args.only:
            res["cost"] = eval_cost(db)
    res["meta"]["finished_at"] = datetime.now(UTC).isoformat()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    (out / f"eval-{stamp}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    (out / f"eval-{stamp}.md").write_text(to_markdown(res))
    print(to_markdown(res))


if __name__ == "__main__":
    main()
