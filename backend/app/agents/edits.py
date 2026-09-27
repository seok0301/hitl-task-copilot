"""AI 초안과 관리자 최종본을 비교해 HITL 수정 통계를 낸다."""

FIELDS = ("title", "description", "estimate_hours", "assignee_id")


def edit_stats(draft: list[dict], final: list[dict]) -> dict:
    """초안 태스크는 `draft_index`로 최종본과 짝을 짓는다. 짝이 없는 최종본 태스크는 관리자가 추가한 것이다."""
    by_index = {f["draft_index"]: f for f in final if f.get("draft_index") is not None}
    added = sum(1 for f in final if f.get("draft_index") is None)
    deleted = 0
    modified = 0
    assignee_changed = 0
    field_changes = dict.fromkeys(FIELDS, 0)
    for i, d in enumerate(draft):
        f = by_index.get(i)
        if f is None:
            deleted += 1
            continue
        changed = [k for k in FIELDS if _norm(d.get(k)) != _norm(f.get(k))]
        for k in changed:
            field_changes[k] += 1
        if changed:
            modified += 1
        if "assignee_id" in changed:
            assignee_changed += 1
    kept = len(draft) - deleted
    denom = len(draft) + added
    return {
        "draft_count": len(draft),
        "final_count": len(final),
        "added": added,
        "deleted": deleted,
        "modified": modified,
        "unchanged": kept - modified,
        "field_changes": field_changes,
        "assignee_changed": assignee_changed,
        # 수정률: 초안에서 손이 간 태스크(수정·삭제·추가)의 비율
        "edit_rate": round((modified + deleted + added) / denom, 4) if denom else 0.0,
        # 담당자 변경률: 남아 있는 초안 태스크 중 담당자가 바뀐 비율
        "assignee_change_rate": round(assignee_changed / kept, 4) if kept else 0.0,
    }


def _norm(v):
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, int | float):
        return float(v)
    return v
