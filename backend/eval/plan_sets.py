"""계획 평가 세트를 공통 형식으로 맞춘다.

기준 태스크: {text, lesson, gold_docs, keywords}
- gold_docs: 교훈 태스크의 근거가 적힌 문서 제목. 검색 재현율(Recall@k)과 정답 문서 제공 조건에 쓴다.
- keywords: LLM 심사와 별도로 쓰는 결정론적 판정 기준. 묶음 하나의 단어가 모두 계획 문장에 있으면 포함으로 본다.
"""

from app.seed.eval_set import EVAL_PROJECTS, HARD_PLAN_PROJECTS
from app.seed.heldout_set import HELDOUT_PLAN_PROJECTS

# 개발 세트(어려운 세트)의 교훈 태스크 근거 문서와 키워드. 세트 원본에는 lesson 표시만 있어서 여기서 보탠다.
DEV_LESSON_GOLD = {
    ("HP1", "환불 정책 법무 검토"): (["[회고] 쇼핑몰 정기구독 결제 도입"], [["법무"]]),
    ("HP1", "환불·부분 취소 QA"): (["QA 체크리스트"], [["부분", "취소"], ["환불", "QA"], ["환불", "검증"]]),
    ("HP1", "결제 코드 팀장 리뷰"): (["개발 표준 가이드"], [["리뷰"]]),
    ("HP2", "스토어 심사 기간 확보"): (["[회고] 모바일 앱 1.0 출시"], [["심사"]]),
    ("HP3", "마이그레이션 리허설"): (
        ["[회고] 소셜 로그인·회원 체계 개편"],
        [["리허설"], ["마이그레이션", "테스트"]],
    ),
    ("HP3", "개인정보 처리방침 개정"): (
        ["[회고] 소셜 로그인·회원 체계 개편", "개인정보 보호 규정"],
        [["처리방침"], ["개인정보"]],
    ),
    ("HP4", "발송 빈도 제한"): (["[회고] 고객 CRM 마케팅 자동화"], [["빈도"], ["발송", "횟수"], ["피로도"]]),
    ("HP5", "2주 이상 A/B 테스트"): (["[회고] 상품 추천 시스템 고도화"], [["A/B"], ["AB"], ["실험"]]),
    ("HP6", "지표 정의 합의"): (["[회고] 사내 관리자 대시보드 구축"], [["지표", "합의"], ["지표", "정의"]]),
    ("HP6", "배포 후 모니터링"): (["배포 프로세스"], [["모니터링"], ["오전"]]),
}


def _ref(title: str, desc: str, lesson: bool, gold: list[str], kw: list[list[str]]) -> dict:
    return {"title": title, "text": f"{title}: {desc}", "lesson": lesson, "gold_docs": gold, "keywords": kw}


def plan_sets() -> dict[str, list[dict]]:
    basic = [
        {
            "id": pid,
            "title": title,
            "goal": goal,
            "days": days,
            "ref": [_ref(t, d, False, [], []) for t, d, _, _, _ in tasks],
        }
        for pid, title, goal, days, tasks in EVAL_PROJECTS
    ]
    hard = [
        {
            "id": pid,
            "title": title,
            "goal": goal,
            "days": days,
            "ref": [_ref(t, d, ls, *DEV_LESSON_GOLD.get((pid, t), ([], []))) for t, d, ls in tasks],
        }
        for pid, title, goal, days, tasks in HARD_PLAN_PROJECTS
    ]
    heldout = [
        {
            "id": pid,
            "title": title,
            "goal": goal,
            "days": days,
            "ref": [_ref(t, d, ls, gold, kw) for t, d, ls, gold, kw in tasks],
        }
        for pid, title, goal, days, tasks in HELDOUT_PLAN_PROJECTS
    ]
    return {"basic": basic, "hard": hard, "heldout": heldout}


def keyword_hit(groups: list[list[str]], plan_text: str) -> bool:
    """키워드 묶음 중 하나라도 모든 단어가 계획 문장에 있으면 True. 띄어쓰기 차이는 무시한다."""
    text = plan_text.replace(" ", "").lower()
    return any(all(w.replace(" ", "").lower() in text for w in g) for g in groups)


# 개발용 관리자 피드백(설정 선택용). 검증용 피드백은 app/seed/heldout_set.py의 HELDOUT_FEEDBACK에 있다.
# (프로젝트 id, 종류, 요청 문장, 대상 기준 태스크 제목 또는 태스크 수 상한)
DEV_FEEDBACK = [
    ("HP1", "add", "환불 정책 법무 검토를 태스크로 넣어 주세요.", "환불 정책 법무 검토"),
    ("HP2", "remove", "출시 알림 캠페인은 마케팅팀이 따로 하니 빼 주세요.", "출시 알림 캠페인"),
    ("HP3", "add", "마이그레이션 리허설을 넣어 주세요.", "마이그레이션 리허설"),
    ("HP4", "limit", "태스크를 4개 이내로 줄여 주세요.", "4"),
    ("HP5", "add", "A/B 테스트를 2주 이상 하는 태스크를 넣어 주세요.", "2주 이상 A/B 테스트"),
    ("HP6", "remove", "배포 후 모니터링은 운영팀이 하니 빼 주세요.", "배포 후 모니터링"),
    ("E1", "limit", "태스크를 5개 이내로 줄여 주세요.", "5"),
    (
        "E2",
        "remove",
        "권한은 기존 관리자 권한을 그대로 쓰니 접근 권한 작업은 빼 주세요.",
        "경영진 전용 접근 권한",
    ),
]
