from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    database_url: str = "postgresql+psycopg://hitl:hitl@localhost:5442/hitl"

    llm_provider: str = "mock"  # litellm | gemini | mock
    llm_model: str = "gemma-4-26b"
    litellm_base_url: str = "http://localhost:4000/v1"
    litellm_api_key: str = ""
    gemini_api_key: str = ""
    # Gemini는 유료 API라서 ALLOW_PAID_API=true로 명시해야만 쓸 수 있다.
    allow_paid_api: bool = False
    llm_temperature: float = 0.2

    embedding_provider: str = "fastembed"  # fastembed | mock
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    embedding_cache_dir: str = str(ROOT_DIR / ".cache" / "fastembed")

    jwt_secret: str = "change-me"
    jwt_expire_minutes: int = 60 * 12

    # 계획 에이전트의 사내 문서 검색 방식: chunk | lexical | draft (app/agents/planner.py의 retrieve 참고)
    rag_mode: str = "lexical"
    # 관리자가 수정을 요청했을 때의 처리: edit | rewrite | regenerate (app/agents/planner.py의 plan_tasks 참고)
    plan_revision: str = "edit"
    # 참고 자료의 교훈을 먼저 골라 태스크로 옮기고 태스크마다 근거를 남기는 프롬프트를 쓴다.
    plan_extract_lessons: bool = True
    # 계획 방식: single(참고 자료와 함께 한 번에 계획) | augment(기본 계획 후 교훈을 추가·수정 명령으로 더함)
    plan_strategy: str = "augment"

    # 담당자 추천 방식: llm_rank(LLM 순위 + 수용 시간 규칙, 점수 방식은 예비) | score(점수 + 규칙)
    match_strategy: str = "llm_rank"
    # 담당자 추천 가중치: score = α·스킬 유사도 + β·이력 점수 − γ·업무량
    match_alpha: float = 0.5
    match_beta: float = 0.3
    match_gamma: float = 0.2
    # 배정하면 기간 수용 시간을 넘는 후보를 뒤로 미룬다.
    match_capacity_constraint: bool = True
    # 수용 시간 제약을 쓸 때, 대체자를 필요 스킬 보유자 → 같은 팀 → 나머지 순서로 찾는다.
    match_role_filter: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
