# HITL Task Copilot

Human-in-the-Loop 기반 다중 AI 에이전트를 활용한 지능형 협업 보조 시스템.

관리자가 목표와 마감일을 입력하면 계획 에이전트가 과거 프로젝트를 검색해 태스크 초안을 만들고, 리소스 에이전트가 담당자를 추천한다. 관리자가 초안을 검토·수정·승인하면 실무자가 산출물을 제출하고, 취합 에이전트가 진척도를 계산해 경영진 브리핑을 만든다.

## 구성

| 영역 | 기술 |
|---|---|
| 백엔드 | FastAPI, SQLAlchemy, LangGraph (interrupt 기반 승인 대기, Postgres 체크포인터) |
| DB | PostgreSQL + pgvector |
| LLM | OpenAI 호환 endpoint(LiteLLM + Gemma), Gemini, mock 중 선택 |
| 임베딩 | fastembed 다국어 모델 (CPU) |
| 프론트엔드 | React, Vite, TypeScript |

## 실행

```bash
cp .env.example .env              # LLM endpoint와 키 입력 (키 없이 해 보려면 LLM_PROVIDER=mock)
docker compose up -d db

cd backend
uv sync
uv run python -m app.seed.seed    # 가상 회사 데이터 적재 (기존 데이터 삭제)
uv run uvicorn app.main:app --port 8200

cd ../frontend
npm install
npm run dev                       # http://localhost:5190
```

데모 계정의 비밀번호는 모두 `password`다. 관리자 `dev.lead@example.com`, 실무자 `seoyeon@example.com`, 경영진 `ceo@example.com`.

## 테스트와 평가

```bash
cd backend
uv run pytest                     # hitl_test DB와 mock LLM으로 전체 사이클 확인
uv run python -m eval.run_eval    # 평가 결과를 docs/report/results/에 저장 (git에 올리지 않음)
```
