# HITL Task Copilot

Human-in-the-Loop 기반 다중 AI 에이전트를 활용한 지능형 협업 보조 시스템.

관리자가 목표와 마감일을 입력하면 계획 에이전트가 과거 프로젝트를 검색해 태스크 초안을 만들고, 리소스 에이전트가 담당자를 추천한다. 관리자가 초안을 검토·수정·승인하면 실무자가 산출물을 제출하고, 취합 에이전트가 진척도를 계산해 경영진 브리핑을 만든다.

설계와 진행 현황은 [docs/design.md](docs/design.md)에 있다.

## 실행

```bash
cp .env.example .env        # LLM endpoint와 키 입력
docker compose up -d db
cd backend && uv sync
```
