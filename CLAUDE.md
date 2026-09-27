# CLAUDE.md

졸업작품 "Human-in-the-Loop 기반 다중 AI 에이전트를 활용한 지능형 협업 보조 시스템" 레포.

- 작업을 시작하기 전에 `docs/design.md`를 읽고 "4. 진행 현황"에서 끝나지 않은 항목부터 이어서 진행한다. 항목을 끝내면 체크한다.
- 이 레포는 public이다. 파일·커밋 메시지에 회사명, 사내 서버 이름·주소, 실명·학번 같은 개인 정보를 넣지 않는다. endpoint URL과 키는 `.env`에만 둔다.
- repo에는 코드베이스만 올린다. `docs/` 전체(설계 문서, 보고서, 평가 결과, 그림, 연구노트, 과제 원본)는 gitignore 대상이며 로컬에만 둔다.
- 커밋: Conventional Commits, 작성자는 git 설정 그대로, 커밋 메시지에 AI 표기(Co-Authored-By 등)를 넣지 않는다. push는 사용자 확인 후에 한다.
- 보고서에 들어가는 수치는 실제 실행 결과만 쓴다. "결론 및 소감"은 사용자 본인이 채운다.
- 결정이 필요한 부분은 사용자에게 질문하면서 진행한다.
- 백엔드: `backend/` (uv). 프론트: `frontend/` (React+Vite+TS). DB: docker compose (pgvector).
