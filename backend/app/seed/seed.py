"""가상 회사 데이터를 DB에 적재한다. 실행: `uv run python -m app.seed.seed`"""

from datetime import date, timedelta

from app.auth import hash_password
from app.db import SessionLocal, init_db
from app.llm.embeddings import embed_texts
from app.models import (
    Document,
    PastProject,
    PastTask,
    Project,
    ProjectStatus,
    Role,
    Task,
    TaskStatus,
    User,
)
from app.rag.ingest import ingest_document
from app.seed.company import DOCUMENTS, ONGOING, PAST_PROJECTS, USERS

PASSWORD = "password"


def skill_text(title: str, team: str, skills: list[str]) -> str:
    return f"{team}팀 {title}. 전문 분야: {', '.join(skills)}"


def retrospective(title: str, goal: str, year: int, tasks, lessons: str, names: dict[int, str]) -> str:
    lines = [f"{year}년 프로젝트 회고: {title}", f"목표: {goal}", "수행한 태스크:"]
    for t_title, desc, uid, hours in tasks:
        lines.append(f"- {t_title} (담당 {names[uid]}, {hours}시간): {desc}")
    lines.append(f"교훈: {lessons}")
    return "\n".join(lines)


def run() -> None:
    init_db(drop=True)
    pw = hash_password(PASSWORD)
    with SessionLocal() as db:
        vecs = embed_texts([skill_text(u[5], u[4], u[6]) for u in USERS])
        for (uid, name, slug, role, team, title, skills), vec in zip(USERS, vecs, strict=True):
            db.add(
                User(
                    id=uid,
                    name=name,
                    email=f"{slug}@example.com",
                    pw_hash=pw,
                    role=Role(role),
                    team=team,
                    title=title,
                    skills=skills,
                    skill_embedding=vec,
                )
            )
        db.flush()
        names = {u[0]: u[1] for u in USERS}

        for pid, year, title, goal, tasks, lessons in PAST_PROJECTS:
            db.add(PastProject(id=pid, title=title, goal=goal, year=year))
            db.flush()
            tvecs = embed_texts([f"{t[0]}. {t[1]}" for t in tasks])
            for (t_title, desc, uid, hours), vec in zip(tasks, tvecs, strict=True):
                db.add(
                    PastTask(
                        past_project_id=pid,
                        title=t_title,
                        description=desc,
                        assignee_id=uid,
                        hours=hours,
                        embedding=vec,
                    )
                )
            doc = Document(
                title=f"[회고] {title}",
                access_level=1,
                past_project_id=pid,
                content=retrospective(title, goal, year, tasks, lessons, names),
            )
            db.add(doc)
            ingest_document(db, doc)

        for title, level, content in DOCUMENTS:
            doc = Document(title=title, access_level=level, content=content)
            db.add(doc)
            ingest_document(db, doc)

        o_title, o_tasks = ONGOING
        proj = Project(
            title=o_title,
            goal="운영 중인 서비스의 품질과 비용을 개선한다.",
            deadline=date.today() + timedelta(days=45),
            status=ProjectStatus.ACTIVE,
            owner_id=3,
        )
        db.add(proj)
        db.flush()
        for t_title, uid, hours in o_tasks:
            db.add(
                Task(
                    project_id=proj.id,
                    title=t_title,
                    assignee_id=uid,
                    estimate_hours=hours,
                    status=TaskStatus.IN_PROGRESS,
                )
            )

        db.commit()
    # 명시적으로 넣은 id 뒤로 시퀀스를 맞춘다.
    from sqlalchemy import text

    with SessionLocal() as db:
        for table in ("users", "past_projects"):
            db.execute(text(f"SELECT setval('{table}_id_seq', (SELECT MAX(id) FROM {table}))"))
        db.commit()
    print(
        f"seed 완료: 사용자 {len(USERS)}명, 과거 프로젝트 {len(PAST_PROJECTS)}개, 문서 {len(PAST_PROJECTS) + len(DOCUMENTS)}개"
    )


if __name__ == "__main__":
    run()
