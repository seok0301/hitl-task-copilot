import logging
import uuid
from datetime import date

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from langgraph.types import Command
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agents.graph import get_graph, pending_review, thread_config
from app.auth import current_user, require_roles
from app.db import SessionLocal, get_db
from app.models import Project, ProjectStatus, Role, Task, User
from app.schemas import ProjectDetail, ProjectIn, ProjectOut, ReviewIn, TaskOut

router = APIRouter(prefix="/projects", tags=["projects"])
log = logging.getLogger(__name__)
manager_only = require_roles(Role.MANAGER)


def _run_graph(project_id: int, payload) -> None:
    """그래프를 다음 interrupt(또는 끝)까지 실행한다. 실패하면 오류를 프로젝트에 남긴다."""
    with SessionLocal() as db:
        p = db.get(Project, project_id)
        p.planning_error = None
        p.status = ProjectStatus.PLANNING
        db.commit()
        thread_id = p.graph_thread_id
    try:
        get_graph().invoke(payload, thread_config(thread_id))
    except Exception as e:
        log.exception("planning failed")
        with SessionLocal() as db:
            p = db.get(Project, project_id)
            p.planning_error = f"{type(e).__name__}: {e}"[:1000]
            p.status = ProjectStatus.REVIEW if pending_review(thread_id) else ProjectStatus.PLANNING
            db.commit()


def task_out(t: Task) -> TaskOut:
    out = TaskOut.model_validate(t)
    out.assignee_name = t.assignee.name if t.assignee else None
    return out


def _visible(db: Session, user: User):
    q = select(Project).order_by(Project.id.desc())
    if user.role == Role.EMPLOYEE:
        q = q.where(Project.tasks.any(Task.assignee_id == user.id))
    return q


@router.post("", response_model=ProjectOut)
def create_project(
    body: ProjectIn, bg: BackgroundTasks, db: Session = Depends(get_db), user: User = Depends(manager_only)
):
    p = Project(
        title=body.title,
        goal=body.goal,
        deadline=body.deadline,
        owner_id=user.id,
        use_rag=body.use_rag,
        graph_thread_id=str(uuid.uuid4()),
    )
    db.add(p)
    db.commit()
    days = max((body.deadline - date.today()).days, 1)
    bg.add_task(
        _run_graph,
        p.id,
        {"project_id": p.id, "title": p.title, "goal": p.goal, "days": days, "use_rag": body.use_rag},
    )
    return p


@router.get("", response_model=list[ProjectOut])
def list_projects(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return db.scalars(_visible(db, user)).all()


@router.get("/{project_id}", response_model=ProjectDetail)
def get_project(project_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    p = db.scalar(
        _visible(db, user)
        .where(Project.id == project_id)
        .options(selectinload(Project.tasks).selectinload(Task.assignee))
    )
    if p is None:
        raise HTTPException(404, "프로젝트를 찾을 수 없습니다.")
    detail = ProjectDetail.model_validate(p)
    detail.tasks = [task_out(t) for t in p.tasks]
    return detail


@router.get("/{project_id}/draft")
def get_draft(project_id: int, db: Session = Depends(get_db), _: User = Depends(manager_only)):
    p = db.get(Project, project_id)
    if p is None:
        raise HTTPException(404, "프로젝트를 찾을 수 없습니다.")
    return {"status": p.status, "error": p.planning_error, "draft": pending_review(p.graph_thread_id)}


@router.post("/{project_id}/review", response_model=ProjectOut)
def review(
    project_id: int,
    body: ReviewIn,
    bg: BackgroundTasks,
    db: Session = Depends(get_db),
    user: User = Depends(manager_only),
):
    p = db.get(Project, project_id)
    if p is None or pending_review(p.graph_thread_id) is None:
        raise HTTPException(409, "승인 대기 중인 초안이 없습니다.")
    if body.action not in ("approve", "regenerate"):
        raise HTTPException(400, "action은 approve 또는 regenerate입니다.")
    if body.action == "approve" and not body.tasks:
        raise HTTPException(400, "승인할 태스크가 없습니다.")
    decision = {
        "action": body.action,
        "feedback": body.feedback,
        "reviewer_id": user.id,
        "tasks": [t.model_dump(mode="json") for t in body.tasks],
    }
    if body.action == "approve":
        _run_graph(p.id, Command(resume=decision))
    else:
        bg.add_task(_run_graph, p.id, Command(resume=decision))
    db.refresh(p)
    return p
