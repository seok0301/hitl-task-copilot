from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from app.agents.reporter import review_submission
from app.auth import current_user, require_roles
from app.db import get_db
from app.models import Project, Role, Submission, Task, TaskStatus, User
from app.routers.projects import task_out
from app.schemas import StatusIn, SubmissionIn, SubmissionOut, TaskOut

router = APIRouter(prefix="/tasks", tags=["tasks"])


def _task(db: Session, task_id: int) -> Task:
    t = db.get(Task, task_id)
    if t is None:
        raise HTTPException(404, "태스크를 찾을 수 없습니다.")
    return t


@router.get("/mine", response_model=list[TaskOut])
def my_tasks(db: Session = Depends(get_db), user: User = Depends(current_user)):
    tasks = db.scalars(
        select(Task)
        .where(Task.assignee_id == user.id)
        .options(selectinload(Task.assignee))
        .order_by(Task.deadline.nulls_last(), Task.id)
    ).all()
    return [task_out(t) for t in tasks]


@router.patch("/{task_id}/status", response_model=TaskOut)
def set_status(
    task_id: int, body: StatusIn, db: Session = Depends(get_db), user: User = Depends(current_user)
):
    t = _task(db, task_id)
    if user.role == Role.EMPLOYEE:
        # 실무자는 자기 태스크를 시작하는 것만 할 수 있다. 완료 처리는 관리자가 한다.
        if t.assignee_id != user.id or body.status not in (TaskStatus.TODO, TaskStatus.IN_PROGRESS):
            raise HTTPException(403, "권한이 없습니다.")
    elif user.role != Role.MANAGER:
        raise HTTPException(403, "권한이 없습니다.")
    t.status = body.status
    db.commit()
    return task_out(t)


@router.get("/{task_id}/submissions", response_model=list[SubmissionOut])
def list_submissions(task_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    t = _task(db, task_id)
    if user.role == Role.EMPLOYEE and t.assignee_id != user.id:
        raise HTTPException(403, "권한이 없습니다.")
    return t.submissions


@router.post("/{task_id}/submissions", response_model=SubmissionOut)
def submit(
    task_id: int, body: SubmissionIn, db: Session = Depends(get_db), user: User = Depends(current_user)
):
    t = _task(db, task_id)
    if t.assignee_id != user.id:
        raise HTTPException(403, "본인에게 배정된 태스크만 제출할 수 있습니다.")
    s = Submission(
        task_id=t.id, user_id=user.id, content=body.content, ai_review=review_submission(t, body.content)
    )
    db.add(s)
    t.status = TaskStatus.SUBMITTED
    db.commit()
    return s


@router.post("/{task_id}/accept", response_model=TaskOut)
def accept(task_id: int, db: Session = Depends(get_db), user: User = Depends(require_roles(Role.MANAGER))):
    t = _task(db, task_id)
    if db.get(Project, t.project_id).owner_id != user.id:
        raise HTTPException(403, "담당 프로젝트가 아닙니다.")
    t.status = TaskStatus.DONE
    db.commit()
    return task_out(t)
