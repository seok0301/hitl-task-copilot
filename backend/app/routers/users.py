from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import require_roles
from app.db import get_db
from app.models import Role, Task, TaskStatus, User
from app.schemas import MemberOut, UserOut

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=list[MemberOut])
def members(db: Session = Depends(get_db), _: User = Depends(require_roles(Role.MANAGER, Role.EXECUTIVE))):
    loads = dict(
        db.execute(
            select(Task.assignee_id, func.sum(Task.estimate_hours))
            .where(Task.status.in_([TaskStatus.TODO, TaskStatus.IN_PROGRESS]))
            .group_by(Task.assignee_id)
        ).all()
    )
    return [
        MemberOut(
            **UserOut.model_validate(u).model_dump(),
            load_hours=float(loads.get(u.id, 0)),
            capacity_hours=u.weekly_capacity_hours,
        )
        for u in db.scalars(select(User).order_by(User.id))
    ]
