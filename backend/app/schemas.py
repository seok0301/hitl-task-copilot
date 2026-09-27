from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from app.models import ProjectStatus, Role, TaskStatus


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginIn(BaseModel):
    email: str
    password: str


class UserOut(ORM):
    id: int
    name: str
    email: str
    role: Role
    team: str
    title: str
    skills: list[str]


class TokenOut(BaseModel):
    access_token: str
    user: UserOut


class MemberOut(UserOut):
    load_hours: float
    capacity_hours: float


class ProjectIn(BaseModel):
    title: str
    goal: str
    deadline: date
    use_rag: bool = True


class TaskOut(ORM):
    id: int
    project_id: int
    title: str
    description: str
    required_skills: list[str]
    estimate_hours: float
    deadline: date | None
    assignee_id: int | None
    assignee_name: str | None = None
    status: TaskStatus
    ai_draft: dict | None


class ProjectOut(ORM):
    id: int
    title: str
    goal: str
    deadline: date
    status: ProjectStatus
    owner_id: int
    use_rag: bool
    planning_error: str | None
    created_at: datetime


class ProjectDetail(ProjectOut):
    tasks: list[TaskOut]


class DraftTask(BaseModel):
    draft_index: int | None = None
    title: str
    description: str = ""
    required_skills: list[str] = []
    estimate_hours: float = 8
    deadline: date | None = None
    assignee_id: int | None = None


class ReviewIn(BaseModel):
    action: str  # approve | regenerate
    feedback: str | None = None
    tasks: list[DraftTask] = []


class SubmissionIn(BaseModel):
    content: str


class SubmissionOut(ORM):
    id: int
    task_id: int
    user_id: int
    content: str
    ai_review: dict | None
    created_at: datetime


class StatusIn(BaseModel):
    status: TaskStatus
