import enum
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import get_settings
from app.db import Base

DIM = get_settings().embedding_dim
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Role(str, enum.Enum):
    EMPLOYEE = "EMPLOYEE"
    MANAGER = "MANAGER"
    EXECUTIVE = "EXECUTIVE"


# 문서 접근 등급: 숫자가 클수록 제한적이다. 역할마다 볼 수 있는 최대 등급이 정해진다.
ACCESS_LEVEL_BY_ROLE = {Role.EMPLOYEE: 1, Role.MANAGER: 2, Role.EXECUTIVE: 3}


class TaskStatus(str, enum.Enum):
    TODO = "TODO"
    IN_PROGRESS = "IN_PROGRESS"
    SUBMITTED = "SUBMITTED"
    DONE = "DONE"


class ProjectStatus(str, enum.Enum):
    PLANNING = "PLANNING"  # 에이전트가 초안 작성 중
    REVIEW = "REVIEW"  # 관리자 승인 대기
    ACTIVE = "ACTIVE"  # 승인되어 진행 중
    DONE = "DONE"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    email: Mapped[str] = mapped_column(String(120), unique=True)
    pw_hash: Mapped[str] = mapped_column(String(100))
    role: Mapped[Role] = mapped_column(Enum(Role))
    team: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(String(50), default="")
    skills: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    skill_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(DIM), nullable=True
    )
    weekly_capacity_hours: Mapped[float] = mapped_column(Float, default=40)


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    goal: Mapped[str] = mapped_column(Text)
    deadline: Mapped[date] = mapped_column(Date)
    status: Mapped[ProjectStatus] = mapped_column(
        Enum(ProjectStatus), default=ProjectStatus.PLANNING
    )
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    graph_thread_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    tasks: Mapped[list["Task"]] = relationship(
        back_populates="project", order_by="Task.id"
    )


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    required_skills: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    estimate_hours: Mapped[float] = mapped_column(Float, default=8)
    deadline: Mapped[date | None] = mapped_column(Date, nullable=True)
    assignee_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"), nullable=True
    )
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus), default=TaskStatus.TODO
    )
    ai_draft: Mapped[dict | None] = mapped_column(JSONType, nullable=True)

    project: Mapped[Project] = relationship(back_populates="tasks")
    assignee: Mapped[User | None] = relationship()
    submissions: Mapped[list["Submission"]] = relationship(
        back_populates="task", order_by="Submission.id"
    )


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("tasks.id"))
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    content: Mapped[str] = mapped_column(Text)
    ai_review: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    task: Mapped[Task] = relationship(back_populates="submissions")


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    access_level: Mapped[int] = mapped_column(Integer, default=1)
    past_project_id: Mapped[int | None] = mapped_column(
        ForeignKey("past_projects.id"), nullable=True
    )

    chunks: Mapped[list["DocChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocChunk(Base):
    __tablename__ = "doc_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    seq: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    access_level: Mapped[int] = mapped_column(Integer, default=1)
    embedding: Mapped[list[float]] = mapped_column(Vector(DIM))

    document: Mapped[Document] = relationship(back_populates="chunks")


class PastProject(Base):
    __tablename__ = "past_projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    goal: Mapped[str] = mapped_column(Text)
    year: Mapped[int] = mapped_column(Integer)

    tasks: Mapped[list["PastTask"]] = relationship(back_populates="project")


class PastTask(Base):
    __tablename__ = "past_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    past_project_id: Mapped[int] = mapped_column(ForeignKey("past_projects.id"))
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    assignee_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    hours: Mapped[float] = mapped_column(Float, default=8)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(DIM), nullable=True)

    project: Mapped[PastProject] = relationship(back_populates="tasks")


class ApprovalLog(Base):
    """HITL 승인 기록. AI 초안과 관리자 최종본을 비교해 수정률을 계산한다."""

    __tablename__ = "approval_logs"

    id: Mapped[int] = mapped_column(primary_key=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    reviewer_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(20))  # approve | regenerate
    draft: Mapped[dict] = mapped_column(JSONType)
    final: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    edit_stats: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    review_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class LLMCall(Base):
    """에이전트별 LLM 호출 기록. 소요 시간과 토큰 평가에 쓴다."""

    __tablename__ = "llm_calls"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent: Mapped[str] = mapped_column(String(30))
    model: Mapped[str] = mapped_column(String(80))
    project_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
