from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.agents.reporter import all_metrics, run_report
from app.auth import current_user, require_roles
from app.db import get_db
from app.models import Role, User
from app.rag.retriever import search

router = APIRouter(tags=["dashboard"])
lead = require_roles(Role.MANAGER, Role.EXECUTIVE)


@router.get("/dashboard/metrics")
def metrics(db: Session = Depends(get_db), _: User = Depends(lead)):
    return all_metrics(db)


@router.post("/dashboard/brief")
def brief(_: User = Depends(lead)):
    return run_report()


@router.get("/documents/search")
def search_documents(
    q: str = Query(min_length=1), db: Session = Depends(get_db), user: User = Depends(current_user)
):
    return [h.__dict__ for h in search(db, q, user.role, k=5)]
