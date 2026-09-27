export type Role = 'EMPLOYEE' | 'MANAGER' | 'EXECUTIVE'
export type TaskStatus = 'TODO' | 'IN_PROGRESS' | 'SUBMITTED' | 'DONE'
export type ProjectStatus = 'PLANNING' | 'REVIEW' | 'ACTIVE' | 'DONE'

export interface User {
  id: number
  name: string
  email: string
  role: Role
  team: string
  title: string
  skills: string[]
}

export interface Member extends User {
  load_hours: number
  capacity_hours: number
}

export interface Candidate {
  user_id: number
  name: string
  team: string
  score: number
  skill: number
  history: number
  load: number
  similar_past_task: string | null
}

export interface DraftTask {
  title: string
  description: string
  required_skills: string[]
  estimate_hours: number
}

export interface Assignment {
  assignee_id: number
  candidates: Candidate[]
  reason: string
}

export interface Draft {
  tasks: DraftTask[]
  assignments: Assignment[]
  sources: { title: string; score: number; content: string }[]
}

export interface Task {
  id: number
  project_id: number
  title: string
  description: string
  required_skills: string[]
  estimate_hours: number
  deadline: string | null
  assignee_id: number | null
  assignee_name: string | null
  status: TaskStatus
  ai_draft: (DraftTask & Assignment) | null
}

export interface Project {
  id: number
  title: string
  goal: string
  deadline: string
  status: ProjectStatus
  owner_id: number
  use_rag: boolean
  planning_error: string | null
  created_at: string
  tasks?: Task[]
}

export interface Review {
  coverage: number
  met: string[]
  missing: string[]
  summary: string
}

export interface Submission {
  id: number
  task_id: number
  user_id: number
  content: string
  ai_review: Review | null
  created_at: string
}

export interface Metrics {
  project_id: number
  title: string
  deadline: string
  days_left: number
  task_count: number
  status_counts: Record<TaskStatus, number>
  progress: number
  overdue_tasks: string[]
  avg_review_coverage: number | null
  remaining_hours_by_person: Record<string, number>
}

export interface Brief {
  headline: string
  highlights: string[]
  risks: string[]
  next_actions: string[]
}

export const STATUS_LABEL: Record<TaskStatus | ProjectStatus, string> = {
  TODO: '할 일',
  IN_PROGRESS: '진행 중',
  SUBMITTED: '제출됨',
  DONE: '완료',
  PLANNING: '초안 작성 중',
  REVIEW: '승인 대기',
  ACTIVE: '진행 중',
}

export const ROLE_LABEL: Record<Role, string> = {
  EMPLOYEE: '실무자',
  MANAGER: '관리자',
  EXECUTIVE: '경영진',
}
