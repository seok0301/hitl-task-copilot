import { Fragment, useCallback, useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api } from '../api'
import { useAuth } from '../auth'
import { STATUS_LABEL, type Project, type Review, type Submission, type Task } from '../types'

export function ReviewResult({ review }: { review: Review }) {
  return (
    <div className="ai-box">
      <p><strong>AI 검토: 명세 충족도 {review.coverage}%</strong> {review.summary}</p>
      {review.met.length > 0 && <p className="muted">충족: {review.met.join(', ')}</p>}
      {review.missing.length > 0 && <p className="muted">부족: {review.missing.join(', ')}</p>}
    </div>
  )
}

export default function ProjectView() {
  const { id } = useParams()
  const { user } = useAuth()
  const [project, setProject] = useState<Project | null>(null)
  const [openTask, setOpenTask] = useState<number | null>(null)
  const [subs, setSubs] = useState<Submission[]>([])
  const load = useCallback(() => api<Project>(`/projects/${id}`).then(setProject), [id])
  useEffect(() => { load() }, [load])
  useEffect(() => {
    if (openTask) api<Submission[]>(`/tasks/${openTask}/submissions`).then(setSubs)
  }, [openTask])

  if (!project) return <p className="muted">불러오는 중…</p>
  const tasks = project.tasks ?? []
  const total = tasks.reduce((a, t) => a + t.estimate_hours, 0) || 1
  const done = tasks.reduce((a, t) => a + t.estimate_hours * (t.status === 'DONE' ? 1 : t.status === 'SUBMITTED' ? 0.5 : 0), 0)

  const accept = async (t: Task) => {
    await api(`/tasks/${t.id}/accept`, { body: {} })
    load()
  }

  return (
    <section>
      <h2>{project.title}</h2>
      <p>{project.goal}</p>
      <p className="muted">마감 {project.deadline} · 진척도 {Math.round((100 * done) / total)}%</p>
      <div className="bar big"><div style={{ width: `${(100 * done) / total}%` }} /></div>
      <table>
        <thead><tr><th>태스크</th><th>담당자</th><th>시간</th><th>상태</th><th /></tr></thead>
        <tbody>
          {tasks.map((t) => (
            <Fragment key={t.id}>
              <tr>
                <td>
                  {t.title}
                  {t.ai_draft && t.ai_draft.assignee_id !== t.assignee_id && <span className="badge edited">담당자 변경</span>}
                  {!t.ai_draft && <span className="badge edited">관리자 추가</span>}
                </td>
                <td>{t.assignee_name ?? '미정'}</td>
                <td>{t.estimate_hours}</td>
                <td><span className={`badge ${t.status}`}>{STATUS_LABEL[t.status]}</span></td>
                <td className="inline">
                  <button className="link" onClick={() => setOpenTask(openTask === t.id ? null : t.id)}>제출물</button>
                  {user?.role === 'MANAGER' && t.status === 'SUBMITTED' && <button onClick={() => accept(t)}>완료 승인</button>}
                </td>
              </tr>
              {openTask === t.id && (
                <tr>
                  <td colSpan={5}>
                    {subs.length === 0 && <span className="muted">아직 제출물이 없습니다.</span>}
                    {subs.map((s) => (
                      <div className="submission" key={s.id}>
                        <pre>{s.content}</pre>
                        {s.ai_review && <ReviewResult review={s.ai_review} />}
                      </div>
                    ))}
                  </td>
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </section>
  )
}
