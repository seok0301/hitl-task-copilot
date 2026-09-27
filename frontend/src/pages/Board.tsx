import { useCallback, useEffect, useState } from 'react'
import { api } from '../api'
import { STATUS_LABEL, type Submission, type Task, type TaskStatus } from '../types'
import { ReviewResult } from './ProjectView'

const COLUMNS: TaskStatus[] = ['TODO', 'IN_PROGRESS', 'SUBMITTED', 'DONE']

export default function Board() {
  const [tasks, setTasks] = useState<Task[]>([])
  const [open, setOpen] = useState<Task | null>(null)
  const load = useCallback(() => api<Task[]>('/tasks/mine').then(setTasks), [])
  useEffect(() => { load() }, [load])

  return (
    <section>
      <h2>내 업무</h2>
      <div className="board">
        {COLUMNS.map((col) => (
          <div className="column" key={col}>
            <h3>{STATUS_LABEL[col]} <span className="muted">{tasks.filter((t) => t.status === col).length}</span></h3>
            {tasks.filter((t) => t.status === col).map((t) => (
              <button className="card task-card" key={t.id} onClick={() => setOpen(t)}>
                <strong>{t.title}</strong>
                <span className="muted">{t.estimate_hours}시간{t.deadline && ` · ~${t.deadline}`}</span>
              </button>
            ))}
          </div>
        ))}
      </div>
      {open && <TaskPanel task={open} onClose={() => setOpen(null)} onChange={() => { load(); setOpen(null) }} />}
    </section>
  )
}

function TaskPanel({ task, onClose, onChange }: { task: Task; onClose: () => void; onChange: () => void }) {
  const [subs, setSubs] = useState<Submission[]>([])
  const [content, setContent] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { api<Submission[]>(`/tasks/${task.id}/submissions`).then(setSubs) }, [task.id])

  const start = async () => {
    await api(`/tasks/${task.id}/status`, { method: 'PATCH', body: { status: 'IN_PROGRESS' } })
    onChange()
  }
  const submit = async () => {
    setBusy(true)
    setError('')
    try {
      const s = await api<Submission>(`/tasks/${task.id}/submissions`, { body: { content } })
      setSubs([...subs, s])
      setContent('')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="overlay" onClick={onClose}>
      <div className="card panel" onClick={(e) => e.stopPropagation()}>
        <h3>{task.title}</h3>
        <p>{task.description}</p>
        <p className="muted">예상 {task.estimate_hours}시간 · 상태 {STATUS_LABEL[task.status]}</p>
        {task.status === 'TODO' && <button onClick={start}>시작하기</button>}
        {subs.map((s) => (
          <div className="submission" key={s.id}>
            <pre>{s.content}</pre>
            {s.ai_review && <ReviewResult review={s.ai_review} />}
          </div>
        ))}
        {task.status !== 'DONE' && (
          <>
            <label>산출물 제출
              <textarea rows={6} value={content} onChange={(e) => setContent(e.target.value)}
                placeholder="작업 결과, 링크, 요약 등을 적어 주세요." />
            </label>
            {error && <p className="error">{error}</p>}
            <button disabled={!content.trim() || busy} onClick={submit}>
              {busy ? 'AI가 명세와 대조하는 중…' : '제출하기'}
            </button>
          </>
        )}
        <button className="ghost" onClick={onChange}>닫기</button>
      </div>
    </div>
  )
}
