import { useCallback, useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api'
import type { Assignment, Draft, Member, ProjectStatus } from '../types'

interface Row {
  draft_index: number | null
  title: string
  description: string
  required_skills: string[]
  estimate_hours: number
  assignee_id: number | null
  ai: Assignment | null
}

interface DraftResponse {
  status: ProjectStatus
  error: string | null
  draft: Draft | null
}

export default function ReviewPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [state, setState] = useState<DraftResponse | null>(null)
  const [rows, setRows] = useState<Row[]>([])
  const [members, setMembers] = useState<Member[]>([])
  const [feedback, setFeedback] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const poll = useCallback(async () => {
    const d = await api<DraftResponse>(`/projects/${id}/draft`)
    setState(d)
    if (d.status === 'ACTIVE') navigate(`/projects/${id}`, { replace: true })
    if (d.draft && d.status === 'REVIEW') {
      const draft = d.draft
      setRows(draft.tasks.map((t, i) => ({ ...t, draft_index: i, assignee_id: draft.assignments[i].assignee_id, ai: draft.assignments[i] })))
    }
    return d
  }, [id, navigate])

  useEffect(() => { api<Member[]>('/users').then((m) => setMembers(m.filter((x) => x.role !== 'EXECUTIVE'))) }, [])
  useEffect(() => {
    let timer: number
    const tick = async () => {
      const d = await poll()
      if (d.status === 'PLANNING' && !d.error) timer = window.setTimeout(tick, 2000)
    }
    tick()
    return () => window.clearTimeout(timer)
  }, [poll])

  const update = (i: number, patch: Partial<Row>) => setRows(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  const send = async (action: 'approve' | 'regenerate') => {
    setBusy(true)
    setError('')
    try {
      const tasks = rows.map(({ ai: _ai, ...r }) => r)
      await api(`/projects/${id}/review`, { body: { action, feedback: feedback || null, tasks } })
      if (action === 'approve') navigate(`/projects/${id}`)
      else { setState({ status: 'PLANNING', error: null, draft: null }); setFeedback(''); setTimeout(poll, 1500) }
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  if (!state) return <p className="muted">불러오는 중…</p>
  if (state.error && state.status !== 'REVIEW') return <p className="error">초안 생성 실패: {state.error}</p>
  if (state.status === 'PLANNING' || !state.draft) {
    return (
      <div className="card center-card">
        <div className="spinner" />
        <p>계획 에이전트가 과거 프로젝트를 검색해 태스크를 나누고, 리소스 에이전트가 담당자를 고르고 있습니다.</p>
        <p className="muted">LLM 응답에 따라 수십 초 걸릴 수 있습니다.</p>
      </div>
    )
  }

  const nameOf = (uid: number) => members.find((m) => m.id === uid)?.name ?? `#${uid}`

  return (
    <section>
      <h2>AI 초안 검토 <span className="muted">수정한 뒤 승인하면 태스크가 배정됩니다.</span></h2>
      {state.draft.sources.length > 0 && (
        <details className="card">
          <summary>참고한 사내 문서 {state.draft.sources.length}건</summary>
          {state.draft.sources.map((s, i) => (
            <div key={i} className="source"><strong>{s.title}</strong> <span className="muted">유사도 {s.score}</span><pre>{s.content}</pre></div>
          ))}
        </details>
      )}
      <div className="review-list">
        {rows.map((r, i) => (
          <div className="card review-row" key={i}>
            <div className="fields">
              <input className="title-input" value={r.title} onChange={(e) => update(i, { title: e.target.value })} />
              <textarea rows={2} value={r.description} onChange={(e) => update(i, { description: e.target.value })} />
              <div className="inline">
                <label className="inline">예상 시간
                  <input type="number" min={1} value={r.estimate_hours} className="num-input"
                    onChange={(e) => update(i, { estimate_hours: Number(e.target.value) })} />
                </label>
                <label className="inline">담당자
                  <select value={r.assignee_id ?? ''} onChange={(e) => update(i, { assignee_id: e.target.value ? Number(e.target.value) : null })}>
                    <option value="">미정</option>
                    {members.map((m) => (
                      <option key={m.id} value={m.id}>
                        {r.ai?.candidates.some((c) => c.user_id === m.id) ? '★ ' : ''}{m.name} ({m.title}, {m.load_hours}h)
                      </option>
                    ))}
                  </select>
                </label>
                <button className="ghost danger" onClick={() => setRows(rows.filter((_, j) => j !== i))}>삭제</button>
              </div>
            </div>
            {r.ai && (
              <div className="ai-box">
                <p><strong>AI 추천: {nameOf(r.ai.assignee_id)}</strong> {r.ai.reason}</p>
                <table className="mini">
                  <thead><tr><th>후보</th><th>종합</th><th>스킬</th><th>이력</th><th>업무량</th></tr></thead>
                  <tbody>
                    {r.ai.candidates.map((c) => (
                      <tr key={c.user_id} className={c.user_id === r.assignee_id ? 'picked' : ''}>
                        <td><button className="link" onClick={() => update(i, { assignee_id: c.user_id })}>{c.name}</button></td>
                        <td>{c.score.toFixed(2)}</td><td>{c.skill.toFixed(2)}</td><td>{c.history.toFixed(2)}</td><td>{Math.round(c.load * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
      <button className="ghost" onClick={() => setRows([...rows, { draft_index: null, title: '새 태스크', description: '', required_skills: [], estimate_hours: 8, assignee_id: null, ai: null }])}>
        + 태스크 추가
      </button>
      <div className="card actions">
        <label>재생성 요청 사항 (선택)
          <textarea rows={2} value={feedback} onChange={(e) => setFeedback(e.target.value)} placeholder="예: QA 태스크를 따로 두고, 마케팅은 빼 주세요." />
        </label>
        {error && <p className="error">{error}</p>}
        <div className="inline">
          <button className="ghost" disabled={busy} onClick={() => send('regenerate')}>다시 만들기</button>
          <button disabled={busy || rows.length === 0} onClick={() => send('approve')}>승인하고 배정하기</button>
        </div>
      </div>
    </section>
  )
}
