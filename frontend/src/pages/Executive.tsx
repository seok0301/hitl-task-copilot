import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../api'
import type { Brief, Metrics } from '../types'

export default function Executive() {
  const [metrics, setMetrics] = useState<Metrics[]>([])
  const [brief, setBrief] = useState<Brief | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => { api<Metrics[]>('/dashboard/metrics').then(setMetrics) }, [])

  const makeBrief = async () => {
    setBusy(true)
    setError('')
    try {
      const r = await api<{ brief: Brief }>('/dashboard/brief', { body: {} })
      setBrief(r.brief)
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section>
      <h2>진행 현황</h2>
      <div className="metric-grid">
        {metrics.map((m) => (
          <Link to={`/projects/${m.project_id}`} className="card metric" key={m.project_id}>
            <strong>{m.title}</strong>
            <div className="big-num">{m.progress}%</div>
            <div className="bar"><div style={{ width: `${m.progress}%` }} /></div>
            <span className="muted">
              태스크 {m.task_count}개 · 완료 {m.status_counts.DONE} · 마감까지 {m.days_left}일
            </span>
            {m.overdue_tasks.length > 0 && <span className="error">기한 초과 {m.overdue_tasks.length}건</span>}
            {m.avg_review_coverage !== null && <span className="muted">제출물 평균 충족도 {m.avg_review_coverage}%</span>}
          </Link>
        ))}
      </div>

      <div className="card">
        <div className="inline between">
          <h3>AI 브리핑</h3>
          <button disabled={busy} onClick={makeBrief}>{busy ? '보고 에이전트가 작성하는 중…' : '브리핑 생성'}</button>
        </div>
        {error && <p className="error">{error}</p>}
        {brief && (
          <div className="brief">
            <p className="headline">{brief.headline}</p>
            <BriefList title="주요 내용" items={brief.highlights} />
            <BriefList title="위험 요소" items={brief.risks} />
            <BriefList title="다음 조치" items={brief.next_actions} />
          </div>
        )}
      </div>
    </section>
  )
}

function BriefList({ title, items }: { title: string; items: string[] }) {
  if (!items?.length) return null
  return (
    <>
      <h4>{title}</h4>
      <ul>{items.map((x, i) => <li key={i}>{x}</li>)}</ul>
    </>
  )
}
