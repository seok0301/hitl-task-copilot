import { useEffect, useState, type FormEvent } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../api'
import { STATUS_LABEL, type Member, type Project } from '../types'

export default function Manager() {
  const [projects, setProjects] = useState<Project[]>([])
  const [members, setMembers] = useState<Member[]>([])
  const navigate = useNavigate()
  const [form, setForm] = useState({ title: '', goal: '', deadline: '', use_rag: true })
  const [error, setError] = useState('')

  useEffect(() => {
    api<Project[]>('/projects').then(setProjects)
    api<Member[]>('/users').then(setMembers)
  }, [])

  const create = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      const p = await api<Project>('/projects', { body: form })
      navigate(`/projects/${p.id}/review`)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="grid-2">
      <section>
        <form className="card" onSubmit={create}>
          <h2>새 프로젝트</h2>
          <label>제목<input required value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} /></label>
          <label>목표
            <textarea required rows={3} value={form.goal} onChange={(e) => setForm({ ...form, goal: e.target.value })}
              placeholder="무엇을 달성하려는지 적어 주세요." />
          </label>
          <label>마감일<input required type="date" value={form.deadline} onChange={(e) => setForm({ ...form, deadline: e.target.value })} /></label>
          <label className="inline">
            <input type="checkbox" checked={form.use_rag} onChange={(e) => setForm({ ...form, use_rag: e.target.checked })} />
            과거 프로젝트 문서 검색(RAG) 사용
          </label>
          {error && <p className="error">{error}</p>}
          <button type="submit">AI 초안 만들기</button>
        </form>

        <h2>프로젝트</h2>
        <table>
          <thead><tr><th>제목</th><th>마감</th><th>상태</th><th /></tr></thead>
          <tbody>
            {projects.map((p) => (
              <tr key={p.id}>
                <td>{p.title}</td>
                <td>{p.deadline}</td>
                <td><span className={`badge ${p.status}`}>{STATUS_LABEL[p.status]}</span></td>
                <td>
                  {p.status === 'ACTIVE' || p.status === 'DONE'
                    ? <Link to={`/projects/${p.id}`}>진척 보기</Link>
                    : <Link to={`/projects/${p.id}/review`}>초안 검토</Link>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section>
        <h2>팀원 업무량</h2>
        <div className="card">
          {members.filter((m) => m.role !== 'EXECUTIVE').map((m) => {
            const ratio = Math.min(m.load_hours / m.capacity_hours, 1)
            return (
              <div className="load-row" key={m.id} title={m.skills.join(', ')}>
                <span>{m.name} <span className="muted">{m.title}</span></span>
                <div className="bar"><div style={{ width: `${ratio * 100}%` }} className={ratio > 0.75 ? 'high' : ''} /></div>
                <span className="muted num">{m.load_hours}h</span>
              </div>
            )
          })}
        </div>
      </section>
    </div>
  )
}
