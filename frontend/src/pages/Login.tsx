import { useState, type FormEvent } from 'react'
import { useAuth } from '../auth'

const DEMO = [
  { email: 'dev.lead@example.com', label: '관리자 (개발팀장)' },
  { email: 'seoyeon@example.com', label: '실무자 (백엔드 개발자)' },
  { email: 'ceo@example.com', label: '경영진 (대표)' },
]

export default function Login() {
  const { login } = useAuth()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setError('')
    try {
      await login(email, password)
    } catch (err) {
      setError((err as Error).message)
    }
  }

  return (
    <div className="center">
      <form className="card login" onSubmit={submit}>
        <h1>HITL Task Copilot</h1>
        <p className="muted">AI가 초안을 만들고, 결정은 사람이 합니다.</p>
        <label>이메일<input value={email} onChange={(e) => setEmail(e.target.value)} autoFocus /></label>
        <label>비밀번호<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} /></label>
        {error && <p className="error">{error}</p>}
        <button type="submit">로그인</button>
        <div className="demo">
          <span className="muted">데모 계정 (비밀번호 password)</span>
          {DEMO.map((d) => (
            <button type="button" className="ghost" key={d.email}
              onClick={() => { setEmail(d.email); setPassword('password') }}>
              {d.label}
            </button>
          ))}
        </div>
      </form>
    </div>
  )
}
