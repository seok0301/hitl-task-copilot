import { Link, Navigate, Route, Routes } from 'react-router-dom'
import { useAuth } from './auth'
import Board from './pages/Board'
import Executive from './pages/Executive'
import Login from './pages/Login'
import Manager from './pages/Manager'
import ProjectView from './pages/ProjectView'
import ReviewPage from './pages/Review'
import { ROLE_LABEL } from './types'

const HOME = { EMPLOYEE: '/board', MANAGER: '/manager', EXECUTIVE: '/executive' } as const

export default function App() {
  const { user, loading, logout } = useAuth()
  if (loading) return <div className="center muted">불러오는 중…</div>
  if (!user) return <Login />

  return (
    <>
      <header className="topbar">
        <Link to={HOME[user.role]} className="brand">HITL Task Copilot</Link>
        <nav>
          {user.role === 'MANAGER' && <Link to="/manager">프로젝트</Link>}
          {user.role !== 'EXECUTIVE' && <Link to="/board">내 업무</Link>}
          {user.role !== 'EMPLOYEE' && <Link to="/executive">현황 대시보드</Link>}
        </nav>
        <div className="me">
          <span>{user.name} · {user.team} · {ROLE_LABEL[user.role]}</span>
          <button className="ghost" onClick={logout}>로그아웃</button>
        </div>
      </header>
      <main>
        <Routes>
          <Route path="/board" element={<Board />} />
          <Route path="/manager" element={<Manager />} />
          <Route path="/projects/:id" element={<ProjectView />} />
          <Route path="/projects/:id/review" element={<ReviewPage />} />
          <Route path="/executive" element={<Executive />} />
          <Route path="*" element={<Navigate to={HOME[user.role]} replace />} />
        </Routes>
      </main>
    </>
  )
}
