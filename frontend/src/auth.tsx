import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { api, token } from './api'
import type { User } from './types'

interface AuthState {
  user: User | null
  loading: boolean
  login: (email: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(Boolean(token.get()))

  useEffect(() => {
    if (!token.get()) return
    api<User>('/auth/me')
      .then(setUser)
      .catch(() => token.clear())
      .finally(() => setLoading(false))
  }, [])

  const login = async (email: string, password: string) => {
    const res = await api<{ access_token: string; user: User }>('/auth/login', { body: { email, password } })
    token.set(res.access_token)
    setUser(res.user)
  }

  const logout = () => {
    token.clear()
    setUser(null)
  }

  return <AuthContext.Provider value={{ user, loading, login, logout }}>{children}</AuthContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {
  const ctx = useContext(AuthContext)
  if (!ctx) throw new Error('AuthProvider가 필요합니다.')
  return ctx
}
