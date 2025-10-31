"use client"

import type React from "react"

import { createContext, useContext, useEffect, useState } from "react"
import { useRouter } from "next/navigation"

interface User {
  id: string
  name: string
  email: string
}

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<boolean>
  signup: (name: string, email: string, password: string) => Promise<boolean>
  logout: () => void
  loading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    // Check if user is logged in on mount
    const token = localStorage.getItem("authToken")
    const userData = localStorage.getItem("currentUser")

    if (token && userData) {
      try {
        setUser(JSON.parse(userData))
        // Set cookie for middleware
        document.cookie = `authToken=${token}; path=/`
      } catch (error) {
        // Clear invalid data
        localStorage.removeItem("authToken")
        localStorage.removeItem("currentUser")
      }
    }
    setLoading(false)
  }, [])

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const users = JSON.parse(localStorage.getItem("users") || "[]")
      const foundUser = users.find((u: any) => u.email === email && u.password === password)

      if (foundUser) {
        setUser(foundUser)
        localStorage.setItem("authToken", "authenticated")
        localStorage.setItem("currentUser", JSON.stringify(foundUser))
        // Set cookie for middleware
        document.cookie = "authToken=authenticated; path=/"
        return true
      }
      return false
    } catch (error) {
      return false
    }
  }

  const signup = async (name: string, email: string, password: string): Promise<boolean> => {
    try {
      const users = JSON.parse(localStorage.getItem("users") || "[]")

      // Check if user already exists
      if (users.find((u: any) => u.email === email)) {
        return false
      }

      const newUser = {
        id: Date.now().toString(),
        name,
        email,
        password,
        createdAt: new Date().toISOString(),
      }

      users.push(newUser)
      localStorage.setItem("users", JSON.stringify(users))

      // Auto login after signup
      setUser(newUser)
      localStorage.setItem("authToken", "authenticated")
      localStorage.setItem("currentUser", JSON.stringify(newUser))
      // Set cookie for middleware
      document.cookie = "authToken=authenticated; path=/"

      return true
    } catch (error) {
      return false
    }
  }

  const logout = () => {
    setUser(null)
    localStorage.removeItem("authToken")
    localStorage.removeItem("currentUser")
    // Clear cookie
    document.cookie = "authToken=; path=/; expires=Thu, 01 Jan 1970 00:00:01 GMT"
    router.push("/login")
  }

  return <AuthContext.Provider value={{ user, login, signup, logout, loading }}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
