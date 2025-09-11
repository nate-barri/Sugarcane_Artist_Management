"use client"

import type React from "react"
import { createContext, useContext, useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { apiService } from "@/lib/api"

interface User {
  id: string
  email: string
  username: string
  full_name: string
  artist_name?: string
  bio?: string
  is_verified: boolean
}

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<boolean>
  signup: (name: string, email: string, password: string) => Promise<boolean>
  logout: () => void
  loading: boolean
  refreshUser: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    checkAuthStatus()
  }, [])

  const checkAuthStatus = async () => {
    const token = localStorage.getItem("authToken")
    const userData = localStorage.getItem("currentUser")

    if (token && userData) {
      try {
        // Verify token with backend
        const response = await apiService.getProfile()
        if (response.data) {
          setUser(response.data)
        } else {
          // Token is invalid, clear local storage
          apiService.clearToken()
        }
      } catch (error) {
        console.error("Auth check failed:", error)
        apiService.clearToken()
      }
    }
    setLoading(false)
  }

  const refreshUser = async () => {
    try {
      const response = await apiService.getProfile()
      if (response.data) {
        setUser(response.data)
      }
    } catch (error) {
      console.error("Failed to refresh user:", error)
    }
  }

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      const response = await apiService.login(email, password)

      if (response.data) {
        setUser(response.data.user)
        return true
      } else {
        console.error("Login failed:", response.error)
        return false
      }
    } catch (error) {
      console.error("Login error:", error)
      return false
    }
  }

  const signup = async (name: string, email: string, password: string): Promise<boolean> => {
    try {
      const response = await apiService.register({
        email,
        username: email.split("@")[0], // Generate username from email
        full_name: name,
        password,
        password_confirm: password,
      })

      if (response.data) {
        setUser(response.data.user)
        return true
      } else {
        console.error("Signup failed:", response.error)
        return false
      }
    } catch (error) {
      console.error("Signup error:", error)
      return false
    }
  }

  const logout = async () => {
    await apiService.logout()
    setUser(null)
    router.push("/login")
  }

  return (
    <AuthContext.Provider value={{ user, login, signup, logout, loading, refreshUser }}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}
