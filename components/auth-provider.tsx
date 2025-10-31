"use client"

import { createContext, useContext, type ReactNode } from "react"
import { useRouter } from "next/navigation"
import { useUser as useStackUser } from "@stackframe/stack"

interface AuthContextType {
  user: any | null
  isLoading: boolean
  logout: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const user = useStackUser()
  const router = useRouter()

  const logout = async () => {
    try {
      if (user) {
        await user.signOut()
      }
      // Redirect to login page after successful logout
      router.push("/login")
    } catch (error) {
      console.error("Logout error:", error)
      // Still redirect even if there's an error
      router.push("/login")
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user: user || null,
        isLoading: false,
        logout,
      }}
    >
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
