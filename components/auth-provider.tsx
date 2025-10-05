"use client"

import { createContext, useContext, type ReactNode } from "react"
import { useUser as useStackUser } from "@stackframe/stack"

interface AuthContextType {
  user: any | null
  isLoading: boolean
  signOut: () => Promise<void>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const user = useStackUser()

  const signOut = async () => {
    // Stack Auth handles sign out through its own methods
    if (user) {
      await user.signOut()
    }
  }

  return (
    <AuthContext.Provider
      value={{
        user: user || null,
        isLoading: false,
        signOut,
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
