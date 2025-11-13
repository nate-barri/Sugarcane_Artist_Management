"use client"

import type React from "react"
import { useRouter } from "next/navigation"
import { useState, useEffect } from "react"
import { signInWithEmailAndPassword, type User } from "firebase/auth"
import { auth } from "@/lib/firebase"

export default function LoginPage() {
  const router = useRouter()
  const [user, setUser] = useState<User | null>(null)
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [isCheckingAuth, setIsCheckingAuth] = useState(true)

  useEffect(() => {
    const unsubscribe = auth.onAuthStateChanged((currentUser) => {
      setUser(currentUser)
      setIsCheckingAuth(false)
      if (currentUser) {
        router.push("/import")
      }
    })

    return () => unsubscribe()
  }, [router])

  if (isCheckingAuth) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError("")

    if (!email || !password) {
      setError("Please fill in all fields")
      setLoading(false)
      return
    }

    try {
      try {
        const result = await signInWithEmailAndPassword(auth, email, password)

        const validateResponse = await fetch("/api/auth/validate-user", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email }),
        })

        if (!validateResponse.ok || !(await validateResponse.json()).exists) {
          await result.user.getIdToken(true)
          setError("User not registered in the system. Please contact your administrator.")
          setLoading(false)
          return
        }

        localStorage.setItem("authToken", email)
        router.push("/import")
      } catch (signInError: any) {
        const errorMessage = signInError?.message || ""
        if (errorMessage.includes("user-not-found") || errorMessage.includes("invalid-credential")) {
          setError("Invalid email or password. Please contact your administrator if you need assistance.")
        } else {
          setError(errorMessage || "Sign in failed. Please try again.")
        }
        setLoading(false)
        return
      }
    } catch (err: any) {
      const errorMessage = err?.message || "Authentication failed. Please try again."

      if (errorMessage.includes("user-not-found")) {
        setError("This account is not registered. Please contact your administrator.")
      } else if (errorMessage.includes("invalid-credential") || errorMessage.includes("wrong-password")) {
        setError("Invalid email or password.")
      } else {
        setError(errorMessage)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-bold text-gray-900">Sign in</h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="email" className="sr-only">
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                className="relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-t-md focus:outline-none focus:ring-[#123458] focus:border-[#123458] focus:z-10 sm:text-sm"
                placeholder="Email address"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="password" className="sr-only">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                className="relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 rounded-b-md focus:outline-none focus:ring-[#123458] focus:border-[#123458] focus:z-10 sm:text-sm"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
          </div>

          {error && (
            <div className="rounded-md bg-red-50 p-4">
              <p className="text-red-800 text-sm">{error}</p>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-[#123458] hover:bg-[#0e2742] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-[#123458] disabled:opacity-50"
            >
              {loading ? "Signing in..." : "Sign in"}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
