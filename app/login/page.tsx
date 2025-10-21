"use client"

import type React from "react"

import { useStackApp, useUser } from "@stackframe/stack"
import { useRouter } from "next/navigation"
import { useState } from "react"

export default function LoginPage() {
  const app = useStackApp()
  const user = useUser()
  const router = useRouter()
  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(false)
  const [isSignUp, setIsSignUp] = useState(false)

  if (user) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8 text-center">
          <h2 className="text-2xl font-bold text-gray-900">You are already logged in</h2>
          <button onClick={() => router.push("/import")} className="text-[#123458] hover:text-[#0e2742] font-medium">
            Go to Dashboard
          </button>
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
      if (isSignUp) {
        // Sign up new user
        await app.signUpWithCredential({ email, password })
        router.push("/import")
      } else {
        try {
          console.log("[v0] Attempting sign in with email:", email)
          const result = await app.signInWithCredential({ email, password })
          console.log("[v0] Sign in result:", result)

          const validateResponse = await fetch("/api/auth/validate-user", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email }),
          })

          console.log("[v0] Validation response status:", validateResponse.status)
          const validationData = await validateResponse.json()
          console.log("[v0] Validation data:", validationData)

          if (!validateResponse.ok || !validationData.exists) {
            console.log("[v0] User not found in Neon database, signing out")
            if (result?.user) {
              await result.user.signOut()
            }
            setError("User not registered in the system. Please contact your administrator.")
            setLoading(false)
            return
          }

          console.log("[v0] User validated in Neon database, redirecting to import")
          localStorage.setItem("authToken", email)
          router.push("/import")
        } catch (signInError: any) {
          console.log("[v0] Sign in error:", signInError)
          const errorMessage = signInError?.message || ""
          if (
            errorMessage.toLowerCase().includes("not found") ||
            errorMessage.toLowerCase().includes("invalid") ||
            errorMessage.toLowerCase().includes("unauthorized") ||
            errorMessage.toLowerCase().includes("user does not exist")
          ) {
            setError(
              "User not found or invalid credentials. Please contact your administrator if you need to create an account.",
            )
            setLoading(false)
            return
          }
          throw signInError
        }
      }
    } catch (err: any) {
      console.log("[v0] Auth error:", err)
      const errorMsg = err?.message || "Authentication failed. Please try again."

      if (errorMsg.toLowerCase().includes("not found") || errorMsg.toLowerCase().includes("does not exist")) {
        setError("This account is not registered. Please contact your administrator.")
      } else if (errorMsg.toLowerCase().includes("invalid") || errorMsg.toLowerCase().includes("unauthorized")) {
        setError("Invalid email or password. Please contact your administrator if you need assistance.")
      } else {
        setError(errorMsg)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-bold text-gray-900">
            {isSignUp ? "Create an account" : "Sign in"}
          </h2>
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
                autoComplete={isSignUp ? "new-password" : "current-password"}
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
              {loading ? (isSignUp ? "Creating account..." : "Signing in...") : isSignUp ? "Sign up" : "Sign in"}
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={() => {
                setIsSignUp(!isSignUp)
                setError("")
              }}
              className="text-sm text-[#123458] hover:text-[#0e2742] font-medium"
            >
              {isSignUp ? "Already have an account? Sign in" : "Don't have an account? Sign up"}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}
