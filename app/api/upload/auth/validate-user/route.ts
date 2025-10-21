import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL || "")

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 })
    }

    console.log("[v0] Validating user in Neon database:", email)

    const result = await sql("SELECT id, email FROM users WHERE email = $1", [email])

    console.log("[v0] Database query result:", result)

    if (!result || result.length === 0) {
      console.log("[v0] User not found in Neon database")
      return NextResponse.json(
        { error: "User not registered in the system. Please contact an administrator.", exists: false },
        { status: 401 },
      )
    }

    console.log("[v0] User found in Neon database:", result[0])
    return NextResponse.json({ success: true, exists: true, user: result[0] })
  } catch (error) {
    console.error("[v0] User validation error:", error)
    return NextResponse.json({ error: "Failed to validate user", exists: false }, { status: 500 })
  }
}
