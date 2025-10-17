import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL || "")

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 })
    }

    const result = await sql("INSERT INTO users (email) VALUES ($1) ON CONFLICT (email) DO NOTHING RETURNING id", [
      email,
    ])

    if (result.length === 0) {
      return NextResponse.json({ success: true, message: "User already exists" })
    }

    return NextResponse.json({ success: true, message: "User registered successfully" })
  } catch (error) {
    console.error("[v0] User registration error:", error)
    return NextResponse.json({ error: "Failed to register user" }, { status: 500 })
  }
}
