import { type NextRequest, NextResponse } from "next/server"
import { stackServerApp } from "@/stack"

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 })
    }

    const user = await stackServerApp.listUsers({
      filter: {
        email: email,
      },
    })

    if (!user || user.items.length === 0) {
      return NextResponse.json(
        { error: "User not registered in the system. Please contact an administrator." },
        { status: 401 },
      )
    }

    return NextResponse.json({ success: true, registered: true })
  } catch (error) {
    console.error("[v0] User validation error:", error)
    return NextResponse.json({ error: "Failed to validate user" }, { status: 500 })
  }
}
