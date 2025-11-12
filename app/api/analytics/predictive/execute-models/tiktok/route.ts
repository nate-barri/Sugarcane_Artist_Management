import { execSync } from "child_process"
import { NextResponse } from "next/server"

export async function GET() {
  try {
    // Execute the TikTok predictive JSON script
    const output = execSync("python3 scripts/predictive_tiktok_json.py", {
      cwd: process.cwd(),
      encoding: "utf-8",
      maxBuffer: 10 * 1024 * 1024,
    })

    const data = JSON.parse(output)
    return NextResponse.json(data)
  } catch (error: any) {
    console.error("TikTok model execution error:", error.message)
    return NextResponse.json({ error: "Failed to execute TikTok model", details: error.message }, { status: 500 })
  }
}
