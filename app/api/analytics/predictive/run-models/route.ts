import { NextResponse } from "next/server"
import { execSync } from "child_process"
import path from "path"

export const maxDuration = 60 // Allow longer execution time for Python scripts

export async function GET() {
  try {
    console.log("[v0] Starting Python model execution...")

    const scriptDir = path.join(process.cwd(), "scripts")
    const allData = {
      meta: { backtest: {}, reach6m: {}, existingPostsForecast: {} },
      youtube: { cumulativeModel: {}, catalogViews: {} },
      tiktok: { channelViews: {}, predictionAccuracy: {}, cumulativeForecast: {} },
    }

    try {
      console.log("[v0] Executing Facebook predictive models...")
      const facebookOutput = execSync(`python3 "${path.join(scriptDir, "predictive_facebook_json.py")}"`, {
        encoding: "utf-8",
        cwd: scriptDir,
        maxBuffer: 10 * 1024 * 1024, // 10MB buffer for large outputs
        timeout: 30000,
      })
      const facebookData = JSON.parse(facebookOutput)
      allData.meta = facebookData.meta || allData.meta
      console.log("[v0] Facebook models completed")
    } catch (fbError) {
      console.error("[v0] Facebook model error:", fbError)
    }

    try {
      console.log("[v0] Executing YouTube predictive models...")
      const youtubeOutput = execSync(`python3 "${path.join(scriptDir, "predictive_youtube_json.py")}"`, {
        encoding: "utf-8",
        cwd: scriptDir,
        maxBuffer: 10 * 1024 * 1024,
        timeout: 30000,
      })
      const youtubeData = JSON.parse(youtubeOutput)
      allData.youtube = youtubeData.youtube || allData.youtube
      console.log("[v0] YouTube models completed")
    } catch (ytError) {
      console.error("[v0] YouTube model error:", ytError)
    }

    try {
      console.log("[v0] Executing TikTok predictive models...")
      const tiktokOutput = execSync(`python3 "${path.join(scriptDir, "predictive_tiktok_json.py")}"`, {
        encoding: "utf-8",
        cwd: scriptDir,
        maxBuffer: 10 * 1024 * 1024,
        timeout: 30000,
      })
      const tiktokData = JSON.parse(tiktokOutput)
      allData.tiktok = tiktokData.tiktok || allData.tiktok
      console.log("[v0] TikTok models completed")
    } catch (tkError) {
      console.error("[v0] TikTok model error:", tkError)
    }

    console.log("[v0] All models executed successfully")
    return NextResponse.json(allData)
  } catch (error) {
    console.error("[v0] Critical error in model execution:", error)
    return NextResponse.json(
      {
        error: "Failed to execute models",
        message: error instanceof Error ? error.message : "Unknown error",
      },
      { status: 500 },
    )
  }
}
