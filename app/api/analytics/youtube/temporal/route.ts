import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    const query = `
      SELECT 
        publish_year,
        publish_month,
        COUNT(*) as video_count,
        COALESCE(SUM(views), 0) as total_views,
        COALESCE(AVG(views), 0) as avg_views,
        COALESCE(SUM(likes), 0) as total_likes,
        COALESCE(SUM(watch_time_hours), 0) as total_watch_time
      FROM yt_video_etl
      WHERE 
        publish_year IS NOT NULL 
        AND publish_month IS NOT NULL 
        AND publish_day IS NOT NULL
        AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      GROUP BY publish_year, publish_month
      ORDER BY publish_year, publish_month
    `

    const result = await executeQuery(query, [startDate, endDate])
    return NextResponse.json({ monthly: result })
  } catch (error) {
    console.error("[v0] YouTube temporal error:", error)
    return NextResponse.json({ error: "Failed to fetch temporal data" }, { status: 500 })
  }
}
