import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    const query = `
      SELECT 
        COUNT(*) as total_videos,
        COALESCE(SUM(views), 0) as total_views,
        COALESCE(SUM(likes), 0) as total_likes,
        COALESCE(SUM(shares), 0) as total_shares,
        COALESCE(SUM(comments_added), 0) as total_comments,
        COALESCE(SUM(watch_time_hours), 0) as total_watch_time,
        COALESCE(AVG(views), 0) as avg_views,
        COALESCE(
          (SUM(likes) + SUM(shares) + SUM(comments_added))::FLOAT / NULLIF(SUM(views), 0) * 100,
          0
        ) as engagement_rate
      FROM yt_video_etl
      WHERE 
        publish_year IS NOT NULL 
        AND publish_month IS NOT NULL 
        AND publish_day IS NOT NULL
        AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
    `

    const result = await executeQuery(query, [startDate, endDate])
    return NextResponse.json(result[0] || {})
  } catch (error) {
    console.error("[v0] YouTube overview error:", error)
    return NextResponse.json({ error: "Failed to fetch overview data" }, { status: 500 })
  }
}
