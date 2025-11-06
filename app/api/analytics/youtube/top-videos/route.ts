import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const limit = Number.parseInt(searchParams.get("limit") || "10")
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    const query = `
      SELECT 
        video_id,
        video_title as title,
        category,
        views,
        likes,
        shares,
        comments_added,
        watch_time_hours,
        publish_year,
        publish_month,
        publish_day
      FROM yt_video_etl
      WHERE 
        publish_year IS NOT NULL 
        AND publish_month IS NOT NULL 
        AND publish_day IS NOT NULL
        AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      ORDER BY views DESC NULLS LAST
      LIMIT $3
    `

    const result = await executeQuery(query, [startDate, endDate, limit])
    return NextResponse.json({ videos: result })
  } catch (error) {
    console.error("[v0] YouTube top videos error:", error)
    return NextResponse.json({ error: "Failed to fetch top videos" }, { status: 500 })
  }
}
