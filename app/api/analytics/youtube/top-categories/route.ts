import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"
  const limit = Number.parseInt(searchParams.get("limit") || "10")

  try {
    const query = `
      SELECT 
        category,
        SUM(views) as total_views
      FROM yt_video_etl
      WHERE 
        publish_year IS NOT NULL 
        AND publish_month IS NOT NULL 
        AND publish_day IS NOT NULL
        AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
        AND category IS NOT NULL
      GROUP BY category
      ORDER BY total_views DESC
      LIMIT $3
    `

    const result = await executeQuery(query, [startDate, endDate, limit])
    return NextResponse.json({ categories: result })
  } catch (error) {
    console.error("[v0] YouTube top categories error:", error)
    return NextResponse.json({ error: "Failed to fetch top categories" }, { status: 500 })
  }
}
