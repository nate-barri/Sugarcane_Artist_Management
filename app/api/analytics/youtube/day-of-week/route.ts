import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    const query = `
      WITH engagement_data AS (
        SELECT 
          CASE EXTRACT(DOW FROM make_date(publish_year, publish_month, publish_day))
            WHEN 0 THEN 'Sunday'
            WHEN 1 THEN 'Monday'
            WHEN 2 THEN 'Tuesday'
            WHEN 3 THEN 'Wednesday'
            WHEN 4 THEN 'Thursday'
            WHEN 5 THEN 'Friday'
            WHEN 6 THEN 'Saturday'
          END as day_of_week,
          (likes + shares + comments_added) as total_engagement
        FROM yt_video_etl
        WHERE 
          publish_year IS NOT NULL 
          AND publish_month IS NOT NULL 
          AND publish_day IS NOT NULL
          AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      )
      SELECT 
        day_of_week,
        AVG(total_engagement) as mean_engagement,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_engagement) as median_engagement
      FROM engagement_data
      GROUP BY day_of_week
      ORDER BY 
        CASE day_of_week
          WHEN 'Monday' THEN 1
          WHEN 'Tuesday' THEN 2
          WHEN 'Wednesday' THEN 3
          WHEN 'Thursday' THEN 4
          WHEN 'Friday' THEN 5
          WHEN 'Saturday' THEN 6
          WHEN 'Sunday' THEN 7
        END
    `

    const result = await executeQuery(query, [startDate, endDate])
    return NextResponse.json({ day_performance: result })
  } catch (error) {
    console.error("[v0] YouTube day of week error:", error)
    return NextResponse.json({ error: "Failed to fetch day of week data" }, { status: 500 })
  }
}
