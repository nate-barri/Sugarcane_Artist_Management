import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    console.log("[v0] Duration API - Date range:", { startDate, endDate })

    const query = `
      WITH duration_parsed AS (
        SELECT 
          *,
          CASE 
            -- If duration is a plain number (seconds), convert to minutes
            WHEN duration ~ '^[0-9]+$' OR duration ~ '^[0-9]+\\.?[0-9]*$' THEN 
              CAST(duration AS FLOAT) / 60.0
            -- If duration is HH:MM:SS format
            WHEN duration ~ '^[0-9]+:[0-9]+:[0-9]+$' THEN 
              CAST(SPLIT_PART(duration, ':', 1) AS INTEGER) * 60.0 + 
              CAST(SPLIT_PART(duration, ':', 2) AS INTEGER) +
              CAST(SPLIT_PART(duration, ':', 3) AS FLOAT) / 60.0
            -- If duration is MM:SS format
            WHEN duration ~ '^[0-9]+:[0-9]+$' THEN 
              CAST(SPLIT_PART(duration, ':', 1) AS INTEGER) + 
              CAST(SPLIT_PART(duration, ':', 2) AS FLOAT) / 60.0
            ELSE NULL
          END as duration_minutes
        FROM yt_video_etl
        WHERE 
          publish_year IS NOT NULL 
          AND publish_month IS NOT NULL 
          AND publish_day IS NOT NULL
          AND duration IS NOT NULL
          AND duration != ''
          AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      ),
      duration_bucketed AS (
        SELECT 
          *,
          CASE 
            WHEN duration_minutes < 2 THEN '<2min'
            WHEN duration_minutes >= 2 AND duration_minutes < 3 THEN '2-3min'
            WHEN duration_minutes >= 3 AND duration_minutes < 4 THEN '3-4min'
            WHEN duration_minutes >= 4 AND duration_minutes < 5 THEN '4-5min'
            WHEN duration_minutes >= 5 AND duration_minutes < 6 THEN '5-6min'
            ELSE '>6min'
          END as duration_bucket
        FROM duration_parsed
        WHERE duration_minutes IS NOT NULL AND duration_minutes > 0
      )
      SELECT 
        duration_bucket,
        COUNT(*) as video_count,
        COALESCE(AVG(views), 0) as avg_views,
        COALESCE(AVG(likes), 0) as avg_likes,
        COALESCE(AVG((likes + COALESCE(shares, 0) + COALESCE(comments_added, 0))::FLOAT / NULLIF(views, 0) * 100), 0) as avg_engagement_rate
      FROM duration_bucketed
      GROUP BY duration_bucket
      ORDER BY 
        CASE duration_bucket
          WHEN '<2min' THEN 1
          WHEN '2-3min' THEN 2
          WHEN '3-4min' THEN 3
          WHEN '4-5min' THEN 4
          WHEN '5-6min' THEN 5
          WHEN '>6min' THEN 6
        END
    `

    const result = await executeQuery(query, [startDate, endDate])

    console.log("[v0] Duration API - Result count:", result.length)
    if (result.length > 0) {
      console.log("[v0] Duration API - Sample data:", result.slice(0, 2))
    }

    const parsedResult = result.map((row: any) => ({
      duration_bucket: row.duration_bucket,
      video_count: Number.parseInt(row.video_count, 10),
      avg_views: Number.parseFloat(row.avg_views),
      avg_likes: Number.parseFloat(row.avg_likes),
      avg_engagement_rate: Number.parseFloat(row.avg_engagement_rate),
    }))

    return NextResponse.json({ duration: parsedResult })
  } catch (error) {
    console.error("[v0] YouTube duration error:", error)
    return NextResponse.json({ error: "Failed to fetch duration data" }, { status: 500 })
  }
}
