import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    console.log("[v0] Content Distribution API - Date range:", { startDate, endDate })

    const query = `
      WITH content_classified AS (
        SELECT 
          CASE 
            WHEN LOWER(video_title) LIKE '%official music video%' THEN 'Official Music Video'
            WHEN LOWER(video_title) LIKE '%official lyric%' OR LOWER(video_title) LIKE '%lyric video%' OR LOWER(video_title) LIKE '%lyric visualiz%' THEN 'Lyric Video'
            WHEN LOWER(video_title) LIKE '%instrumental%' OR LOWER(video_title) LIKE '%karaoke%' THEN 'Instrumental/Karaoke'
            WHEN LOWER(video_title) LIKE '%official audio%' THEN 'Official Audio'
            WHEN LOWER(video_title) LIKE '%playthrough%' OR LOWER(video_title) LIKE '%chord%' OR LOWER(video_title) LIKE '%tab%' THEN 'Tutorial/Playthrough'
            WHEN LOWER(video_title) LIKE '%live%' THEN 'Live Performance'
            ELSE 'Other'
          END as content_type,
          views,
          likes,
          shares,
          comments_added
        FROM yt_video_etl
        WHERE 
          publish_year IS NOT NULL 
          AND publish_month IS NOT NULL 
          AND publish_day IS NOT NULL
          AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      )
      SELECT 
        content_type,
        COUNT(*) as video_count,
        ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER () * 100, 1) as percentage
      FROM content_classified
      GROUP BY content_type
      ORDER BY video_count DESC
    `

    const result = await executeQuery(query, [startDate, endDate])

    const parsedResult = result.map((row: any) => ({
      content_type: row.content_type,
      video_count: Number.parseInt(row.video_count, 10),
      percentage: Number.parseFloat(row.percentage),
    }))

    console.log("[v0] Content Distribution API - Result count:", parsedResult.length)
    console.log("[v0] Content Distribution API - Sample data:", parsedResult.slice(0, 2))

    return NextResponse.json({ distribution: parsedResult })
  } catch (error) {
    console.error("[v0] YouTube content type distribution error:", error)
    return NextResponse.json({ error: "Failed to fetch content type distribution" }, { status: 500 })
  }
}
