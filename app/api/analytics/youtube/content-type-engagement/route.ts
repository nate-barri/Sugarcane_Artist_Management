import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
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
          comments_added,
          CASE WHEN views > 0 THEN (likes + shares + comments_added)::FLOAT / views ELSE 0 END as engagement_rate
        FROM yt_video_etl
        WHERE 
          publish_year IS NOT NULL 
          AND publish_month IS NOT NULL 
          AND publish_day IS NOT NULL
          AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      )
      SELECT 
        content_type,
        AVG(engagement_rate) * 100 as avg_engagement_rate
      FROM content_classified
      GROUP BY content_type
      ORDER BY avg_engagement_rate DESC
    `

    const result = await executeQuery(query, [startDate, endDate])
    return NextResponse.json({ engagement: result })
  } catch (error) {
    console.error("[v0] YouTube content type engagement error:", error)
    return NextResponse.json({ error: "Failed to fetch content type engagement" }, { status: 500 })
  }
}
