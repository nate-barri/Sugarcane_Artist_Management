import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    const query = `
      SELECT 
        CASE 
          WHEN LOWER(video_title) LIKE '%official music video%' THEN 'Official Music Video'
          WHEN LOWER(video_title) LIKE '%official lyric%' OR LOWER(video_title) LIKE '%lyric video%' OR LOWER(video_title) LIKE '%lyric visualizer%' THEN 'Lyric Video'
          WHEN LOWER(video_title) LIKE '%instrumental%' OR LOWER(video_title) LIKE '%karaoke%' THEN 'Instrumental/Karaoke'
          WHEN LOWER(video_title) LIKE '%official audio%' THEN 'Official Audio'
          WHEN LOWER(video_title) LIKE '%playthrough%' OR LOWER(video_title) LIKE '%chords%' OR LOWER(video_title) LIKE '%tabs%' THEN 'Tutorial/Playthrough'
          WHEN LOWER(video_title) LIKE '%live%' THEN 'Live Performance'
          WHEN LOWER(video_title) LIKE '%bts%' THEN 'Behind The Scenes'
          ELSE 'Other'
        END as content_type,
        COUNT(*) as video_count,
        COALESCE(AVG(views), 0) as avg_views,
        COALESCE(AVG(likes), 0) as avg_likes,
        COALESCE(AVG((likes + shares + comments_added)::FLOAT / NULLIF(views, 0) * 100), 0) as avg_engagement_rate
      FROM yt_video_etl
      WHERE 
        publish_year IS NOT NULL 
        AND publish_month IS NOT NULL 
        AND publish_day IS NOT NULL
        AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
      GROUP BY content_type
      ORDER BY avg_views DESC
    `

    const result = await executeQuery(query, [startDate, endDate])
    return NextResponse.json({ content_type: result })
  } catch (error) {
    console.error("[v0] YouTube content type error:", error)
    return NextResponse.json({ error: "Failed to fetch content type data" }, { status: 500 })
  }
}
