import { type NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const startDate = searchParams.get("startDate") || "2021-01-01"
  const endDate = searchParams.get("endDate") || "2025-12-31"

  try {
    const query = `
      SELECT 
        COALESCE(SUM(views), 0) as total_views,
        COALESCE(SUM(likes), 0) as total_likes,
        COALESCE(SUM(shares), 0) as total_shares,
        COALESCE(SUM(comments_added), 0) as total_comments,
        COALESCE(SUM(likes)::FLOAT / NULLIF(SUM(views), 0) * 100, 0) as like_rate,
        COALESCE(SUM(shares)::FLOAT / NULLIF(SUM(views), 0) * 100, 0) as share_rate,
        COALESCE(SUM(comments_added)::FLOAT / NULLIF(SUM(views), 0) * 100, 0) as comment_rate,
        COALESCE((SUM(likes) + SUM(shares) + SUM(comments_added))::FLOAT / NULLIF(SUM(views), 0) * 100, 0) as engagement_rate
      FROM yt_video_etl
      WHERE 
        publish_year IS NOT NULL 
        AND publish_month IS NOT NULL 
        AND publish_day IS NOT NULL
        AND make_date(publish_year, publish_month, publish_day) BETWEEN $1 AND $2
    `

    const result = await executeQuery(query, [startDate, endDate])
    const data = result[0]

    return NextResponse.json({
      engagement_distribution: {
        total_views: Number.parseFloat(data.total_views),
        total_likes: Number.parseFloat(data.total_likes),
        total_shares: Number.parseFloat(data.total_shares),
        total_comments: Number.parseFloat(data.total_comments),
      },
      engagement_rates: {
        like_rate: Number.parseFloat(data.like_rate),
        share_rate: Number.parseFloat(data.share_rate),
        comment_rate: Number.parseFloat(data.comment_rate),
        engagement_rate: Number.parseFloat(data.engagement_rate),
      },
    })
  } catch (error) {
    console.error("[v0] YouTube engagement distribution error:", error)
    return NextResponse.json({ error: "Failed to fetch engagement distribution" }, { status: 500 })
  }
}
