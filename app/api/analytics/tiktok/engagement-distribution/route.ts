import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    const query = `
      SELECT 
        SUM(views)::BIGINT AS total_views,
        SUM(likes)::BIGINT AS total_likes,
        SUM(shares)::BIGINT AS total_shares,
        SUM(comments_added)::BIGINT AS total_comments,
        SUM(saves)::BIGINT AS total_saves,
        AVG(CASE WHEN views > 0 THEN (likes::FLOAT / views) * 100 ELSE 0 END) AS avg_like_rate,
        AVG(CASE WHEN views > 0 THEN (shares::FLOAT / views) * 100 ELSE 0 END) AS avg_share_rate,
        AVG(CASE WHEN views > 0 THEN (comments_added::FLOAT / views) * 100 ELSE 0 END) AS avg_comment_rate,
        AVG(CASE WHEN views > 0 THEN 
          ((COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0))::FLOAT / views) * 100
        ELSE 0 END) AS avg_engagement_rate
      FROM public.tt_video_etl
      WHERE views > 0
        AND DATE(publish_time) >= $1
        AND DATE(publish_time) <= $2;
    `

    const result = await executeQuery(query, [startDate, endDate])
    const row = result[0]

    return NextResponse.json({
      engagement_distribution: {
        total_views: Number(row.total_views) || 0,
        total_likes: Number(row.total_likes) || 0,
        total_shares: Number(row.total_shares) || 0,
        total_comments: Number(row.total_comments) || 0,
        total_saves: Number(row.total_saves) || 0,
      },
      engagement_rates: {
        like_rate: Number(row.avg_like_rate) || 0,
        share_rate: Number(row.avg_share_rate) || 0,
        comment_rate: Number(row.avg_comment_rate) || 0,
        engagement_rate: Number(row.avg_engagement_rate) || 0,
      },
    })
  } catch (error) {
    console.error("[v0] Engagement distribution error:", error)
    return NextResponse.json({ error: "Failed to fetch engagement distribution data" }, { status: 500 })
  }
}
