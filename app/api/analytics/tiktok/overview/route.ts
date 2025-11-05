import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET() {
  try {
    const query = `
      SELECT 
        COUNT(*) as total_videos,
        SUM(views) as total_views,
        SUM(likes) as total_likes,
        SUM(shares) as total_shares,
        SUM(comments_added) as total_comments,
        SUM(saves) as total_saves,
        AVG(views) as avg_views,
        AVG(likes) as avg_likes,
        AVG(shares) as avg_shares,
        AVG(comments_added) as avg_comments,
        AVG(saves) as avg_saves,
        AVG(duration_sec) as avg_duration_seconds,
        COUNT(DISTINCT CONCAT(publish_year, '-', publish_month)) as total_months
      FROM public.tt_video_etl
      WHERE views IS NOT NULL;
    `

    const result = await executeQuery(query)
    const row = result[0] || {}

    const total_views = Number(row.total_views) || 0
    const total_likes = Number(row.total_likes) || 0
    const total_shares = Number(row.total_shares) || 0
    const total_comments = Number(row.total_comments) || 0
    const total_saves = Number(row.total_saves) || 0
    const total_videos = Number(row.total_videos) || 0
    const total_months = Number(row.total_months) || 1

    const safe_divide = (a: number, b: number) => (b !== 0 ? a / b : 0)

    const engagement_rate = safe_divide(total_likes + total_shares + total_comments + total_saves, total_views) * 100

    return NextResponse.json({
      total_videos,
      total_views,
      total_likes,
      total_shares,
      total_comments,
      total_saves,
      avg_views: Number(row.avg_views) || 0,
      avg_likes: Number(row.avg_likes) || 0,
      avg_shares: Number(row.avg_shares) || 0,
      avg_comments: Number(row.avg_comments) || 0,
      avg_saves: Number(row.avg_saves) || 0,
      avg_duration_seconds: Number(row.avg_duration_seconds) || 0,
      engagement_rate: Number(engagement_rate.toFixed(2)),
      like_rate: Number((safe_divide(total_likes, total_views) * 100).toFixed(2)),
      share_rate: Number((safe_divide(total_shares, total_views) * 100).toFixed(2)),
      comment_rate: Number((safe_divide(total_comments, total_views) * 100).toFixed(2)),
      save_rate: Number((safe_divide(total_saves, total_views) * 100).toFixed(2)),
      avg_videos_per_month: Number(safe_divide(total_videos, total_months).toFixed(2)),
    })
  } catch (error) {
    console.error("[v0] Overview metrics error:", error)
    return NextResponse.json({ error: "Failed to fetch overview metrics" }, { status: 500 })
  }
}
