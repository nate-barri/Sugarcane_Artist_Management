import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    const query = `
      SELECT 
        COUNT(*) as total_videos,
        SUM(views) as total_views,
        SUM(likes) as total_likes,
        SUM(shares) as total_shares,
        SUM(comments_added) as total_comments,
        SUM(watch_time_hours) as total_watch_time,
        AVG(views) as avg_views,
        AVG(likes) as avg_likes,
        AVG(watch_time_hours) as avg_watch_time,
        AVG(unique_viewers) as avg_unique_viewers
      FROM public.yt_video_etl
      WHERE views IS NOT NULL
        AND publish_year IS NOT NULL
        AND publish_month IS NOT NULL
        AND publish_day IS NOT NULL
        AND TO_DATE(
          CONCAT(publish_year, '-', LPAD(publish_month::TEXT, 2, '0'), '-', LPAD(publish_day::TEXT, 2, '0')),
          'YYYY-MM-DD'
        ) >= $1::date
        AND TO_DATE(
          CONCAT(publish_year, '-', LPAD(publish_month::TEXT, 2, '0'), '-', LPAD(publish_day::TEXT, 2, '0')),
          'YYYY-MM-DD'
        ) <= $2::date;
    `

    const result = await executeQuery(query, [startDate, endDate])
    const row = result[0] || {}

    const total_views = Number(row.total_views) || 0
    const total_likes = Number(row.total_likes) || 0
    const total_shares = Number(row.total_shares) || 0
    const total_comments = Number(row.total_comments) || 0
    const total_videos = Number(row.total_videos) || 0
    const total_watch_time = Number(row.total_watch_time) || 0

    const safe_divide = (a: number, b: number) => (b !== 0 ? a / b : 0)

    // Calculate engagement rate: (likes + shares + comments) / views * 100
    const engagement_rate = safe_divide(total_likes + total_shares + total_comments, total_views) * 100

    return NextResponse.json({
      total_videos,
      total_views,
      total_likes,
      total_shares,
      total_comments,
      total_watch_time,
      avg_views: Number(row.avg_views) || 0,
      avg_likes: Number(row.avg_likes) || 0,
      avg_watch_time: Number(row.avg_watch_time) || 0,
      avg_unique_viewers: Number(row.avg_unique_viewers) || 0,
      engagement_rate: Number(engagement_rate.toFixed(2)),
      like_rate: Number((safe_divide(total_likes, total_views) * 100).toFixed(2)),
      share_rate: Number((safe_divide(total_shares, total_views) * 100).toFixed(2)),
      comment_rate: Number((safe_divide(total_comments, total_views) * 100).toFixed(2)),
    })
  } catch (error) {
    console.error("[v0] YouTube overview metrics error:", error)
    return NextResponse.json({ error: "Failed to fetch YouTube overview metrics" }, { status: 500 })
  }
}
