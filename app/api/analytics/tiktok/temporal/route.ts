import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    const queryMonthly = `
      SELECT publish_year, publish_month, COUNT(*) as video_count,
             SUM(views) as total_views, SUM(likes) as total_likes,
             SUM(shares) as total_shares, SUM(comments_added) as total_comments,
             AVG(views) as avg_views, AVG(likes) as avg_likes
      FROM public.tt_video_etl
      WHERE publish_year IS NOT NULL AND publish_month IS NOT NULL AND views IS NOT NULL
        AND DATE(publish_time) >= $1
        AND DATE(publish_time) <= $2
      GROUP BY publish_year, publish_month
      ORDER BY publish_year ASC, publish_month ASC;
    `

    const result = await executeQuery(queryMonthly, [startDate, endDate])

    return NextResponse.json({
      monthly: result.map((row: any) => ({
        publish_year: Number(row.publish_year),
        publish_month: Number(row.publish_month),
        video_count: Number(row.video_count),
        total_views: Number(row.total_views),
        total_likes: Number(row.total_likes),
        total_shares: Number(row.total_shares),
        total_comments: Number(row.total_comments),
        avg_views: Number(row.avg_views),
        avg_likes: Number(row.avg_likes),
      })),
    })
  } catch (error) {
    console.error("[v0] Temporal analysis error:", error)
    return NextResponse.json({ error: "Failed to fetch temporal data" }, { status: 500 })
  }
}
