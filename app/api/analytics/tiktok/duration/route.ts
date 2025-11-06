import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    const query = `
      WITH duration_buckets AS (
        SELECT 
          CASE 
            WHEN duration_sec <= 15 THEN '0-15s'
            WHEN duration_sec <= 30 THEN '16-30s'
            WHEN duration_sec <= 60 THEN '31-60s'
            WHEN duration_sec <= 120 THEN '61-120s'
            ELSE '120s+'
          END as duration_bucket,
          CASE 
            WHEN duration_sec <= 15 THEN 1
            WHEN duration_sec <= 30 THEN 2
            WHEN duration_sec <= 60 THEN 3
            WHEN duration_sec <= 120 THEN 4
            ELSE 5
          END as sort_order,
          views, likes, shares, comments_added, saves
        FROM public.tt_video_etl
        WHERE duration_sec IS NOT NULL AND views IS NOT NULL
          AND DATE(publish_time) >= $1
          AND DATE(publish_time) <= $2
      )
      SELECT duration_bucket, COUNT(*) as video_count,
             AVG(views) as avg_views, AVG(likes) as avg_likes,
             AVG(COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0)) as avg_engagement
      FROM duration_buckets
      GROUP BY duration_bucket, sort_order
      ORDER BY sort_order;
    `

    const result = await executeQuery(query, [startDate, endDate])

    return NextResponse.json({
      duration: result.map((row: any) => ({
        duration_bucket: row.duration_bucket,
        video_count: Number(row.video_count),
        avg_views: Number(row.avg_views),
        avg_likes: Number(row.avg_likes),
        avg_engagement: Number(row.avg_engagement),
      })),
    })
  } catch (error) {
    console.error("[v0] Duration analysis error:", error)
    return NextResponse.json({ error: "Failed to fetch duration data" }, { status: 500 })
  }
}
