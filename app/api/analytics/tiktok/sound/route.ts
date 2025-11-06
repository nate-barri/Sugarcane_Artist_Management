import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    const query = `
      SELECT 
        CASE 
          WHEN sound_used IS NULL OR sound_used = '' OR TRIM(sound_used) = '' THEN 'No Sound'
          ELSE TRIM(sound_used)
        END as sound_category,
        COUNT(*) as video_count, 
        SUM(views) as total_views,
        AVG(views) as avg_views, 
        AVG(likes) as avg_likes
      FROM public.tt_video_etl
      WHERE views IS NOT NULL
        AND LOWER(TRIM(sound_used)) NOT IN ('sunet original', 'please', 'apt. - rose')
        AND DATE(publish_time) >= $1
        AND DATE(publish_time) <= $2
      GROUP BY sound_category
      ORDER BY total_views DESC
      LIMIT 20;
    `

    const result = await executeQuery(query, [startDate, endDate])

    return NextResponse.json({
      sound: result.map((row: any) => ({
        sound_category: row.sound_category,
        video_count: Number(row.video_count),
        total_views: Number(row.total_views),
        avg_views: Number(row.avg_views),
        avg_likes: Number(row.avg_likes),
      })),
    })
  } catch (error) {
    console.error("[v0] Sound analysis error:", error)
    return NextResponse.json({ error: "Failed to fetch sound data" }, { status: 500 })
  }
}
