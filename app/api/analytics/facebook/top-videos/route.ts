import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Number.parseInt(searchParams.get("limit") || "10")
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    console.log("[v0] Facebook top-videos - Date range:", startDate, "to", endDate)

    const result = await sql`
      SELECT 
        post_id,
        title,
        CASE WHEN duration_sec::text = 'NaN' THEN 0 ELSE COALESCE(duration_sec::numeric, 0) END as duration_sec,
        (CASE WHEN reactions::text = 'NaN' THEN 0 ELSE COALESCE(reactions::numeric, 0) END) +
        (CASE WHEN comments::text = 'NaN' THEN 0 ELSE COALESCE(comments::numeric, 0) END) +
        (CASE WHEN shares::text = 'NaN' THEN 0 ELSE COALESCE(shares::numeric, 0) END) as total_engagement,
        CASE WHEN reach::text = 'NaN' THEN 0 ELSE COALESCE(reach::numeric, 0) END as reach
      FROM facebook_data_set
      WHERE publish_time >= ${startDate}::date 
        AND publish_time <= ${endDate}::date
        AND publish_time IS NOT NULL
        AND title IS NOT NULL
      ORDER BY total_engagement DESC
      LIMIT ${limit}
    `

    console.log("[v0] Facebook top-videos - Found", result.length, "videos")

    return NextResponse.json({
      videos: result.map((row: any) => ({
        post_id: row.post_id,
        title: row.title || "Untitled",
        total_engagement: Number(row.total_engagement) || 0,
        reach: Number(row.reach) || 0,
      })),
    })
  } catch (error) {
    console.error("Error fetching Facebook top videos:", error)
    return NextResponse.json({ error: "Failed to fetch top videos" }, { status: 500 })
  }
}
