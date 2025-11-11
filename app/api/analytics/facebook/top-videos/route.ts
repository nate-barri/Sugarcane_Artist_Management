import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Number.parseInt(searchParams.get("limit") || "10")

    console.log("[v0] Facebook top-videos - Fetching top", limit, "videos")

    const result = await sql`
      SELECT 
        post_id,
        title,
        COALESCE(duration_sec, 0) as duration_sec,
        (COALESCE(reactions, 0) + COALESCE(comments, 0) + COALESCE(shares, 0)) as total_engagement,
        COALESCE(reach, 0) as reach,
        CASE 
          WHEN COALESCE(duration_sec, 0) > 0 AND average_seconds_viewed IS NOT NULL 
          THEN (average_seconds_viewed / duration_sec * 100)
          ELSE NULL
        END as completion_rate
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
      ORDER BY (COALESCE(reactions, 0) + COALESCE(comments, 0) + COALESCE(shares, 0)) DESC
      LIMIT ${limit}
    `

    console.log("[v0] Facebook top-videos - Found", result.length, "videos")

    return NextResponse.json({
      videos: result.map((row: any) => ({
        post_id: String(row.post_id),
        title: row.title || "Untitled",
        duration: Number(row.duration_sec) || 0,
        total_engagement: Number(row.total_engagement) || 0,
        reach: Number(row.reach) || 0,
        completion_rate: row.completion_rate ? Number(row.completion_rate) : null,
      })),
    })
  } catch (error) {
    console.error("Error fetching Facebook top videos:", error)
    return NextResponse.json({ error: "Failed to fetch top videos" }, { status: 500 })
  }
}
