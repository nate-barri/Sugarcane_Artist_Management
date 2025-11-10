import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    console.log("[v0] Facebook post-type-distribution - Fetching all data")

    const result = await sql`
      SELECT 
        post_type,
        COUNT(*) as post_count
      FROM facebook_data_set
      WHERE post_type IS NOT NULL
      GROUP BY post_type
      ORDER BY post_count DESC
    `

    const total = result.reduce((sum: number, row: any) => sum + Number(row.post_count), 0)

    return NextResponse.json({
      distribution: result.map((row: any) => ({
        post_type: row.post_type,
        count: Number(row.post_count),
        percentage: ((Number(row.post_count) / total) * 100).toFixed(1),
      })),
    })
  } catch (error) {
    console.error("Error fetching Facebook post type distribution:", error)
    return NextResponse.json({ error: "Failed to fetch post type distribution" }, { status: 500 })
  }
}
