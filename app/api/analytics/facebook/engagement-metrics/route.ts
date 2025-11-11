import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    console.log("[v0] Facebook engagement-metrics - Fetching all data")

    const result = await sql`
      SELECT 
        SUM(COALESCE(reactions, 0)) as total_reactions,
        SUM(COALESCE(comments, 0)) as total_comments,
        SUM(COALESCE(shares, 0)) as total_shares,
        SUM(COALESCE(reach, 0)) as total_reach
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
    `

    console.log("[v0] Facebook engagement-metrics result:", result[0])

    const row = result[0]
    const total_reach = Number(row?.total_reach) || 0
    const total_reactions = Number(row?.total_reactions) || 0
    const total_comments = Number(row?.total_comments) || 0
    const total_shares = Number(row?.total_shares) || 0

    const response = {
      metrics: {
        reactions: total_reactions,
        comments: total_comments,
        shares: total_shares,
      },
      rates: {
        like_rate: total_reach > 0 ? (total_reactions / total_reach) * 100 : 0,
        comment_rate: total_reach > 0 ? (total_comments / total_reach) * 100 : 0,
        share_rate: total_reach > 0 ? (total_shares / total_reach) * 100 : 0,
        overall_engagement:
          total_reach > 0 ? ((total_reactions + total_comments + total_shares) / total_reach) * 100 : 0,
      },
    }

    console.log("[v0] Facebook engagement-metrics response:", response)

    return NextResponse.json(response)
  } catch (error) {
    console.error("Error fetching Facebook engagement metrics:", error)
    return NextResponse.json({ error: "Failed to fetch engagement metrics" }, { status: 500 })
  }
}
