import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const result = await sql`
      SELECT 
        COUNT(DISTINCT post_id) as total_posts,
        SUM(COALESCE(reach, 0)) as total_reach,
        SUM(COALESCE(reactions, 0)) as total_reactions,
        SUM(COALESCE(comments, 0)) as total_comments,
        SUM(COALESCE(shares, 0)) as total_shares,
        SUM(COALESCE(reactions, 0) + COALESCE(comments, 0) + COALESCE(shares, 0)) as total_engagement
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
    `

    const totalReach = Number(result[0]?.total_reach) || 0
    const totalEngagement = Number(result[0]?.total_engagement) || 0
    const engagementRate = totalReach > 0 ? (totalEngagement / totalReach) * 100 : 0

    const response = {
      total_posts: Number(result[0]?.total_posts) || 0,
      total_reach: totalReach,
      total_engagement: totalEngagement,
      engagement_rate: engagementRate,
    }

    return NextResponse.json(response)
  } catch (error) {
    console.error("Error fetching Facebook overview:", error)
    return NextResponse.json({ error: "Failed to fetch overview data" }, { status: 500 })
  }
}
