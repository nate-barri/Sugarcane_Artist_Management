import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    console.log("[v0] Facebook post-type-engagement - Fetching all data")

    const result = await sql`
      SELECT 
        post_type,
        SUM(COALESCE(reactions, 0)) as total_reactions,
        SUM(COALESCE(comments, 0)) as total_comments,
        SUM(COALESCE(shares, 0)) as total_shares,
        SUM(COALESCE(reach, 0)) as total_reach,
        SUM(COALESCE(reactions, 0) + COALESCE(comments, 0) + COALESCE(shares, 0)) as total_engagement
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
        AND post_type IS NOT NULL
        AND TRIM(post_type) != ''
      GROUP BY post_type
      ORDER BY total_engagement DESC
    `

    console.log("[v0] Facebook post-type-engagement - Found", result.length, "post types:", result)

    return NextResponse.json({
      post_type_engagement: result.map((row: any) => {
        const reach = Number(row.total_reach) || 0
        const reactions = Number(row.total_reactions) || 0
        const comments = Number(row.total_comments) || 0
        const shares = Number(row.total_shares) || 0

        return {
          post_type: row.post_type,
          reactions_rate: reach > 0 ? (reactions / reach) * 100 : 0,
          comments_rate: reach > 0 ? (comments / reach) * 100 : 0,
          shares_rate: reach > 0 ? (shares / reach) * 100 : 0,
          total_engagement_rate: reach > 0 ? ((reactions + comments + shares) / reach) * 100 : 0,
        }
      }),
    })
  } catch (error) {
    console.error("Error fetching Facebook post type engagement:", error)
    return NextResponse.json({ error: "Failed to fetch post type engagement" }, { status: 500 })
  }
}
