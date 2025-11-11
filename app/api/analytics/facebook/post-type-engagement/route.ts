import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const startDate = searchParams.get("startDate") || "2021-01-01"
    const endDate = searchParams.get("endDate") || "2025-12-31"

    console.log("[v0] Facebook post-type-engagement - Date range:", startDate, "to", endDate)

    const result = await sql`
      SELECT 
        post_type,
        SUM(CASE WHEN reactions::text = 'NaN' THEN 0 ELSE COALESCE(reactions::numeric, 0) END)::bigint as total_reactions,
        SUM(CASE WHEN comments::text = 'NaN' THEN 0 ELSE COALESCE(comments::numeric, 0) END)::bigint as total_comments,
        SUM(CASE WHEN shares::text = 'NaN' THEN 0 ELSE COALESCE(shares::numeric, 0) END)::bigint as total_shares,
        SUM(CASE WHEN reach::text = 'NaN' THEN 0 ELSE COALESCE(reach::numeric, 0) END)::bigint as total_reach
      FROM facebook_data_set
      WHERE publish_time >= ${startDate}::date 
        AND publish_time <= ${endDate}::date
        AND publish_time IS NOT NULL
        AND post_type IS NOT NULL
        AND TRIM(post_type) != ''
      GROUP BY post_type
      ORDER BY post_type
    `

    console.log("[v0] Facebook post-type-engagement - Found", result.length, "post types:", result)

    return NextResponse.json({
      post_type_engagement: result.map((row: any) => {
        const reach = Number(row.total_reach) || 1
        const reactions = Number(row.total_reactions) || 0
        const comments = Number(row.total_comments) || 0
        const shares = Number(row.total_shares) || 0

        return {
          post_type: row.post_type,
          reactions_rate: reach > 0 ? (reactions / reach) * 100 : 0,
          comments_rate: reach > 0 ? (comments / reach) * 100 : 0,
          shares_rate: reach > 0 ? (shares / reach) * 100 : 0,
        }
      }),
    })
  } catch (error) {
    console.error("Error fetching Facebook post type engagement:", error)
    return NextResponse.json({ error: "Failed to fetch post type engagement" }, { status: 500 })
  }
}
