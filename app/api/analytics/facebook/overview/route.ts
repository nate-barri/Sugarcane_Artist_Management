import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const startDate = request.nextUrl.searchParams.get("startDate") || "2021-01-01"
    const endDate = request.nextUrl.searchParams.get("endDate") || "2025-12-31"

    const result = await sql`
      SELECT 
        COUNT(*) as total_posts,
        SUM(CASE WHEN reach::text = 'NaN' THEN 0 ELSE COALESCE(reach::numeric, 0) END)::bigint as total_reach,
        SUM(CASE WHEN reactions::text = 'NaN' THEN 0 ELSE COALESCE(reactions::numeric, 0) END)::bigint as total_reactions,
        SUM(CASE WHEN comments::text = 'NaN' THEN 0 ELSE COALESCE(comments::numeric, 0) END)::bigint as total_comments,
        SUM(CASE WHEN shares::text = 'NaN' THEN 0 ELSE COALESCE(shares::numeric, 0) END)::bigint as total_shares
      FROM facebook_data_set
      WHERE publish_time >= ${startDate}::date
        AND publish_time <= ${endDate}::date
    `

    const row = result[0]
    const total_posts = Number(row?.total_posts) || 0
    const total_reach = Number(row?.total_reach) || 0
    const total_reactions = Number(row?.total_reactions) || 0
    const total_comments = Number(row?.total_comments) || 0
    const total_shares = Number(row?.total_shares) || 0
    const total_engagement = total_reactions + total_comments + total_shares
    const engagement_rate = total_reach > 0 ? (total_engagement / total_reach) * 100 : 0

    return NextResponse.json({
      total_posts,
      total_reach,
      total_engagement,
      total_reactions,
      total_comments,
      total_shares,
      engagement_rate,
    })
  } catch (error) {
    console.error("Error fetching Facebook overview:", error)
    return NextResponse.json({ error: "Failed to fetch overview data" }, { status: 500 })
  }
}
