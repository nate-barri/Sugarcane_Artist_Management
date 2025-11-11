import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const startDate = request.nextUrl.searchParams.get("startDate") || "2021-01-01"
    const endDate = request.nextUrl.searchParams.get("endDate") || "2025-12-31"

    const result = await sql`
      SELECT 
        publish_time::date as date,
        SUM(CASE WHEN reach::text = 'NaN' THEN 0 ELSE COALESCE(reach::numeric, 0) END)::bigint as daily_reach,
        SUM(
          (CASE WHEN reactions::text = 'NaN' THEN 0 ELSE COALESCE(reactions::numeric, 0) END) +
          (CASE WHEN comments::text = 'NaN' THEN 0 ELSE COALESCE(comments::numeric, 0) END) +
          (CASE WHEN shares::text = 'NaN' THEN 0 ELSE COALESCE(shares::numeric, 0) END)
        )::bigint as daily_engagement
      FROM facebook_data_set
      WHERE publish_time >= ${startDate}::date
        AND publish_time <= ${endDate}::date
      GROUP BY publish_time::date
      ORDER BY publish_time::date
    `

    let cumulativeReach = 0
    let cumulativeEngagement = 0

    const cumulativeData = result.map((row: any) => {
      cumulativeReach += Number(row.daily_reach) || 0
      cumulativeEngagement += Number(row.daily_engagement) || 0

      const engagementRate = cumulativeReach > 0 ? (cumulativeEngagement / cumulativeReach) * 100 : 0

      return {
        date: row.date,
        cumulative_reach: cumulativeReach,
        cumulative_engagement_rate: engagementRate,
      }
    })

    return NextResponse.json({ cumulative: cumulativeData })
  } catch (error) {
    console.error("Error fetching Facebook cumulative data:", error)
    return NextResponse.json({ error: "Failed to fetch cumulative data" }, { status: 500 })
  }
}
