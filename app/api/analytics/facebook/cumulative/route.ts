import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    console.log("[v0] Facebook cumulative - Fetching all data")

    const result = await sql`
      SELECT 
        publish_time,
        reach,
        reactions + comments + shares as total_engagement
      FROM facebook_data_set
      ORDER BY publish_time
    `

    let cumulativeReach = 0
    let cumulativeEngagement = 0

    const cumulativeData = result.map((row: any) => {
      cumulativeReach += Number(row.reach) || 0
      cumulativeEngagement += Number(row.total_engagement) || 0

      return {
        date: row.publish_time,
        cumulative_reach: cumulativeReach,
        cumulative_engagement_rate: cumulativeReach > 0 ? (cumulativeEngagement / cumulativeReach) * 100 : 0,
      }
    })

    return NextResponse.json({ cumulative: cumulativeData })
  } catch (error) {
    console.error("Error fetching Facebook cumulative data:", error)
    return NextResponse.json({ error: "Failed to fetch cumulative data" }, { status: 500 })
  }
}
