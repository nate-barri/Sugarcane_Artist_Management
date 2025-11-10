import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    console.log("[v0] Facebook temporal - Fetching all temporal data")

    // Posts per month
    const monthlyPosts = await sql`
      SELECT 
        year,
        month,
        COUNT(*) as post_count
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
      GROUP BY year, month
      ORDER BY year, month
    `

    const dayOfWeek = await sql`
      SELECT 
        EXTRACT(DOW FROM publish_time) as day_num,
        CASE EXTRACT(DOW FROM publish_time)
          WHEN 0 THEN 'Sunday'
          WHEN 1 THEN 'Monday'
          WHEN 2 THEN 'Tuesday'
          WHEN 3 THEN 'Wednesday'
          WHEN 4 THEN 'Thursday'
          WHEN 5 THEN 'Friday'
          WHEN 6 THEN 'Saturday'
        END as day_name,
        AVG(COALESCE(reactions, 0) + COALESCE(comments, 0) + COALESCE(shares, 0)) as avg_engagement
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
      GROUP BY day_num, day_name
      ORDER BY day_num
    `

    const hourlyReachData = await sql`
      SELECT 
        EXTRACT(HOUR FROM publish_time) as hour,
        AVG(COALESCE(reach, 0)) as avg_reach
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
      GROUP BY hour
      ORDER BY hour
    `

    const hourlyMap = new Map<number, number>()
    hourlyReachData.forEach((row: any) => {
      hourlyMap.set(Number(row.hour), Number(row.avg_reach) || 0)
    })

    const hourlyReach = []
    for (let hour = 0; hour < 24; hour++) {
      hourlyReach.push({
        hour,
        avg_reach: hourlyMap.get(hour) || 0,
      })
    }

    const monthlyReach = await sql`
      SELECT 
        year,
        month,
        SUM(COALESCE(reach, 0)) as total_reach
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
      GROUP BY year, month
      ORDER BY year, month
    `

    console.log("[v0] Facebook temporal - day_of_week:", dayOfWeek.length, "rows")
    console.log("[v0] Facebook temporal - hourly_reach:", hourlyReach.length, "hours")

    return NextResponse.json({
      monthly_posts: monthlyPosts.map((row: any) => ({
        year: Number(row.year),
        month: Number(row.month),
        post_count: Number(row.post_count),
      })),
      day_of_week: dayOfWeek.map((row: any) => ({
        day_name: row.day_name,
        avg_engagement: Number(row.avg_engagement) || 0,
      })),
      hourly_reach: hourlyReach,
      monthly_reach: monthlyReach.map((row: any) => ({
        year: Number(row.year),
        month: Number(row.month),
        total_reach: Number(row.total_reach) || 0,
      })),
    })
  } catch (error) {
    console.error("Error fetching Facebook temporal data:", error)
    return NextResponse.json({ error: "Failed to fetch temporal data" }, { status: 500 })
  }
}
