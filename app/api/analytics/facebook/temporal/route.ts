import { type NextRequest, NextResponse } from "next/server"
import { neon } from "@neondatabase/serverless"

const sql = neon(process.env.DATABASE_URL!)

export async function GET(request: NextRequest) {
  try {
    const startDate = request.nextUrl.searchParams.get("startDate") || "2021-01-01"
    const endDate = request.nextUrl.searchParams.get("endDate") || "2025-12-31"

    // Posts per month
    const monthlyPosts = await sql`
      SELECT 
        year,
        month,
        COUNT(*) as post_count
      FROM facebook_data_set
      WHERE publish_time >= ${startDate}::date
        AND publish_time <= ${endDate}::date
      GROUP BY year, month
      ORDER BY year, month
    `

    // Engagement by day of week
    const dayOfWeek = await sql`
      SELECT 
        EXTRACT(DOW FROM publish_time)::int as day_num,
        CASE EXTRACT(DOW FROM publish_time)
          WHEN 0 THEN 'Sunday'
          WHEN 1 THEN 'Monday'
          WHEN 2 THEN 'Tuesday'
          WHEN 3 THEN 'Wednesday'
          WHEN 4 THEN 'Thursday'
          WHEN 5 THEN 'Friday'
          WHEN 6 THEN 'Saturday'
        END as day_name,
        AVG((CASE WHEN reactions::text = 'NaN' THEN 0 ELSE COALESCE(reactions::numeric, 0) END) + 
            (CASE WHEN comments::text = 'NaN' THEN 0 ELSE COALESCE(comments::numeric, 0) END) + 
            (CASE WHEN shares::text = 'NaN' THEN 0 ELSE COALESCE(shares::numeric, 0) END))::numeric as avg_engagement
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
        AND publish_time >= ${startDate}::date
        AND publish_time <= ${endDate}::date
      GROUP BY day_num, day_name
      ORDER BY day_num
    `

    const hourlyReachData = await sql`
      SELECT 
        EXTRACT(HOUR FROM publish_time)::int as hour,
        AVG(CASE WHEN reach::text = 'NaN' THEN 0 ELSE COALESCE(reach::numeric, 0) END)::numeric as avg_reach
      FROM facebook_data_set
      WHERE publish_time IS NOT NULL
        AND publish_time >= ${startDate}::date
        AND publish_time <= ${endDate}::date
      GROUP BY hour
      ORDER BY hour
    `

    // Create a map of existing hourly data
    const hourlyMap = new Map<number, number>()
    hourlyReachData.forEach((row: any) => {
      hourlyMap.set(Number(row.hour), Number(row.avg_reach) || 0)
    })

    // Fill in all 24 hours (0-23)
    const hourlyReach = []
    for (let hour = 0; hour < 24; hour++) {
      hourlyReach.push({
        hour,
        avg_reach: hourlyMap.get(hour) || 0,
      })
    }

    // Monthly reach trend
    const monthlyReach = await sql`
      SELECT 
        year,
        month,
        SUM(CASE WHEN reach::text = 'NaN' THEN 0 ELSE COALESCE(reach::numeric, 0) END)::bigint as total_reach
      FROM facebook_data_set
      WHERE publish_time >= ${startDate}::date
        AND publish_time <= ${endDate}::date
      GROUP BY year, month
      ORDER BY year, month
    `

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
