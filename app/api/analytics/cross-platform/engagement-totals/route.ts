// app/api/analytics/cross-platform/engagement-totals/route.ts
import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export const runtime = "nodejs"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const startDate = searchParams.get("startDate")
    const endDate   = searchParams.get("endDate")

    const sql = `
      WITH base AS (
        SELECT
          CASE
            WHEN LOWER(TRIM(platform)) IN ('facebook','meta','fb') THEN 'Facebook'
            WHEN LOWER(TRIM(platform)) IN ('tiktok','tik tok','tt') THEN 'TikTok'
            WHEN LOWER(REPLACE(TRIM(platform),' ','')) IN ('youtube','yt') THEN 'YouTube'
            ELSE 'Other'
          END AS platform_norm,
          COALESCE(total_engagement, 0)::bigint AS engagement,
          post_date::date AS d
        FROM unified_social_analytics
      ),
      filtered AS (
        SELECT *
        FROM base
        /* include rows with NULL dates (to mirror your Python behavior) */
        WHERE
          ($1::date IS NULL AND $2::date IS NULL)
          OR d IS NULL
          OR (d >= $1::date AND d <= $2::date)
      )
      SELECT platform_norm AS platform,
             SUM(engagement)::bigint AS total
      FROM filtered
      WHERE platform_norm IN ('Facebook','TikTok','YouTube')
      GROUP BY platform_norm;
    `

    const rows = await executeQuery(sql, [startDate ?? null, endDate ?? null]) as Array<{platform:string,total:string|number}>

    // Normalize to always return all three platforms for your BarChart
    const map = new Map(rows.map(r => [r.platform, Number(r.total) || 0]))
    const data = [
      { platform: "Facebook", total: map.get("Facebook") ?? 0 },
      { platform: "TikTok",   total: map.get("TikTok")   ?? 0 },
      { platform: "YouTube",  total: map.get("YouTube")  ?? 0 },
    ]

    return NextResponse.json({ data }, { status: 200 })
  } catch (e: any) {
    return NextResponse.json(
      { error: "Failed to fetch engagement totals", detail: e?.message || "Unknown error" },
      { status: 500 }
    )
  }
}
