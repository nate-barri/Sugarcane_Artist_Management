// app/api/analytics/cross-platform/reach-totals/route.ts
import { NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const startDate = searchParams.get("startDate") || "1900-01-01"
    const endDate   = searchParams.get("endDate")   || "2100-12-31"

    const sql = `
      WITH base AS (
        SELECT
          CASE
            WHEN LOWER(platform) IN ('facebook','meta') THEN 'Facebook'
            WHEN LOWER(platform) IN ('tiktok','tik tok') THEN 'TikTok'
            WHEN LOWER(platform) IN ('youtube','yt') THEN 'YouTube'
            ELSE 'Other'
          END AS platform_norm,
          COALESCE(views_reach, 0) AS views_reach
        FROM unified_social_analytics
        WHERE post_date::date BETWEEN $1::date AND $2::date
      )
      SELECT
        platform_norm AS platform,
        SUM(views_reach)::bigint AS total
      FROM base
      GROUP BY 1
      ORDER BY
        CASE platform_norm
          WHEN 'Facebook' THEN 1
          WHEN 'TikTok'   THEN 2
          WHEN 'YouTube'  THEN 3
          ELSE 99
        END;
    `

    const rows = await executeQuery(sql, [startDate, endDate])

    const byPlat: Record<string, number> =
      Object.fromEntries((rows as any[]).map(r => [r.platform, Number(r.total) || 0]))

    const data = [
      { platform: "Facebook", total: byPlat["Facebook"] || 0 },
      { platform: "TikTok",   total: byPlat["TikTok"]   || 0 },
      { platform: "YouTube",  total: byPlat["YouTube"]  || 0 },
    ]

    return NextResponse.json({ data }, { status: 200 })
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "Failed to fetch reach totals" },
      { status: 500 }
    )
  }
}
