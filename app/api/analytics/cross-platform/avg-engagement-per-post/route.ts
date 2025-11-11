// app/api/analytics/cross-platform/avg-engagement-per-post/route.ts
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
          post_id,
          COALESCE(total_engagement, 0) AS total_engagement
        FROM unified_social_analytics
        WHERE post_date::date BETWEEN $1::date AND $2::date
      ),
      agg AS (
        SELECT
          platform_norm AS platform,
          SUM(total_engagement)::bigint AS total_eng,
          COUNT(DISTINCT post_id)::bigint AS posts
        FROM base
        GROUP BY 1
      )
      SELECT
        platform,
        CASE WHEN posts > 0
          THEN ROUND((total_eng::numeric / posts), 0)
          ELSE 0
        END::bigint AS avg_per_post
      FROM agg
      ORDER BY
        CASE platform
          WHEN 'Facebook' THEN 1
          WHEN 'TikTok'   THEN 2
          WHEN 'YouTube'  THEN 3
          ELSE 99
        END;
    `

    const rows = await executeQuery(sql, [startDate, endDate])

    const byPlat: Record<string, number> =
      Object.fromEntries((rows as any[]).map(r => [r.platform, Number(r.avg_per_post) || 0]))

    const data = [
      { platform: "Facebook", avg: byPlat["Facebook"] || 0 },
      { platform: "TikTok",   avg: byPlat["TikTok"]   || 0 },
      { platform: "YouTube",  avg: byPlat["YouTube"]  || 0 },
    ]

    return NextResponse.json({ data }, { status: 200 })
  } catch (err: any) {
    return NextResponse.json(
      { error: err?.message || "Failed to fetch avg engagement per post" },
      { status: 500 }
    )
  }
}
