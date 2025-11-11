// app/api/analytics/cross-platform/post-totals/route.ts
import { NextRequest, NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(req: NextRequest) {
  const t0 = Date.now()
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
          post_id
        FROM unified_social_analytics
        WHERE post_date::date BETWEEN $1::date AND $2::date
      )
      SELECT platform_norm AS platform, COUNT(DISTINCT post_id)::bigint AS total
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
    const byPlat = Object.fromEntries((rows as any[]).map(r => [r.platform, Number(r.total) || 0]))

    const data = [
      { platform: "Facebook", total: byPlat["Facebook"] || 0 },
      { platform: "TikTok",   total: byPlat["TikTok"]   || 0 },
      { platform: "YouTube",  total: byPlat["YouTube"]  || 0 },
    ]

    console.log("[post-totals]", { startDate, endDate, rows, ms: Date.now() - t0 })
    return NextResponse.json({ data }, { status: 200 })
  } catch (err: any) {
    console.error("[post-totals][ERROR]", err)
    return NextResponse.json(
      { error: err?.message || "Failed to fetch post totals" },
      { status: 500 }
    )
  }
}
