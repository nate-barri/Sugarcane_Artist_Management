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
            WHEN LOWER(TRIM(platform)) IN ('tiktok','tik tok','tt') THEN 'TikTok'
            WHEN LOWER(TRIM(platform)) IN ('facebook','meta','fb') THEN 'Facebook'
            WHEN LOWER(REPLACE(TRIM(platform),' ','')) IN ('youtube','yt') THEN 'YouTube'
            ELSE 'Other'
          END AS platform_norm,
          COALESCE(total_engagement, 0)::bigint AS engagement,
          (post_date)::date AS d
        FROM unified_social_analytics
      ),
      filtered AS (
        SELECT *
        FROM base
        /* include rows with NULL dates (Python includes them) */
        WHERE
          ($1::date IS NULL AND $2::date IS NULL)
          OR d IS NULL
          OR (d >= $1::date AND d <= $2::date)
      )
      SELECT platform_norm AS platform, SUM(engagement)::bigint AS total
      FROM filtered
      GROUP BY platform_norm
    `

    const rows: { platform: string; total: string | number }[] =
      await executeQuery(sql, [startDate ?? null, endDate ?? null])

    const totals: Record<string, number> = { TikTok: 0, Facebook: 0, YouTube: 0 }
    for (const r of rows) {
      const p = (r.platform || "").trim()
      const v = Number(r.total) || 0
      if (p in totals) totals[p as keyof typeof totals] = v
    }

    const data = [
      { platform: "Facebook", total: totals.Facebook },
      { platform: "TikTok",   total: totals.TikTok },
      { platform: "YouTube",  total: totals.YouTube },
    ]

    return NextResponse.json({ data, totals }, { status: 200 })
  } catch (e: any) {
    console.error("engagement-totals error:", e)
    return NextResponse.json(
      { error: "Failed to fetch engagement totals", detail: e?.message || "Unknown error" },
      { status: 500 }
    )
  }
}
