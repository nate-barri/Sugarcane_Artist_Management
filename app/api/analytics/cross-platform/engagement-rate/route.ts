import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export const runtime = "nodejs"
export const revalidate = 0
export const dynamic = "force-dynamic"

export async function GET(req: Request) {
  try {
    const { searchParams } = new URL(req.url)
    const startDate = searchParams.get("startDate")
    const endDate   = searchParams.get("endDate")

    // Define engagement and denominator (prefer impressions, then views_reach)
    const sql = `
      WITH base AS (
        SELECT
          CASE
            WHEN LOWER(TRIM(platform)) IN ('tiktok','tik tok','tt') THEN 'TikTok'
            WHEN LOWER(TRIM(platform)) IN ('facebook','meta','fb') THEN 'Facebook'
            WHEN LOWER(REPLACE(TRIM(platform),' ','')) IN ('youtube','yt') THEN 'YouTube'
            ELSE 'Other'
          END AS platform_norm,
          COALESCE(total_engagement,
                   COALESCE(likes_reactions,0)
                   + COALESCE(shares,0)
                   + COALESCE(comments,0)
                   + COALESCE(saves,0))::bigint AS engagement,
          -- denominator for rate: impressions first, else views/reach
          COALESCE(NULLIF(impressions,0), NULLIF(views_reach,0), 0)::bigint AS denom,
          (post_date)::date AS d
        FROM unified_social_analytics
      ),
      filtered AS (
        SELECT *
        FROM base
        WHERE
          ($1::date IS NULL AND $2::date IS NULL)
          OR d IS NULL
          OR (d >= $1::date AND d <= $2::date)
      ),
      agg AS (
        SELECT platform_norm AS platform,
               SUM(engagement)::numeric AS engagement_sum,
               SUM(denom)::numeric      AS denom_sum
        FROM filtered
        GROUP BY platform_norm
      )
      SELECT
        platform,
        engagement_sum,
        denom_sum,
        CASE WHEN denom_sum > 0
          THEN (engagement_sum / denom_sum) * 100
          ELSE 0
        END AS rate_pct
      FROM agg
    `

    const rows: { platform: string; engagement_sum: string | number; denom_sum: string | number; rate_pct: string | number }[] =
      await executeQuery(sql, [startDate ?? null, endDate ?? null])

    // keep only the 3 platforms; return in the order you want to render
    const byPlat: Record<string, number> = { Facebook: 0, TikTok: 0, YouTube: 0 }
    for (const r of rows) {
      const p = (r.platform || "").trim()
      if (p in byPlat) byPlat[p] = Number(r.rate_pct) || 0
    }

    const data = [
      { platform: "Facebook", rate: byPlat.Facebook },
      { platform: "TikTok",   rate: byPlat.TikTok },
      { platform: "YouTube",  rate: byPlat.YouTube },
    ]

    return NextResponse.json({ data }, { status: 200 })
  } catch (e: any) {
    console.error("engagement-rate error:", e)
    return NextResponse.json(
      { error: "Failed to compute engagement rate", detail: e?.message || "Unknown error" },
      { status: 500 }
    )
  }
}
