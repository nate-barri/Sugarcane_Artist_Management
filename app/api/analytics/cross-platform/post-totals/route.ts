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

    // Count DISTINCT posts per platform.
    // Includes rows with NULL post_date (to match your Python "all-time" behavior).
    const sql = `
      WITH base AS (
        SELECT
          CASE
            WHEN LOWER(TRIM(platform)) IN ('tiktok','tik tok','tt') THEN 'TikTok'
            WHEN LOWER(TRIM(platform)) IN ('facebook','meta','fb') THEN 'Facebook'
            WHEN LOWER(REPLACE(TRIM(platform),' ','')) IN ('youtube','yt') THEN 'YouTube'
            ELSE 'Other'
          END AS platform_norm,
          NULLIF(TRIM(post_id), '') AS post_id,
          (post_date)::date AS d
        FROM unified_social_analytics
      ),
      filtered AS (
        SELECT *
        FROM base
        WHERE
          ($1::date IS NULL AND $2::date IS NULL)  -- no range: include all
          OR d IS NULL                              -- include NULL dates even when range is provided
          OR (d >= $1::date AND d <= $2::date)
      ),
      dedup AS (  -- avoid counting the same post multiple times
        SELECT DISTINCT platform_norm, post_id
        FROM filtered
        WHERE post_id IS NOT NULL
      )
      SELECT platform_norm AS platform, COUNT(*)::bigint AS posts
      FROM dedup
      GROUP BY platform_norm
    `

    const rows: { platform: string; posts: string | number }[] =
      await executeQuery(sql, [startDate ?? null, endDate ?? null])

    const totals: Record<"Facebook"|"TikTok"|"YouTube", number> = {
      Facebook: 0, TikTok: 0, YouTube: 0
    }
    for (const r of rows) {
      const p = (r.platform || "").trim()
      const v = Number(r.posts) || 0
      if (p in totals) totals[p as keyof typeof totals] = v
    }

    const data = [
      { platform: "Facebook", posts: totals.Facebook },
      { platform: "TikTok",   posts: totals.TikTok },
      { platform: "YouTube",  posts: totals.YouTube },
    ]

    return NextResponse.json({ data, totals }, { status: 200 })
  } catch (e: any) {
    console.error("posts-totals error:", e)
    return NextResponse.json(
      { error: "Failed to fetch posts totals", detail: e?.message || "Unknown error" },
      { status: 500 }
    )
  }
}
