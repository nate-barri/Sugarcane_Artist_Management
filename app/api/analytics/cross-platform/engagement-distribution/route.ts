import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET() {
  try {
    // Mirror cross_plat.py: SUM(total_engagement) by platform, no date filter.
    const sql = `
      SELECT
        TRIM(platform) AS platform,
        SUM(total_engagement)::bigint AS total_engagement
      FROM unified_social_analytics
      GROUP BY TRIM(platform)
      ORDER BY total_engagement DESC
    `

    const rows: { platform: string; total_engagement: string | number }[] =
      await executeQuery(sql)

    // Keep only the three platforms used on the chart
    const totals: Record<string, number> = { TikTok: 0, Facebook: 0, YouTube: 0 }
    for (const r of rows) {
      const p = (r.platform || "").trim()
      if (p in totals) totals[p] = Number(r.total_engagement) || 0
    }

    const grand = totals.TikTok + totals.Facebook + totals.YouTube
    const pct = (n: number) => (grand > 0 ? (n / grand) * 100 : 0)

    return NextResponse.json({
      totals,
      distribution: [
        { platform: "YouTube",  value: pct(totals.YouTube)  },
        { platform: "Facebook", value: pct(totals.Facebook) },
        { platform: "TikTok",   value: pct(totals.TikTok)   },
      ],
      note: "Matches cross_plat.py: SUM(total_engagement) by platform, no date filter."
    })
  } catch (e: any) {
    console.error(e)
    return NextResponse.json({ error: "Failed to compute engagement distribution" }, { status: 500 })
  }
}
