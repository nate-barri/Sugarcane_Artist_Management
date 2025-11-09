// app/api/analytics/spotify/streams-growth-with-releases/route.ts
import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

// Returns daily % growth from the first non-zero total streams (baseline),
// plus song releases for markers.
export async function GET() {
  try {
    // 1) Daily total streams (sum over all songs per date)
    const streamsSql = `
      SELECT
        date::date AS date,
        COALESCE(SUM(streams), 0) AS total_streams
      FROM spotify_stats
      WHERE date >= NOW()::date - INTERVAL '5 years'
      GROUP BY date::date
      ORDER BY date::date ASC;
    `

    // 2) Releases (for markers)
    const releasesSql = `
      SELECT
        song AS title,
        release_date::date AS release_date
      FROM spotify_songs
      WHERE release_date IS NOT NULL
        AND release_date >= NOW()::date - INTERVAL '5 years'
      ORDER BY release_date::date ASC;
    `

    const streamRows: any[] = await executeQuery(streamsSql)
    const releaseRows: any[] = await executeQuery(releasesSql)

    // Compute baseline = first non-zero total_streams (fallback to first row or 1 to avoid /0)
    let baseline = 0
    for (const r of streamRows) {
      const v = Number(r.total_streams) || 0
      if (v > 0) { baseline = v; break }
    }
    if (baseline === 0) baseline = Number(streamRows[0]?.total_streams || 1)

    const streams_growth = streamRows.map(r => {
      const date = new Date(r.date).toISOString().slice(0, 10)
      const total = Number(r.total_streams) || 0
      const growth_pct = ((total - baseline) / baseline) * 100
      return { date, growth_pct }
    })

    const song_releases = releaseRows.map(r => ({
      title: r.title,
      release_date: new Date(r.release_date).toISOString().slice(0, 10),
    }))

    return NextResponse.json({ streams_growth, song_releases })
  } catch (err) {
    console.error("streams-growth-with-releases error:", err)
    return NextResponse.json(
      { error: "Failed to load streams growth with releases" },
      { status: 500 }
    )
  }
}
