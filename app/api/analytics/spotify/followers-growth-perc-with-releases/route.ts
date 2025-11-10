import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

// Uses your tables:
// - spotify_stats(date, followers, ...)
// - spotify_songs(song, release_date, ...)
//
// Logic:
// 1) MAX(followers) per day (since stats can repeat per song)
// 2) Baseline = earliest non-zero followers value
// 3) growth_pct = ((followers - baseline) / baseline) * 100

export async function GET() {
  try {
    const followersSql = `
      SELECT
        date::date AS date,
        COALESCE(MAX(followers), 0) AS total_followers
      FROM spotify_stats
      WHERE date >= NOW()::date - INTERVAL '5 years'
      GROUP BY date::date
      ORDER BY date::date ASC;
    `;

    const releasesSql = `
      SELECT
        song AS title,
        release_date::date AS release_date
      FROM spotify_songs
      WHERE release_date IS NOT NULL
        AND release_date >= NOW()::date - INTERVAL '5 years'
      ORDER BY release_date::date ASC;
    `;

    const followerRows: any[] = await executeQuery(followersSql);
    const releaseRows: any[]  = await executeQuery(releasesSql);

    // baseline = first non-zero followers (fallback to first value or 1 to avoid /0)
    let baseline = 0;
    for (const r of followerRows) {
      const v = Number(r.total_followers) || 0;
      if (v > 0) { baseline = v; break; }
    }
    if (baseline === 0) baseline = Number(followerRows[0]?.total_followers || 1);

    const followers_growth = followerRows.map(r => {
      const date = new Date(r.date).toISOString().slice(0, 10);
      const total = Number(r.total_followers) || 0;
      const growth_pct = ((total - baseline) / baseline) * 100;
      return { date, growth_pct };
    });

    const song_releases = releaseRows.map(r => ({
      title: r.title,
      release_date: new Date(r.release_date).toISOString().slice(0, 10),
    }));

    return NextResponse.json({ followers_growth, song_releases });
  } catch (err) {
    console.error("followers-growth-with-releases error:", err);
    return NextResponse.json(
      { error: "Failed to load followers growth with releases" },
      { status: 500 }
    );
  }
}
