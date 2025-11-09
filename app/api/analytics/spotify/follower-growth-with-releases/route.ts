// app/api/analytics/spotify/follower-growth-with-releases/route.ts
import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET() {
  try {
    // 1) Followers (use MAX per day in case multiple rows exist)
    //    Window widened to 5 YEARS so early releases are inside the range.
    const followersSql = `
      SELECT
        date::date AS date,
        COALESCE(MAX(followers), 0) AS total_followers
      FROM spotify_stats
      WHERE date >= NOW()::date - INTERVAL '5 years'
      GROUP BY date::date
      ORDER BY date::date ASC;
    `;

    // 2) Song releases (also widen to 5 YEARS)
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

    const follower_growth = followerRows.map(r => ({
      date: new Date(r.date).toISOString().slice(0, 10),
      total_followers: Number(r.total_followers) || 0,
    }));

    const song_releases = releaseRows.map(r => ({
      title: r.title,
      release_date: new Date(r.release_date).toISOString().slice(0, 10),
    }));

    return NextResponse.json({ follower_growth, song_releases });
  } catch (err) {
    console.error("follower-growth-with-releases error:", err);
    return NextResponse.json(
      { error: "Failed to load follower growth with releases" },
      { status: 500 }
    );
  }
}
