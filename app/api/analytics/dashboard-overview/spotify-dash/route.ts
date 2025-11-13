// app/api/dashboard-overview/spotify-dash/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils";

export async function GET() {
  try {
    // Option 1: highest followers ever recorded
    const sqlMax = `
      SELECT COALESCE(MAX(followers), 0) AS total_followers
      FROM spotify_stats
    `;
    const [row] = await executeQuery(sqlMax);

    // Option 2 (alternative): followers on the latest date only
    // const sqlLatest = `
    //   SELECT COALESCE(followers, 0) AS total_followers
    //   FROM spotify_stats
    //   WHERE date = (SELECT MAX(date) FROM spotify_stats)
    //   ORDER BY followers DESC
    //   LIMIT 1
    // `;
    // const [row] = await executeQuery(sqlLatest);

    // For Total Engagement, Spotify
    const sqlEngagement = `
      SELECT
        SUM(listeners) AS total_listeners,
        SUM(streams) AS total_streams
      FROM spotify_stats
    `
    const [row2] = await executeQuery(sqlEngagement)
    const totalSpotifyEngagementCalculation =
      (Number(row2.total_listeners) || 0) +
      (Number(row2.total_streams) || 0);

    const sqlGrowth = `
      WITH ordered_songs AS (
        SELECT
          song,
          streams,
          LAG(streams) OVER (ORDER BY release_date ASC) AS prev_streams
        FROM spotify_songs
      )
      SELECT
        ((streams::numeric - prev_streams::numeric) / prev_streams::numeric) * 100 AS pct_growth
      FROM ordered_songs
      WHERE prev_streams IS NOT NULL AND prev_streams != 0;
    `;

    const rows: { pct_growth: string | null }[] = await executeQuery(sqlGrowth);

    const growthValues = rows
      .map(r => r.pct_growth !== null ? Number(r.pct_growth) : null)
      .filter(v => v !== null) as number[];

    // Calculate average growth
    const avgGrowth =
      growthValues.length > 0
        ? growthValues.reduce((acc, val) => acc + val, 0) / growthValues.length
        : 0;

    const avgGrowthRounded = Number(avgGrowth.toFixed(2))
    return NextResponse.json({ total_followers: Number(row.total_followers), totalSpotifyEngagement: totalSpotifyEngagementCalculation, avg_growth: avgGrowthRounded });
  } catch (err) {
    console.error("[spotify-dash] Error:", err);
    return NextResponse.json(
      { error: "Failed to fetch total followers" },
      { status: 500 }
    );
  }
}
