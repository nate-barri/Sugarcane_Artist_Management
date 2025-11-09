// app/api/analytics/spotify/daily-listeners-with-releases/route.ts
import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

// Returns:
// { daily: [{ date: '2025-05-01', listeners: 1234 }, ...],
//   releases: [{ date: '2025-04-28', song: 'Set You Free' }, ...] }
export async function GET() {
  try {
    // Total DAILY listeners across all songs
    const dailySql = `
      SELECT
        sts."date"::date AS date,
        SUM(COALESCE(sts.listeners, 0))::bigint AS listeners
      FROM spotify_stats AS sts
      GROUP BY 1
      ORDER BY 1
    `;

    // Release dates for markers
    const releasesSql = `
      SELECT
        release_date::date AS date,
        song
      FROM spotify_songs
      WHERE release_date IS NOT NULL
      ORDER BY release_date
    `;

    const daily = await executeQuery<{ date: string; listeners: number }>(dailySql);
    const releases = await executeQuery<{ date: string; song: string }>(releasesSql);

    return NextResponse.json({ daily, releases });
  } catch (err: any) {
    console.error("daily-listeners-with-releases error:", err);
    return NextResponse.json({ error: err?.message ?? "Internal error" }, { status: 500 });
  }
}
