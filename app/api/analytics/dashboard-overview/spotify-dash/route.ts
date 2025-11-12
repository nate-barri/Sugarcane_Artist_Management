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

    return NextResponse.json({ total_followers: Number(row.total_followers) });
  } catch (err) {
    console.error("[spotify-dash] Error:", err);
    return NextResponse.json(
      { error: "Failed to fetch total followers" },
      { status: 500 }
    );
  }
}
