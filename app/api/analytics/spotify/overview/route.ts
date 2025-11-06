// api/analytics/spotify/overview/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils"; 

export async function GET() {
  try {
    const query = `
      SELECT 
        SUM(streams) AS total_streams,
        SUM(followers) AS total_followers,
        SUM(listeners) AS total_listeners,
        COUNT(DISTINCT song) AS top_tracks_count
      FROM spotify_stats
      WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    `;
    const result = await executeQuery(query);

    console.log(result);  // Log the result to check if it's correct

    return NextResponse.json({
      overview: result[0], // Ensure this returns the correct data
    });
  } catch (error) {
    console.error("[v0] Overview error:", error);
    return NextResponse.json({ error: "Failed to fetch Spotify overview data" }, { status: 500 });
  }
}
