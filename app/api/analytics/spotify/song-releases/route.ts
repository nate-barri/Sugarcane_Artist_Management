// api/analytics/spotify/song-releases/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils"; 

export async function GET() {
  try {
    // Step 1: Query to fetch song release dates and titles
    const queryReleases = `
      SELECT 
        release_date, 
        song
      FROM spotify_songs
      WHERE release_date >= CURRENT_DATE - INTERVAL '1 year' 
      ORDER BY release_date;
    `;
    const resultReleases = await executeQuery(queryReleases);

    // Step 2: Format the data for the frontend
    const releasesData = resultReleases.map((row: any) => ({
      release_date: row.release_date,
      song: row.song,
    }));

    // Return the data as JSON
    return NextResponse.json({
      song_releases: releasesData,
    });
  } catch (error) {
    console.error("[v0] Song releases error:", error);
    return NextResponse.json({ error: "Failed to fetch song releases" }, { status: 500 });
  }
}
