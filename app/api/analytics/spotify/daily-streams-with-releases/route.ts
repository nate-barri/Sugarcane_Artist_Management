// api/analytics/spotify/daily-streams-with-releases/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils"; 

export async function GET() {
  try {
    // Step 1: Query to fetch daily streams
    const queryStreams = `
      SELECT 
        date, 
        streams 
      FROM spotify_stats 
      WHERE date >= CURRENT_DATE - INTERVAL '1 year' 
      ORDER BY date;
    `;
    const resultStreams = await executeQuery(queryStreams);

    // Step 2: Query to fetch song release dates and titles
    const queryReleases = `
      SELECT 
        release_date, 
        song 
      FROM spotify_songs
      WHERE release_date >= CURRENT_DATE - INTERVAL '1 year' 
      ORDER BY release_date;
    `;
    const resultReleases = await executeQuery(queryReleases);

    // Format the data for the frontend
    const streamsData = resultStreams.map((row: any) => ({
      date: row.date,
      streams: Number(row.streams),
    }));

    // For song releases, we only need the release date and song title
    const releasesData = resultReleases.map((row: any) => ({
      release_date: row.release_date,
      song: row.song,
    }));

    // Return the data as JSON
    return NextResponse.json({
      daily_streams: streamsData,
      song_releases: releasesData,
    });
  } catch (error) {
    console.error("[v0] Daily streams with releases error:", error);
    return NextResponse.json({ error: "Failed to fetch daily streams and song releases" }, { status: 500 });
  }
}
