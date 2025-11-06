// api/analytics/spotify/top-tracks/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils"; // Assuming you have a utility function to execute queries

export async function GET() {
  try {
    const query = `
      SELECT song, streams 
      FROM spotify_songs
      ORDER BY streams DESC
      LIMIT 10
    `;

    const result = await executeQuery(query);

    return NextResponse.json({
      tracks: result.map((row: any) => ({
        track: row.song,
        streams: Number(row.streams),
      })),
    });
  } catch (error) {
    console.error("[v0] Top tracks error:", error);
    return NextResponse.json({ error: "Failed to fetch top tracks data" }, { status: 500 });
  }
}
