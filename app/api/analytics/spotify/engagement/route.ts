// api/analytics/spotify/engagement/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils"; // Assuming you have a utility function to execute queries

export async function GET() {
  try {
    const query = `
      SELECT followers, streams, listeners 
      FROM spotify_stats
      ORDER BY date DESC
      LIMIT 100
    `;

    const result = await executeQuery(query);

    return NextResponse.json({
      engagement_distribution: result.map((row: any) => ({
        followers: Number(row.followers),
        streams: Number(row.streams),
        listeners: Number(row.listeners),
      })),
    });
  } catch (error) {
    console.error("[v0] Engagement analysis error:", error);
    return NextResponse.json({ error: "Failed to fetch engagement data" }, { status: 500 });
  }
}
