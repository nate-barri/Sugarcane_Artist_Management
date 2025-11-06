// api/analytics/spotify/monthly/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils"; // Assuming you have a utility function to execute queries

export async function GET() {
  try {
    const query = `
      SELECT 
        EXTRACT(MONTH FROM date) AS month,
        EXTRACT(YEAR FROM date) AS year,
        SUM(streams) AS total_views
      FROM spotify_stats
      GROUP BY year, month
      ORDER BY year DESC, month DESC
    `;

    const result = await executeQuery(query);

    return NextResponse.json({
      monthly: result.map((row: any) => ({
        month: `${String(row.year)}-${String(row.month).padStart(2, "0")}`,
        total_views: Number(row.total_views),
      })),
    });
  } catch (error) {
    console.error("[v0] Monthly views error:", error);
    return NextResponse.json({ error: "Failed to fetch monthly views data" }, { status: 500 });
  }
}
