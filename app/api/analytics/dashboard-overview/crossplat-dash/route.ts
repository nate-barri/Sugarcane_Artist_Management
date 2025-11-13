import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils";

export async function GET() {
    try {
        // total
        const sqlViews = `
        SELECT
            (SELECT SUM(CASE WHEN views::text = 'NaN' THEN 0 ELSE views END) FROM tt_video_etl) AS ttViews,
            (SELECT SUM(CASE WHEN views::text = 'NaN' THEN 0 ELSE views END) FROM yt_video_etl) AS ytViews
        `;
        const [row] = await executeQuery(sqlViews);

        const totalViewsCalculator =
            (Number(row.ttviews) || 0) +
            (Number(row.ytviews) || 0);

        return NextResponse.json({ totalViews: totalViewsCalculator, viewsDebug: row });
    } catch (err) {
        console.error("[youtube-dash] Error:", err);
        return NextResponse.json(
            { error: "Failed to fetch" },
            { status: 500 }
        );
    }
}