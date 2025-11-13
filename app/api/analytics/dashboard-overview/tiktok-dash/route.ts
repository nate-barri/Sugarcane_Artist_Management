import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils";

export async function GET() {
    try {
        // total
        const sqlSum = `
        SELECT
            SUM(likes) AS total_likes,
            SUM(shares) AS total_shares,
            SUM(comments_added) AS total_comments,
            SUM(saves) AS total_saves
        FROM tt_video_etl;
    `;
        const [row] = await executeQuery(sqlSum);

        const totalEngagementCalculator =
            (Number(row.total_likes) || 0) +
            (Number(row.total_shares) || 0) +
            (Number(row.total_comments) || 0) +
            (Number(row.total_saves) || 0);

        
        return NextResponse.json({ totalTiktokEngagement: totalEngagementCalculator });
    } catch (err) {
        console.error("[youtube-dash] Error:", err);
        return NextResponse.json(
            { error: "Failed to fetch" },
            { status: 500 }
        );
    }
}