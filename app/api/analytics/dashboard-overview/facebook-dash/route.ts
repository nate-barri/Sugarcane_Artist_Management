import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils";

export async function GET() {
    try {
        const sqlFB = `
        SELECT
            SUM(CASE WHEN shares::text = 'NaN' THEN 0 ELSE shares END) AS total_shares,
            SUM(CASE WHEN reactions::text = 'NaN' THEN 0 ELSE reactions END) AS total_reactions,
            SUM(CASE WHEN comments::text = 'NaN' THEN 0 ELSE comments END) AS total_comments
        FROM facebook_data_set;
        `;

        const [row] = await executeQuery(sqlFB);

        const totalMetaEngagement =
            Number(row.total_reactions) +
            Number(row.total_shares) +
            Number(row.total_comments);

        const sqlFbReach =
            `
        SELECT
            SUM(CASE WHEN reach::text = 'NaN' THEN 0 ELSE reach END) AS total_reach
        FROM facebook_data_set;
        `;
        const [row2] = await executeQuery(sqlFbReach);
        const metaReach = Number(row2.total_reach);

        return NextResponse.json({ total_meta_engagement: totalMetaEngagement, total_meta_reach: metaReach });
    } catch (err) {
        console.error("[meta-dash] Error:", err);
        return NextResponse.json(
            { error: "Failed to fetch" },
            { status: 500 }
        );
    }
}
