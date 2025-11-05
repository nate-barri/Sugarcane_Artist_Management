import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET() {
  try {
    const query = `
      SELECT 
        views,
        likes,
        shares,
        comments_added,
        saves,
        CASE WHEN views > 0 THEN (likes::FLOAT / views) * 100 ELSE 0 END AS like_rate,
        CASE WHEN views > 0 THEN (shares::FLOAT / views) * 100 ELSE 0 END AS share_rate,
        CASE WHEN views > 0 THEN (comments_added::FLOAT / views) * 100 ELSE 0 END AS comment_rate,
        CASE WHEN views > 0 THEN 
          ((COALESCE(likes,0) + COALESCE(shares,0) + COALESCE(comments_added,0) + COALESCE(saves,0))::FLOAT / views) * 100
        ELSE 0 END AS engagement_rate
      FROM public.tt_video_etl
      WHERE views > 0;
    `

    const result = await executeQuery(query)

    return NextResponse.json({
      engagement_distribution: result.map((row: any) => ({
        views: Number(row.views),
        likes: Number(row.likes),
        shares: Number(row.shares),
        comments_added: Number(row.comments_added),
        saves: Number(row.saves),
        like_rate: Number(row.like_rate),
        share_rate: Number(row.share_rate),
        comment_rate: Number(row.comment_rate),
        engagement_rate: Number(row.engagement_rate),
      })),
    })
  } catch (error) {
    console.error("[v0] Engagement distribution error:", error)
    return NextResponse.json({ error: "Failed to fetch engagement distribution data" }, { status: 500 })
  }
}
