import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET() {
  try {
    const query = `
      SELECT post_type, COUNT(*) as video_count,
             AVG(views) as avg_views, AVG(likes) as avg_likes,
             AVG(shares) as avg_shares, AVG(comments_added) as avg_comments,
             SUM(views) as total_views
      FROM public.tt_video_etl
      WHERE post_type IS NOT NULL AND views IS NOT NULL
      GROUP BY post_type
      ORDER BY total_views DESC;
    `

    const result = await executeQuery(query)

    return NextResponse.json({
      post_type: result.map((row: any) => ({
        post_type: row.post_type,
        video_count: Number(row.video_count),
        avg_views: Number(row.avg_views),
        avg_likes: Number(row.avg_likes),
        avg_shares: Number(row.avg_shares),
        avg_comments: Number(row.avg_comments),
        total_views: Number(row.total_views),
      })),
    })
  } catch (error) {
    console.error("[v0] Post type analysis error:", error)
    return NextResponse.json({ error: "Failed to fetch post type data" }, { status: 500 })
  }
}
