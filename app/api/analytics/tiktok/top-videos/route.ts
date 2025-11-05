import { NextResponse } from "next/server"
import { executeQuery } from "@/lib/db-utils"

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url)
    const limit = Number.parseInt(searchParams.get("limit") || "10", 10)

    const queryByViews = `
      SELECT video_id, title, views, likes, shares, comments_added, saves, publish_time, url
      FROM public.tt_video_etl
      WHERE views IS NOT NULL
      ORDER BY views DESC
      LIMIT $1;
    `

    const result = await executeQuery(queryByViews, [limit])

    return NextResponse.json({
      videos: result.map((row: any) => ({
        video_id: row.video_id,
        title: row.title,
        views: Number(row.views),
        likes: Number(row.likes),
        shares: Number(row.shares),
        comments_added: Number(row.comments_added),
        saves: Number(row.saves),
        publish_time: row.publish_time,
        url: row.url,
      })),
    })
  } catch (error) {
    console.error("[v0] Top videos error:", error)
    return NextResponse.json({ error: "Failed to fetch top videos" }, { status: 500 })
  }
}
