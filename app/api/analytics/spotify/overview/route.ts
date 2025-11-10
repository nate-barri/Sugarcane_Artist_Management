// app/api/analytics/spotify/overview/route.ts
import { NextResponse } from "next/server";
import { executeQuery } from "@/lib/db-utils";

export async function GET() {
  try {
    const totalStreamsSql = `
      WITH by_day AS (
        SELECT
          (date)::date AS d,
          COALESCE(
            MAX(CASE WHEN LOWER(TRIM(song)) = 'all songs' THEN streams END),
            SUM(streams)
          )::bigint AS daily_streams
        FROM spotify_stats
        GROUP BY (date)::date
      )
      SELECT COALESCE(SUM(daily_streams),0)::bigint AS total_streams
      FROM by_day
    `;

    const latestTotalsSql = `
      WITH latest AS (SELECT MAX((date)::date) AS latest_d FROM spotify_stats)
      SELECT
        COALESCE(MAX(CASE WHEN LOWER(TRIM(song)) = 'all songs' THEN followers END), MAX(followers))::bigint AS total_followers,
        COALESCE(MAX(CASE WHEN LOWER(TRIM(song)) = 'all songs' THEN listeners END), MAX(listeners))::bigint AS total_listeners
      FROM spotify_stats s
      CROSS JOIN latest
      WHERE (s.date)::date = latest.latest_d
    `;

    // Sum per song, alias as total_streams so it matches the JSON below
    const topTrackSql = `
      WITH per_song AS (
        SELECT
          song_id,
          TRIM(song) AS song,
          SUM(COALESCE(streams,0))::bigint AS total_streams
        FROM spotify_songs
        WHERE COALESCE(TRIM(song), '') <> ''
        GROUP BY song_id, TRIM(song)
      )
      SELECT song_id, song, total_streams
      FROM per_song
      ORDER BY total_streams DESC, song ASC
      LIMIT 1
    `;

    const [{ total_streams }] = await executeQuery(totalStreamsSql);
    const [{ total_followers, total_listeners }] = await executeQuery(latestTotalsSql);
    const [topTrack] = await executeQuery(topTrackSql);

    return NextResponse.json({
      overview: {
        total_streams: Number(total_streams),
        total_followers: Number(total_followers),
        total_listeners: Number(total_listeners),
        top_track: topTrack?.song ?? "",
        top_track_streams: Number(topTrack?.total_streams ?? 0), // ← now matches SQL alias
        top_track_id: topTrack?.song_id ?? null,                  // (optional but handy)
      },
    });
  } catch (e) {
    console.error("overview error:", e);
    return NextResponse.json({ error: "Failed to load overview" }, { status: 500 });
  }
}
