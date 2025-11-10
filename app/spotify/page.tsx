"use client"

import Sidebar from "@/components/sidebar"
import { useEffect, useState } from "react"
import { ReferenceDot, Label } from "recharts"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from "recharts"
import { generateSpotifyCSV, generateSpotifyPDF } from "@/lib/spotify-report"

type Overview = {
  total_streams: number
  total_followers: number
  total_listeners: number
  top_track: string
  top_track_streams: number
}

export default function SpotifyDashboard() {
  const [overview, setOverview] = useState<any>({})
  const [topTracks, setTopTracks] = useState<any[]>([])
  const [monthly, setMonthly] = useState<any[]>([])
  const [engagement, setEngagement] = useState<any[]>([])
  const [dailyStreams, setDailyStreams] = useState<any[]>([])
  const [dailyListeners, setDailyListeners] = useState<any[]>([])
  const [songReleases, setSongReleases] = useState<any[]>([])
  const [releaseMarkers, setReleaseMarkers] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generatingReport, setGeneratingReport] = useState(false)
  const [followerGrowth, setFollowerGrowth] = useState<any[]>([])
  const [followerReleaseMarkers, setFollowerReleaseMarkers] = useState<any[]>([])
  const [streamsGrowth, setStreamsGrowth] = useState<any[]>([])
  const [streamsGrowthReleases, setStreamsGrowthReleases] = useState<any[]>([])
  const [followersGrowthPct, setFollowersGrowthPct] = useState<any[]>([])
  const [followersGrowthReleases, setFollowersGrowthReleases] = useState<any[]>([])

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true)
        setError(null)

        const [
          overviewRes,
          topTracksRes,
          monthlyRes,
          engagementRes,
          dailyStreamsRes,
          dailyListenersRes,
          followerGrowthRes,
          streamsGrowthRes,
          followersGrowthPctRes,
        ] = await Promise.all([
          fetch("/api/analytics/spotify/overview"),
          fetch("/api/analytics/spotify/top-tracks"),
          fetch("/api/analytics/spotify/monthly"),
          fetch("/api/analytics/spotify/engagement"),
          fetch("/api/analytics/spotify/daily-streams-with-releases"),
          fetch("/api/analytics/spotify/daily-listeners-with-releases"),
          fetch("/api/analytics/spotify/follower-growth-with-releases"),
          fetch("/api/analytics/spotify/streams-growth-with-releases"),
          fetch("/api/analytics/spotify/followers-growth-perc-with-releases"),
        ])

        if (!overviewRes.ok) throw new Error("Failed to fetch overview")
        const overviewData = await overviewRes.json()

        setOverview({
          total_streams: Number(overviewData.overview.total_streams),
          total_followers: Number(overviewData.overview.total_followers),
          total_listeners: Number(overviewData.overview.total_listeners),
          top_track: String(overviewData.overview.top_track ?? ""),
          top_track_streams: Number(overviewData.overview.top_track_streams ?? 0),
        })

        if (!topTracksRes.ok) throw new Error("Failed to fetch top tracks")
        const topTracksData = await topTracksRes.json()
        setTopTracks(topTracksData.tracks || [])

        if (!monthlyRes.ok) throw new Error("Failed to fetch monthly data")
        const monthlyData = await monthlyRes.json()
        setMonthly(monthlyData.monthly || [])

        if (!engagementRes.ok) throw new Error("Failed to fetch engagement data")
        const engagementData = await engagementRes.json()
        setEngagement(engagementData.engagement_distribution || [])

        if (!dailyStreamsRes.ok) throw new Error("Failed to fetch daily streams and releases")
        const dailyStreamsData = await dailyStreamsRes.json()
        setDailyStreams(dailyStreamsData.daily_streams || [])
        setSongReleases(dailyStreamsData.song_releases || [])

        if (!dailyListenersRes.ok) throw new Error("Failed to fetch daily listeners and releases")
        const dailyListenersData = await dailyListenersRes.json()
        setDailyListeners(dailyListenersData.daily || [])
        setReleaseMarkers(dailyListenersData.releases || [])

        if (!followerGrowthRes.ok) throw new Error("Failed to fetch follower growth and releases")
        const followerGrowthData = await followerGrowthRes.json()
        setFollowerGrowth(followerGrowthData.follower_growth || followerGrowthData.data || [])
        setFollowerReleaseMarkers(followerGrowthData.song_releases || followerGrowthData.releases || [])

        if (!streamsGrowthRes.ok) throw new Error("Failed to fetch streams growth with releases")
        const streamsGrowthData = await streamsGrowthRes.json()
        setStreamsGrowth(streamsGrowthData.streams_growth || [])
        setStreamsGrowthReleases(streamsGrowthData.song_releases || [])

        if (!followersGrowthPctRes.ok) throw new Error("Failed to fetch followers growth % with releases")
        const followersGrowthPctData = await followersGrowthPctRes.json()
        setFollowersGrowthPct(followersGrowthPctData.followers_growth || [])
        setFollowersGrowthReleases(followersGrowthPctData.song_releases || [])
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data")
      } finally {
        setLoading(false)
      }
    }

    fetchAllData()
  }, [])

  const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—"
    return `${n.toFixed(2)}%`
  }
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`

  const findNearestStreamPoint = (series: any[], targetDateStr: string) => {
    if (!series?.length) return null
    const target = new Date(targetDateStr).getTime()
    let nearest = series[0]
    let best = Math.abs(new Date(series[0].date).getTime() - target)
    for (let i = 1; i < series.length; i++) {
      const t = Math.abs(new Date(series[i].date).getTime() - target)
      if (t < best) {
        best = t
        nearest = series[i]
      }
    }
    return nearest
  }

  const findNearestFollowerPoint = (series: any[], targetDateStr: string) => {
    if (!series?.length) return null
    const target = new Date(targetDateStr).getTime()
    let nearest = series[0]
    let best = Math.abs(new Date(series[0].date).getTime() - target)
    for (let i = 1; i < series.length; i++) {
      const t = Math.abs(new Date(series[i].date).getTime() - target)
      if (t < best) {
        best = t
        nearest = series[i]
      }
    }
    return nearest
  }

  const findNearestGrowthPoint = (series: any[], targetDateStr: string) => {
    if (!series?.length) return null
    const target = new Date(targetDateStr).getTime()
    let nearest = series[0]
    let best = Math.abs(new Date(series[0].date).getTime() - target)
    for (let i = 1; i < series.length; i++) {
      const t = Math.abs(new Date(series[i].date).getTime() - target)
      if (t < best) {
        best = t
        nearest = series[i]
      }
    }
    return nearest
  }

  const handleGenerateReport = async (format: "csv" | "pdf") => {
    setGeneratingReport(true)
    try {
      if (format === "csv") {
        generateSpotifyCSV({
          overview,
          topTracks,
          monthly,
          dailyListeners,
          followerGrowth,
          streamsGrowth,
          followersGrowthPct,
        })
      } else {
        generateSpotifyPDF({
          overview,
          topTracks,
          monthly,
          dailyListeners,
          followerGrowth,
          streamsGrowth,
          followersGrowthPct,
        })
      }
    } catch (err) {
      console.error("Error generating report:", err)
    } finally {
      setGeneratingReport(false)
    }
  }

  if (loading) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3] text-white">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <p className="text-xl text-[#123458]">Loading dashboard data...</p>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3] text-white">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <div className="bg-red-900 p-6 rounded-lg">
            <p className="text-lg font-semibold">Error loading dashboard</p>
            <p className="text-sm mt-2">{error}</p>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-white">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">Spotify Analytics Dashboard</h1>
        </header>

        {/* KPIs Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Streams</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.total_streams)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Listeners</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_listeners)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Followers</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.total_followers)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Top Track</h3>
            <p className="text-base font-semibold text-gray-800 mt-2 truncate" title={overview.top_track}>
              {overview.top_track || "—"}
            </p>
            <p className="text-3xl font-bold text-gray-900">{fmtInt(overview.top_track_streams)}</p>
          </div>
        </section>

        {/* Top Tracks Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Tracks by Streams</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {topTracks.length > 0 ? (
              <div className="space-y-2">
                {topTracks.map((track, index) => (
                  <div
                    key={`${track.track}-${index}`}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-gray-500 font-semibold text-sm">#{index + 1}</span>
                        <p className="text-gray-800 font-medium truncate flex-1">{track.track}</p>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{fmtInt(track.streams)} streams</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Daily Streams with Song Releases Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Daily Streams</h2>
          <div className="h-96">
            {dailyStreams.length > 0 && songReleases.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dailyStreams} margin={{ top: 20, right: 30, left: 70, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) =>
                      new Date(value).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
                    }
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis
                    width={72}
                    tickMargin={8}
                    tickFormatter={(v) => `${Number(v).toFixed(0)}`}
                    label={{
                      value: "Daily Streams",
                      angle: -90,
                      position: "left",
                      offset: 48,
                      style: { textAnchor: "middle", dominantBaseline: "central" },
                    }}
                  />
                  <Tooltip formatter={(value) => fmtInt(value as number)} />
                  <Legend verticalAlign="top" align="center" height={25} iconType="line" />
                  <Line dataKey="streams" stroke="#1DB954" strokeWidth={2} name="Daily Streams" dot={false} />

                  {(() => {
                    const exactLeonora = songReleases.find(
                      (r: any) => (r.title || "").trim().toLowerCase() === "leonora",
                    )
                    const containsLeonora = songReleases.find((r: any) =>
                      (r.title || "").toLowerCase().includes("leonora"),
                    )
                    const earliest = songReleases[0]

                    const startDateStr =
                      exactLeonora?.release_date ?? containsLeonora?.release_date ?? earliest?.release_date

                    const markersFromLeonora = songReleases.filter(
                      (r: any) => (r.release_date ?? r.date) >= startDateStr,
                    )

                    return markersFromLeonora.map((rel: any, i: number) => {
                      const relDate = rel.release_date ?? rel.date
                      const nearest = findNearestStreamPoint(dailyStreams, relDate)
                      if (!nearest) return null
                      return (
                        <ReferenceDot
                          key={`${relDate}-${i}`}
                          x={nearest.date}
                          y={nearest.streams}
                          r={5}
                          fill="#E11D48"
                          stroke="#ffffff"
                          strokeWidth={2}
                          ifOverflow="discard"
                        >
                          <Label value={rel.title} position="top" offset={10} fill="#111827" fontSize={11} />
                        </ReferenceDot>
                      )
                    })
                  })()}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Daily Listeners with Song Releases Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Daily Listeners</h2>
          <div className="h-96">
            {dailyListeners.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dailyListeners} margin={{ top: 20, right: 30, left: 70, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) =>
                      new Date(value as string).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })
                    }
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis
                    width={72}
                    tickMargin={8}
                    tickFormatter={(v) => `${Number(v).toFixed(0)}`}
                    label={{
                      value: "Daily Listeners",
                      angle: -90,
                      position: "left",
                      offset: 48,
                      style: { textAnchor: "middle", dominantBaseline: "central" },
                    }}
                  />
                  <Tooltip
                    formatter={(value) => fmtInt(value as number)}
                    labelFormatter={(value) =>
                      new Date(value as string).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })
                    }
                  />
                  <Legend verticalAlign="top" align="center" height={25} iconType="line" />
                  <Line dataKey="listeners" stroke="#0c4d8f" strokeWidth={2} name="Daily Listeners" dot={false} />

                  {releaseMarkers.map((rel: any, i: number) => {
                    const y = dailyListeners.find((d: any) => d.date === rel.date)?.listeners ?? 0
                    return (
                      <ReferenceDot
                        key={`${rel.date}-${i}`}
                        x={rel.date}
                        y={y}
                        r={5}
                        fill="#E11D48"
                        stroke="#ffffff"
                        strokeWidth={2}
                        ifOverflow="discard"
                      >
                        <Label value={rel.song ?? "Release"} position="top" offset={10} fill="#111827" fontSize={11} />
                      </ReferenceDot>
                    )
                  })}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Follower Growth Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Follower Growth</h2>
          <div className="h-96">
            {followerGrowth.length > 0 ? (
              (() => {
                const exactLeonora = followerReleaseMarkers.find(
                  (r: any) => (r.title || "").trim().toLowerCase() === "leonora",
                )
                const containsLeonora = followerReleaseMarkers.find((r: any) =>
                  (r.title || "").toLowerCase().includes("leonora"),
                )
                const earliestMarker = followerReleaseMarkers.reduce(
                  (min: any, r: any) => (!min || (r.release_date ?? r.date) < (min.release_date ?? min.date) ? r : min),
                  null as any,
                )

                const startDateStr =
                  exactLeonora?.release_date ??
                  containsLeonora?.release_date ??
                  earliestMarker?.release_date ??
                  followerGrowth[0]?.date

                const fgFromStart = followerGrowth.filter((d: any) => d.date >= startDateStr)
                const markersFromStart = followerReleaseMarkers.filter(
                  (r: any) => (r.release_date ?? r.date) >= startDateStr,
                )

                return (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={fgFromStart} margin={{ top: 20, right: 30, left: 70, bottom: 60 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="date"
                        tickFormatter={(value) =>
                          new Date(value as string).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })
                        }
                        angle={-45}
                        textAnchor="end"
                      />
                      <YAxis
                        width={72}
                        tickMargin={8}
                        tickFormatter={(v) => `${Number(v).toFixed(0)}`}
                        label={{
                          value: "Total Followers",
                          angle: -90,
                          position: "left",
                          offset: 48,
                          style: { textAnchor: "middle", dominantBaseline: "central" },
                        }}
                      />
                      <Tooltip
                        formatter={(value) => fmtInt(value as number)}
                        labelFormatter={(value) =>
                          new Date(value as string).toLocaleDateString("en-US", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })
                        }
                      />
                      <Legend verticalAlign="top" align="center" height={25} iconType="line" />
                      <Line
                        dataKey="total_followers"
                        stroke="#7c3aed"
                        strokeWidth={2}
                        name="Total Followers"
                        dot={false}
                      />

                      {markersFromStart.map((rel: any, i: number) => {
                        const relDate = rel.release_date ?? rel.date
                        const relTitle = rel.title ?? rel.song ?? "Release"
                        const nearest = findNearestFollowerPoint(fgFromStart, relDate)
                        if (!nearest) return null
                        return (
                          <ReferenceDot
                            key={`${relDate}-${i}`}
                            x={nearest.date}
                            y={nearest.total_followers}
                            r={5}
                            fill="#E11D48"
                            stroke="#ffffff"
                            strokeWidth={2}
                            ifOverflow="discard"
                          >
                            <Label value={relTitle} position="top" offset={10} fill="#111827" fontSize={11} />
                          </ReferenceDot>
                        )
                      })}
                    </LineChart>
                  </ResponsiveContainer>
                )
              })()
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Streams Growth % Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Streams Growth Percentage</h2>
          <div className="h-96">
            {streamsGrowth.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={streamsGrowth} margin={{ top: 20, right: 30, left: 70, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) =>
                      new Date(value as string).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })
                    }
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis
                    width={72}
                    tickMargin={8}
                    tickFormatter={(v) => `${Number(v).toFixed(0)}`}
                    label={{
                      value: "Stream Growth from Baseline (%)",
                      angle: -90,
                      position: "left",
                      offset: 48,
                      style: { textAnchor: "middle", dominantBaseline: "central" },
                    }}
                  />
                  <Tooltip
                    formatter={(value) => `${Number(value as number).toFixed(2)}%`}
                    labelFormatter={(value) =>
                      new Date(value as string).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })
                    }
                  />
                  <Legend verticalAlign="top" align="center" height={25} iconType="line" />
                  <Line dataKey="growth_pct" stroke="#f87171" strokeWidth={2} name="Streams Growth (%)" dot={false} />

                  {(() => {
                    const exactLeonora = streamsGrowthReleases.find(
                      (r: any) => (r.title || "").trim().toLowerCase() === "leonora",
                    )
                    const containsLeonora = streamsGrowthReleases.find((r: any) =>
                      (r.title || "").toLowerCase().includes("leonora"),
                    )
                    const earliest = streamsGrowthReleases[0]

                    const startDateStr =
                      exactLeonora?.release_date ?? containsLeonora?.release_date ?? earliest?.release_date

                    const markersFromLeonora = streamsGrowthReleases.filter(
                      (r: any) => (r.release_date ?? r.date) >= startDateStr,
                    )

                    return markersFromLeonora.map((rel: any, i: number) => {
                      const relDate = rel.release_date ?? rel.date
                      const relTitle = rel.title ?? rel.song ?? "Release"
                      const nearest = findNearestGrowthPoint(streamsGrowth, relDate)
                      if (!nearest) return null
                      return (
                        <ReferenceDot
                          key={`${relDate}-${i}`}
                          x={nearest.date}
                          y={nearest.growth_pct}
                          r={5}
                          fill="#E11D48"
                          stroke="#ffffff"
                          strokeWidth={2}
                          ifOverflow="discard"
                        >
                          <Label value={relTitle} position="top" offset={10} fill="#111827" fontSize={11} />
                        </ReferenceDot>
                      )
                    })
                  })()}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Followers Growth % Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Followers Growth Percentage</h2>
          <div className="h-96">
            {followersGrowthPct.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={followersGrowthPct} margin={{ top: 20, right: 30, left: 70, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(v) =>
                      new Date(v as string).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })
                    }
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis
                    width={72}
                    tickMargin={8}
                    tickFormatter={(v) => `${Number(v).toFixed(0)}%`}
                    label={{
                      value: "Growth from Baseline (%)",
                      angle: -90,
                      position: "left",
                      offset: 48,
                      style: { textAnchor: "middle", dominantBaseline: "central" },
                    }}
                  />
                  <Tooltip
                    formatter={(value) => `${Number(value as number).toFixed(2)}%`}
                    labelFormatter={(v) =>
                      new Date(v as string).toLocaleDateString("en-US", {
                        year: "numeric",
                        month: "short",
                        day: "numeric",
                      })
                    }
                  />
                  <Legend verticalAlign="top" align="center" height={25} iconType="line" />
                  <Line dataKey="growth_pct" stroke="#f59e0b" strokeWidth={2} name="Followers Growth (%)" dot={false} />

                  {(() => {
                    const exactLeonora = followersGrowthReleases.find(
                      (r: any) => (r.title || "").trim().toLowerCase() === "leonora",
                    )
                    const containsLeonora = followersGrowthReleases.find((r: any) =>
                      (r.title || "").toLowerCase().includes("leonora"),
                    )
                    const earliest = followersGrowthReleases[0]
                    const startDateStr =
                      exactLeonora?.release_date ?? containsLeonora?.release_date ?? earliest?.release_date

                    const markersFromLeonora = followersGrowthReleases.filter(
                      (r: any) => (r.release_date ?? r.date) >= startDateStr,
                    )

                    return markersFromLeonora.map((rel: any, i: number) => {
                      const relDate = rel.release_date ?? rel.date
                      const nearest = findNearestGrowthPoint(followersGrowthPct, relDate)
                      if (!nearest) return null
                      return (
                        <ReferenceDot
                          key={`${relDate}-${i}`}
                          x={nearest.date}
                          y={nearest.growth_pct}
                          r={5}
                          fill="#E11D48"
                          stroke="#ffffff"
                          strokeWidth={2}
                          ifOverflow="discard"
                        >
                          <Label
                            value={rel.title ?? "Release"}
                            position="top"
                            offset={10}
                            fill="#111827"
                            fontSize={11}
                          />
                        </ReferenceDot>
                      )
                    })
                  })()}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        <section className="flex justify-center gap-4 mb-8">
          <button
            onClick={() => handleGenerateReport("csv")}
            disabled={generatingReport}
            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg transition disabled:opacity-50"
          >
            {generatingReport ? "Generating..." : "Download CSV Report"}
          </button>
          <button
            onClick={() => handleGenerateReport("pdf")}
            disabled={generatingReport}
            className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-8 rounded-lg transition disabled:opacity-50"
          >
            {generatingReport ? "Generating..." : "Download PDF Report"}
          </button>
        </section>
      </main>
    </div>
  )
}
