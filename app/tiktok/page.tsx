"use client"

import Sidebar from "@/components/sidebar"
import { useEffect, useState } from "react"
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ScatterChart,
  Scatter,
} from "recharts"

export default function TikTokDashboard() {
  const [overview, setOverview] = useState<any>({})
  const [topVideos, setTopVideos] = useState<any[]>([])
  const [monthly, setMonthly] = useState<any[]>([])
  const [postType, setPostType] = useState<any[]>([])
  const [duration, setDuration] = useState<any[]>([])
  const [sound, setSound] = useState<any[]>([])
  const [engagement, setEngagement] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true)
        setError(null)

        const [overviewRes, videosRes, temporalRes, postTypeRes, durationRes, soundRes, engagementRes] =
          await Promise.all([
            fetch("/api/analytics/tiktok/overview"),
            fetch("/api/analytics/tiktok/top-videos?limit=10"),
            fetch("/api/analytics/tiktok/temporal"),
            fetch("/api/analytics/tiktok/post-type"),
            fetch("/api/analytics/tiktok/duration"),
            fetch("/api/analytics/tiktok/sound"),
            fetch("/api/analytics/tiktok/engagement-distribution"),
          ])

        if (!overviewRes.ok) throw new Error("Failed to fetch overview")
        const overviewData = await overviewRes.json()
        setOverview(overviewData)

        if (!videosRes.ok) throw new Error("Failed to fetch videos")
        const videosData = await videosRes.json()
        setTopVideos(videosData.videos || [])

        if (!temporalRes.ok) throw new Error("Failed to fetch temporal data")
        const temporalData = await temporalRes.json()
        setMonthly(temporalData.monthly || [])

        if (!postTypeRes.ok) throw new Error("Failed to fetch post type data")
        const postTypeData = await postTypeRes.json()
        setPostType(postTypeData.post_type || [])

        if (!durationRes.ok) throw new Error("Failed to fetch duration data")
        const durationData = await durationRes.json()
        setDuration(durationData.duration || [])

        if (!soundRes.ok) throw new Error("Failed to fetch sound data")
        const soundData = await soundRes.json()
        setSound(soundData.sound || [])

        if (!engagementRes.ok) throw new Error("Failed to fetch engagement data")
        const engagementData = await engagementRes.json()
        setEngagement(engagementData.engagement_distribution || [])
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

  if (loading) {
    return (
      <div className="flex min-h-screen bg-[#123458] text-white">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <p className="text-xl">Loading dashboard data...</p>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex min-h-screen bg-[#123458] text-white">
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
    <div className="flex min-h-screen bg-[#123458] text-white">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">TikTok Analytics Dashboard</h1>
        </header>

        {/* KPIs */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Videos</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.total_videos)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Total Views</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_views)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Engagement Rate</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtPct(overview.engagement_rate)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-sm font-medium text-gray-600">Avg Videos/Month</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.avg_videos_per_month)}</p>
          </div>
        </section>

        {/* Top Videos */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Videos by Views</h2>
          <div className="h-96">
            {topVideos.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topVideos} layout="vertical" margin={{ top: 5, right: 30, left: 200, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={fmtCompact} />
                  <YAxis dataKey="title" type="category" width={190} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmtInt(v as number)} />
                  <Bar dataKey="views" fill="#0c4d8f" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Monthly Trends */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Monthly Views Trend</h2>
          <div className="h-96">
            {monthly.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={monthly.map((m) => ({
                    month: `${m.publish_year}-${String(m.publish_month).padStart(2, "0")}`,
                    views: m.total_views,
                  }))}
                  margin={{ top: 5, right: 30, left: 0, bottom: 30 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" angle={-45} textAnchor="end" height={60} />
                  <YAxis />
                  <Tooltip formatter={(v) => fmtInt(v as number)} />
                  <Line type="monotone" dataKey="views" stroke="#2c0379" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Post Type Analysis */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Performance by Post Type</h2>
            <div className="h-80">
              {postType.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={postType} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="post_type" />
                    <YAxis />
                    <Tooltip formatter={(v) => fmtInt(v as number)} />
                    <Legend />
                    <Bar dataKey="avg_views" fill="#8b5cf6" name="Avg Views" />
                    <Bar dataKey="avg_likes" fill="#ec4899" name="Avg Likes" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          {/* Duration Analysis */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Performance by Duration</h2>
            <div className="h-80">
              {duration.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={duration} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="duration_bucket" />
                    <YAxis />
                    <Tooltip formatter={(v) => fmtInt(v as number)} />
                    <Legend />
                    <Bar dataKey="avg_views" fill="#06b6d4" name="Avg Views" />
                    <Bar dataKey="avg_engagement" fill="#14b8a6" name="Avg Engagement" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>
        </section>

        {/* Sound Analysis */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 20 Sounds by Views</h2>
          <div className="h-96">
            {sound.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={sound.slice(0, 10)}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 150, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={fmtCompact} />
                  <YAxis dataKey="sound_category" type="category" width={140} tick={{ fontSize: 10 }} />
                  <Tooltip formatter={(v) => fmtInt(v as number)} />
                  <Bar dataKey="total_views" fill="#f59e0b" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Engagement Rate Distribution */}
        <section className="bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Engagement Rate Distribution</h2>
          <div className="h-96">
            {engagement.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="views" name="Views" tickFormatter={fmtCompact} />
                  <YAxis dataKey="engagement_rate" name="Engagement Rate %" />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v) => fmtPct(v as number)} />
                  <Scatter name="Videos" data={engagement.slice(0, 500)} fill="#10b981" />
                </ScatterChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>
      </main>
    </div>
  )
}
