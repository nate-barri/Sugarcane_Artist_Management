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
  PieChart,
  Pie,
  Cell,
} from "recharts"

export default function TikTokDashboard() {
  const [tempStartDate, setTempStartDate] = useState<string>("2021-01-01")
  const [tempEndDate, setTempEndDate] = useState<string>("2025-12-31")
  const [startDate, setStartDate] = useState<string>("2021-01-01")
  const [endDate, setEndDate] = useState<string>("2025-12-31")

  const [overview, setOverview] = useState<any>({})
  const [topVideos, setTopVideos] = useState<any[]>([])
  const [monthly, setMonthly] = useState<any[]>([])
  const [postType, setPostType] = useState<any[]>([])
  const [duration, setDuration] = useState<any[]>([])
  const [sound, setSound] = useState<any[]>([])
  const [engagementDist, setEngagementDist] = useState<any>(null)
  const [engagementRates, setEngagementRates] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true)
        setError(null)

        const dateParams = `startDate=${startDate}&endDate=${endDate}`

        const [overviewRes, videosRes, temporalRes, postTypeRes, durationRes, soundRes, engagementRes] =
          await Promise.all([
            fetch(`/api/analytics/tiktok/overview?${dateParams}`),
            fetch(`/api/analytics/tiktok/top-videos?limit=10&${dateParams}`),
            fetch(`/api/analytics/tiktok/temporal?${dateParams}`),
            fetch(`/api/analytics/tiktok/post-type?${dateParams}`),
            fetch(`/api/analytics/tiktok/duration?${dateParams}`),
            fetch(`/api/analytics/tiktok/sound?${dateParams}`),
            fetch(`/api/analytics/tiktok/engagement-distribution?${dateParams}`),
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
        setEngagementDist(engagementData.engagement_distribution)
        setEngagementRates(engagementData.engagement_rates)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data")
      } finally {
        setLoading(false)
      }
    }

    fetchAllData()
  }, [startDate, endDate])

  const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—"
    return `${n.toFixed(2)}%`
  }
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`

  const barChartData = engagementDist
    ? [
        {
          name: "Metrics",
          Views: engagementDist.total_views,
          Likes: engagementDist.total_likes,
          Shares: engagementDist.total_shares,
          Comments: engagementDist.total_comments,
          Saves: engagementDist.total_saves,
        },
      ]
    : []

  const pieChartData = engagementRates
    ? [
        { name: "Like Rate", value: engagementRates.like_rate },
        { name: "Share Rate", value: engagementRates.share_rate },
        { name: "Comment Rate", value: engagementRates.comment_rate },
        { name: "Engagement Rate", value: engagementRates.engagement_rate },
      ]
    : []

  const PIE_COLORS = ["#f59e0b", "#10b981", "#ec4899", "#8b5cf6"]

  const handleApplyFilter = () => {
    setStartDate(tempStartDate)
    setEndDate(tempEndDate)
  }

  const handleResetFilters = () => {
    setTempStartDate("2021-01-01")
    setTempEndDate("2025-12-31")
    setStartDate("2021-01-01")
    setEndDate("2025-12-31")
  }

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

        <section className="bg-white p-6 rounded-lg shadow-md mb-8 text-[#123458]">
          <h3 className="text-lg font-semibold mb-4">Date Range Filter</h3>
          <div className="flex gap-4 items-end">
            <div className="flex flex-col">
              <label className="text-sm font-medium mb-2">Start Date</label>
              <input
                type="date"
                value={tempStartDate}
                onChange={(e) => setTempStartDate(e.target.value)}
                min="2021-01-01"
                max="2025-12-31"
                className="px-4 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            <div className="flex flex-col">
              <label className="text-sm font-medium mb-2">End Date</label>
              <input
                type="date"
                value={tempEndDate}
                onChange={(e) => setTempEndDate(e.target.value)}
                min="2021-01-01"
                max="2025-12-31"
                className="px-4 py-2 border border-gray-300 rounded-lg"
              />
            </div>
            <button
              onClick={handleApplyFilter}
              className="px-4 py-2 bg-[#1e7a96] text-white rounded-lg hover:bg-[#155a73] font-medium"
            >
              Apply Filter
            </button>
            <button
              onClick={handleResetFilters}
              className="px-4 py-2 bg-[#3396D3] text-white rounded-lg hover:bg-[#2A75A4]"
            >
              Reset Filters
            </button>
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8">
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-xs font-medium text-gray-600 uppercase">Total Videos</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtInt(overview.total_videos)}</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-xs font-medium text-gray-600 uppercase">Total Views</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_views)}</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-xs font-medium text-gray-600 uppercase">Total Likes</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_likes)}</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-xs font-medium text-gray-600 uppercase">Total Shares</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_shares)}</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-xs font-medium text-gray-600 uppercase">Engagement Rate</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtPct(overview.engagement_rate)}</p>
          </div>
        </section>

        {/* Top Videos */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Videos by Views</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {topVideos.length > 0 ? (
              <div className="space-y-2">
                {topVideos.map((video, index) => (
                  <div
                    key={video.video_id}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-gray-500 font-semibold text-sm">#{index + 1}</span>
                        <p className="text-gray-800 font-medium truncate flex-1">{video.title}</p>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{fmtInt(video.views)} views</p>
                    </div>
                  </div>
                ))}
              </div>
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
                    <Bar dataKey="video_count" stackId="a" fill="#06b6d4" name="Video Count" />
                    <Bar dataKey="avg_views" stackId="a" fill="#f59e0b" name="Avg Views" />
                    <Bar dataKey="avg_likes" stackId="a" fill="#ec4899" name="Avg Likes" />
                    <Bar dataKey="avg_engagement" stackId="a" fill="#10b981" name="Avg Engagement" />
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

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Grouped Bar Chart for Engagement Metrics */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Engagement Distribution</h2>
            <div className="h-96">
              {barChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={barChartData} margin={{ top: 5, right: 30, left: 60, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis
                      scale="log"
                      type="number"
                      domain={[1, "dataMax"]}
                      tickFormatter={fmtCompact}
                      tick={{ fontSize: 14, fontWeight: 600 }}
                      label={{ value: "Log Scale", angle: -90, position: "insideLeft", fontSize: 12, fontWeight: 600 }}
                    />
                    <Tooltip formatter={(v) => fmtCompact(v as number)} />
                    <Legend />
                    <Bar dataKey="Views" fill="#0c4d8f" />
                    <Bar dataKey="Likes" fill="#f59e0b" />
                    <Bar dataKey="Shares" fill="#10b981" />
                    <Bar dataKey="Comments" fill="#ec4899" />
                    <Bar dataKey="Saves" fill="#8b5cf6" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Engagement Rate Distribution</h2>
            <div className="h-96">
              {pieChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <Pie
                      data={pieChartData}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, value }) => `${name}: ${value.toFixed(2)}%`}
                      outerRadius={130}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {pieChartData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v) => `${(v as number).toFixed(2)}%`} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}
