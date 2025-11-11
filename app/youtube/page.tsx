"use client"

import Sidebar from "@/components/sidebar"
import { useEffect, useState } from "react"
import {
  ResponsiveContainer,
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
  ComposedChart,
  Line,
} from "recharts"
import { generateYouTubeCSV, generateYouTubePDF } from "@/lib/youtube-report"

export default function YouTubeDashboard() {
  const [tempStartDate, setTempStartDate] = useState<string>("2021-01-01")
  const [tempEndDate, setTempEndDate] = useState<string>("2025-12-31")
  const [startDate, setStartDate] = useState<string>("2021-01-01")
  const [endDate, setEndDate] = useState<string>("2025-12-31")

  const [overview, setOverview] = useState<any>({})
  const [topVideos, setTopVideos] = useState<any[]>([])
  const [topCategories, setTopCategories] = useState<any[]>([])
  const [monthly, setMonthly] = useState<any[]>([])
  const [contentType, setContentType] = useState<any[]>([])
  const [duration, setDuration] = useState<any[]>([])
  const [dayOfWeek, setDayOfWeek] = useState<any[]>([])
  const [contentDistribution, setContentDistribution] = useState<any[]>([])
  const [contentEngagement, setContentEngagement] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—"
    return `${n.toFixed(2)}%`
  }
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`

  const handleExportCSV = () => {
    const csv = generateYouTubeCSV(
      startDate,
      endDate,
      overview,
      topVideos,
      topCategories,
      monthly,
      duration,
      dayOfWeek,
      contentType,
      contentDistribution,
      contentEngagement,
    )
    const blob = new Blob([csv], { type: "text/csv" })
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `youtube-report-${new Date().toISOString().split("T")[0]}.csv`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  const handleExportPDF = () => {
    const doc = generateYouTubePDF(
      startDate,
      endDate,
      overview,
      topVideos,
      topCategories,
      monthly,
      duration,
      dayOfWeek,
      contentType,
      contentDistribution,
      contentEngagement,
    )
    doc.save(`youtube-report-${new Date().toISOString().split("T")[0]}.pdf`)
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 border border-gray-300 rounded shadow-lg">
          <p className="font-semibold text-gray-800">{label}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {entry.name}: {typeof entry.value === "number" ? fmtCompact(entry.value) : entry.value}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  const categoryColors: Record<string, string> = {
    leonora: "#e74c3c",
    "kung maging akin ka": "#3498db",
    "paalam, leonora": "#2ecc71",
    dalangin: "#9b59b6",
    paruparo: "#f1c40f",
    gunita: "#e67e22",
    "tanging ikaw": "#e91e63",
    gabi: "#95a5a6",
    shehan: "#f39c12",
  }

  const getCategoryColor = (category: string) => {
    return categoryColors[category?.toLowerCase()] || "#95a5a6"
  }

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

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true)
        setError(null)

        const dateParams = `startDate=${startDate}&endDate=${endDate}`

        const [
          overviewRes,
          videosRes,
          categoriesRes,
          temporalRes,
          contentTypeRes,
          durationRes,
          dayRes,
          distRes,
          engRes,
        ] = await Promise.all([
          fetch(`/api/analytics/youtube/overview?${dateParams}`),
          fetch(`/api/analytics/youtube/top-videos?limit=15&${dateParams}`),
          fetch(`/api/analytics/youtube/top-categories?limit=10&${dateParams}`),
          fetch(`/api/analytics/youtube/temporal?${dateParams}`),
          fetch(`/api/analytics/youtube/content-type?${dateParams}`),
          fetch(`/api/analytics/youtube/duration?${dateParams}`),
          fetch(`/api/analytics/youtube/day-of-week?${dateParams}`),
          fetch(`/api/analytics/youtube/content-type-distribution?${dateParams}`),
          fetch(`/api/analytics/youtube/content-type-engagement?${dateParams}`),
        ])

        if (!overviewRes.ok) throw new Error("Failed to fetch overview")
        const overviewData = await overviewRes.json()
        setOverview(overviewData)

        if (!videosRes.ok) throw new Error("Failed to fetch videos")
        const videosData = await videosRes.json()
        setTopVideos(videosData.videos || [])

        if (!categoriesRes.ok) throw new Error("Failed to fetch categories")
        const categoriesData = await categoriesRes.json()
        setTopCategories(categoriesData.categories || [])

        if (!temporalRes.ok) throw new Error("Failed to fetch temporal data")
        const temporalData = await temporalRes.json()
        setMonthly(temporalData.monthly || [])

        if (!contentTypeRes.ok) throw new Error("Failed to fetch content type data")
        const contentTypeData = await contentTypeRes.json()
        setContentType(contentTypeData.content_type || [])

        if (!durationRes.ok) throw new Error("Failed to fetch duration data")
        const durationData = await durationRes.json()
        setDuration(durationData.duration || [])

        if (!dayRes.ok) throw new Error("Failed to fetch day of week data")
        const dayData = await dayRes.json()
        setDayOfWeek(dayData.day_performance || [])

        if (!distRes.ok) throw new Error("Failed to fetch content distribution")
        const distData = await distRes.json()
        setContentDistribution(distData.distribution || [])

        if (!engRes.ok) throw new Error("Failed to fetch content engagement")
        const engData = await engRes.json()
        setContentEngagement(engData.engagement || [])
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data")
      } finally {
        setLoading(false)
      }
    }

    fetchAllData()
  }, [startDate, endDate])

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
        <header className="mb-8 flex items-center justify-between">
          <h1 className="text-3xl font-bold">YouTube Analytics Dashboard</h1>
        </header>

        {/* Date Range Filter */}
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

        {/* Overview Metrics */}
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
            <h3 className="text-xs font-medium text-gray-600 uppercase">Watch Time (hrs)</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtCompact(overview.total_watch_time)}</p>
          </div>
          <div className="bg-white p-4 rounded-lg shadow-md">
            <h3 className="text-xs font-medium text-gray-600 uppercase">Engagement Rate</h3>
            <p className="text-2xl font-bold text-gray-900 mt-2">{fmtPct(overview.engagement_rate)}</p>
          </div>
        </section>

        {/* Top 10 Songs */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Songs</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {topCategories.length > 0 ? (
              <div className="space-y-2">
                {topCategories.map((song, index) => (
                  <div
                    key={song.category}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-gray-500 font-semibold text-sm">#{index + 1}</span>
                        <p className="text-gray-800 font-medium capitalize flex-1">{song.category}</p>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{fmtCompact(song.total_views)} views</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Top 15 Videos by Views */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 15 Videos by Views</h2>
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {topVideos.length > 0 ? (
              <div className="space-y-2">
                {topVideos.map((video, index) => (
                  <div
                    key={video.video_id}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div
                      className="w-1 h-12 rounded-full"
                      style={{ backgroundColor: getCategoryColor(video.category) }}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-gray-500 font-semibold text-sm">#{index + 1}</span>
                        <p className="text-gray-800 font-medium flex-1">{video.title}</p>
                      </div>
                      <div className="flex items-center gap-3 mt-1">
                        <p className="text-sm text-gray-500">{fmtCompact(video.views)} views</p>
                        <span className="text-xs text-gray-400">•</span>
                        <p className="text-sm text-gray-400 capitalize">{video.category}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Monthly Upload Volume vs Total Views */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Monthly Upload Volume vs Total Views</h2>
          <div className="h-96">
            {monthly.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={monthly.map((m) => ({
                    month: `${m.publish_year}-${String(m.publish_month).padStart(2, "0")}`,
                    video_count: m.video_count,
                    total_views: m.total_views / 1_000_000,
                  }))}
                  margin={{ top: 5, right: 30, left: 0, bottom: 60 }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" angle={-45} textAnchor="end" height={80} />
                  <YAxis yAxisId="left" label={{ value: "Videos Uploaded", angle: -90, position: "insideLeft" }} />
                  <YAxis
                    yAxisId="right"
                    orientation="right"
                    label={{ value: "Total Views (M)", angle: 90, position: "insideRight" }}
                  />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="video_count"
                    stroke="#3498db"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Videos Uploaded"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="total_views"
                    stroke="#e74c3c"
                    strokeWidth={2}
                    dot={{ r: 4 }}
                    name="Total Views (M)"
                  />
                </ComposedChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Average Engagement Rate by Duration */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Average Engagement Rate by Duration</h2>
          <div className="h-96">
            {duration.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={duration} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="duration_bucket" />
                  <YAxis label={{ value: "Engagement Rate (%)", angle: -90, position: "insideLeft" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avg_engagement_rate" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* POSTING DAY PERFORMANCE (Cumulative Performance, NOT Audience Activity) */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">
            POSTING DAY PERFORMANCE (Cumulative Performance, NOT Audience Activity)
          </h2>
          <div className="h-96">
            {dayOfWeek.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={dayOfWeek} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day_of_week" />
                  <YAxis label={{ value: "Total Engagement", angle: -90, position: "insideLeft" }} />
                  <Tooltip formatter={(v) => fmtInt(v as number)} />
                  <Legend />
                  <Bar dataKey="mean_engagement" fill="#87ceeb" name="Mean Engagement" />
                  <Bar dataKey="median_engagement" fill="#4682b4" name="Median Engagement" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Average Views by Content Type */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Average Views by Content Type</h2>
            <div className="h-80">
              {contentType.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={contentType} margin={{ top: 5, right: 30, left: 0, bottom: 80 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="content_type" angle={-45} textAnchor="end" height={100} />
                    <YAxis tickFormatter={fmtCompact} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="avg_views" fill="#10b981" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Content Type Distribution</h2>
            <div className="h-80">
              {contentDistribution.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={contentDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ content_type, percentage }) => `${content_type}: ${percentage}%`}
                      outerRadius={100}
                      fill="#8884d8"
                      dataKey="video_count"
                    >
                      {contentDistribution.map((entry, index) => (
                        <Cell
                          key={`cell-${index}`}
                          fill={
                            ["#7dd3c0", "#f7dc6f", "#bb8fce", "#f1948a", "#85c1e9", "#f8b88b", "#aed6f1"][index % 7]
                          }
                        />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>
        </section>

        {/* Average Engagement Rate by Content Type */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Average Engagement Rate by Content Type</h2>
          <div className="h-96">
            {contentEngagement.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={contentEngagement} margin={{ top: 5, right: 30, left: 0, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="content_type" angle={-45} textAnchor="end" height={100} />
                  <YAxis label={{ value: "Engagement Rate (%)", angle: -90, position: "insideLeft" }} />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="avg_engagement_rate" fill="#9b59b6" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>
{/* Export Buttons */}
        <section className="flex gap-4 mt-12 mb-8 justify-center">
          <button
            onClick={handleExportCSV}
            className="px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 font-medium transition-colors"
          >
            Export CSV
          </button>
          <button
            onClick={handleExportPDF}
            className="px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 font-medium transition-colors"
          >
            Export PDF
          </button>
        </section>
      </main>
    </div>
  )
}