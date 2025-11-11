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
  AreaChart,
  Area,
} from "recharts"
import jsPDF from "jspdf"

export default function FacebookDashboard() {
  const [tempStartDate, setTempStartDate] = useState<string>("2021-01-01")
  const [tempEndDate, setTempEndDate] = useState<string>("2025-12-31")
  const [startDate, setStartDate] = useState<string>("2021-01-01")
  const [endDate, setEndDate] = useState<string>("2025-12-31")

  const [overview, setOverview] = useState<any>({})
  const [topVideos, setTopVideos] = useState<any[]>([])
  const [engagementMetrics, setEngagementMetrics] = useState<any>(null)
  const [postTypeEngagement, setPostTypeEngagement] = useState<any[]>([])
  const [temporal, setTemporal] = useState<any>(null)
  const [postTypeDistribution, setPostTypeDistribution] = useState<any[]>([])
  const [cumulative, setCumulative] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generatingReport, setGeneratingReport] = useState(false)

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true)
        setError(null)

        const dateParams = `startDate=${startDate}&endDate=${endDate}`

        console.log("[v0] Fetching Facebook data with params:", dateParams)

        const [overviewRes, videosRes, metricsRes, postTypeEngRes, temporalRes, distributionRes, cumulativeRes] =
          await Promise.all([
            fetch(`/api/analytics/facebook/overview?${dateParams}`),
            fetch(`/api/analytics/facebook/top-videos?limit=10&${dateParams}`),
            fetch(`/api/analytics/facebook/engagement-metrics?${dateParams}`),
            fetch(`/api/analytics/facebook/post-type-engagement?${dateParams}`),
            fetch(`/api/analytics/facebook/temporal?${dateParams}`),
            fetch(`/api/analytics/facebook/post-type-distribution?${dateParams}`),
            fetch(`/api/analytics/facebook/cumulative?${dateParams}`),
          ])

        if (!overviewRes.ok) throw new Error("Failed to fetch overview")
        const overviewData = await overviewRes.json()
        console.log("[v0] Overview data received:", overviewData)
        setOverview(overviewData)

        if (!videosRes.ok) throw new Error("Failed to fetch videos")
        const videosData = await videosRes.json()
        console.log("[v0] Top videos data received:", videosData)
        setTopVideos(videosData.videos || [])

        if (!metricsRes.ok) throw new Error("Failed to fetch engagement metrics")
        const metricsData = await metricsRes.json()
        console.log("[v0] Engagement metrics data received:", metricsData)
        setEngagementMetrics(metricsData)

        if (!postTypeEngRes.ok) throw new Error("Failed to fetch post type engagement")
        const postTypeEngData = await postTypeEngRes.json()
        console.log("[v0] Post type engagement data received:", postTypeEngData)
        setPostTypeEngagement(postTypeEngData.post_type_engagement || [])

        if (!temporalRes.ok) throw new Error("Failed to fetch temporal data")
        const temporalData = await temporalRes.json()
        console.log("[v0] Temporal data received:", temporalData)
        setTemporal(temporalData)

        if (!distributionRes.ok) throw new Error("Failed to fetch distribution")
        const distributionData = await distributionRes.json()
        console.log("[v0] Distribution data received:", distributionData)
        setPostTypeDistribution(distributionData.distribution || [])

        if (!cumulativeRes.ok) throw new Error("Failed to fetch cumulative data")
        const cumulativeData = await cumulativeRes.json()
        console.log("[v0] Cumulative data received:", cumulativeData)
        setCumulative(cumulativeData.cumulative || [])
      } catch (err) {
        console.error("[v0] Error loading Facebook data:", err)
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

  const PIE_COLORS = ["#9b6dd6", "#ff8c42", "#4ade80", "#60a5fa", "#f43f5e"]

  const drawTable = (head: string[], body: (string | number)[][], startY: number, pdf: jsPDF) => {
    let currentY = startY
    const cellHeight = 6
    const colWidths = head.length === 2 ? [80, 50] : head.length === 5 ? [15, 80, 25, 25, 25] : [15, 105, 40]
    const pageHeight = pdf.internal.pageSize.getHeight()

    pdf.setFillColor(18, 52, 88)
    pdf.setTextColor(255, 255, 255)
    pdf.setFontSize(10)
    pdf.setFont(undefined, "bold")

    let xPos = 10
    head.forEach((h, i) => {
      pdf.rect(xPos, currentY, colWidths[i], cellHeight, "F")
      pdf.text(h, xPos + 2, currentY + 4)
      xPos += colWidths[i]
    })
    currentY += cellHeight

    pdf.setTextColor(0, 0, 0)
    pdf.setFont(undefined, "normal")
    pdf.setFontSize(9)

    body.forEach((row, idx) => {
      if (currentY + cellHeight > pageHeight - 15) {
        pdf.addPage()
        currentY = 15
      }

      if (idx % 2 === 0) {
        pdf.setFillColor(240, 240, 240)
        xPos = 10
        head.forEach((_, i) => {
          pdf.rect(xPos, currentY, colWidths[i], cellHeight, "F")
          xPos += colWidths[i]
        })
      }

      xPos = 10
      row.forEach((cell, i) => {
        pdf.text(String(cell), xPos + 2, currentY + 4)
        xPos += colWidths[i]
      })

      currentY += cellHeight
    })

    return currentY
  }

  const generateCSVReport = () => {
    const csvContent = [
      ["Meta (Facebook) Analytics Report"],
      [`Generated on: ${new Date().toLocaleString()}`],
      [`Date Range: ${startDate} to ${endDate}`],
      [],
      ["OVERVIEW METRICS"],
      ["Total Posts", fmtInt(overview.total_posts)],
      ["Total Reach", fmtCompact(overview.total_reach)],
      ["Total Engagement", fmtCompact(overview.total_engagement)],
      ["Engagement Rate", fmtPct(overview.engagement_rate)],
      [],
      ["ENGAGEMENT METRICS"],
      ["Reactions", fmtInt(engagementMetrics?.metrics?.reactions)],
      ["Comments", fmtInt(engagementMetrics?.metrics?.comments)],
      ["Shares", fmtInt(engagementMetrics?.metrics?.shares)],
      [],
      ["ENGAGEMENT RATES"],
      ["Like Rate", fmtPct(engagementMetrics?.rates?.like_rate)],
      ["Comment Rate", fmtPct(engagementMetrics?.rates?.comment_rate)],
      ["Share Rate", fmtPct(engagementMetrics?.rates?.share_rate)],
      ["Overall Engagement", fmtPct(engagementMetrics?.rates?.overall_engagement)],
      [],
      ["TOP 10 POSTS BY ENGAGEMENT"],
      ["Rank", "Title", "Engagement", "Reach"],
      ...topVideos.map((video, idx) => [idx + 1, video.title, fmtInt(video.total_engagement), fmtInt(video.reach)]),
    ]

    const csvString = csvContent.map((row) => row.map((cell) => `"${cell}"`).join(",")).join("\n")
    const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" })
    const link = document.createElement("a")
    const url = URL.createObjectURL(blob)
    link.setAttribute("href", url)
    link.setAttribute("download", `facebook-report-${new Date().toISOString().split("T")[0]}.csv`)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const generatePDFReport = () => {
    const pdf = new jsPDF()
    let yPosition = 15

    pdf.setFontSize(18)
    pdf.setTextColor(18, 52, 88)
    pdf.setFont(undefined, "bold")
    pdf.text("Meta (Facebook) Analytics Report", 10, yPosition)
    yPosition += 12

    pdf.setFontSize(10)
    pdf.setTextColor(0, 0, 0)
    pdf.setFont(undefined, "normal")
    pdf.text(`Generated on: ${new Date().toLocaleString()}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Date Range: ${startDate} to ${endDate}`, 10, yPosition)
    yPosition += 12

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Overview Metrics", 10, yPosition)
    yPosition += 8

    pdf.setFontSize(10)
    pdf.setFont(undefined, "normal")
    pdf.setTextColor(0, 0, 0)
    pdf.text(`Total Posts: ${fmtInt(overview.total_posts)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Total Reach: ${fmtCompact(overview.total_reach)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Total Engagement: ${fmtCompact(overview.total_engagement)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Engagement Rate: ${fmtPct(overview.engagement_rate)}`, 10, yPosition)
    yPosition += 12

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Top 10 Posts by Engagement", 10, yPosition)
    yPosition += 8

    yPosition = drawTable(
      ["Rank", "Title", "Engagement", "Reach"],
      topVideos.map((video, idx) => [
        idx + 1,
        video.title.substring(0, 40),
        fmtInt(video.total_engagement),
        fmtInt(video.reach),
      ]),
      yPosition,
      pdf,
    )

    pdf.save(`facebook-report-${new Date().toISOString().split("T")[0]}.pdf`)
  }

  const handleGenerateReport = async (format: "csv" | "pdf") => {
    setGeneratingReport(true)
    try {
      if (format === "csv") {
        generateCSVReport()
      } else {
        generatePDFReport()
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
          <div className="bg-red-100 p-6 rounded-lg">
            <p className="text-lg font-semibold text-red-800">Error loading dashboard</p>
            <p className="text-sm mt-2 text-red-600">{error}</p>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-gray-800">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">Meta (Facebook)</h1>
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

        {/* Key Metrics Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Posts</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtInt(overview.total_posts)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Reach</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtCompact(overview.total_reach)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Engagement</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtCompact(overview.total_engagement)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Engagement Rate</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtPct(overview.engagement_rate)}</p>
          </div>
        </section>

        {/* Engagement Metrics Charts */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Total Engagement Metrics</h2>
            <div className="h-96">
              {engagementMetrics?.metrics ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      { name: "Reactions", value: engagementMetrics.metrics.reactions, fill: "#ff7f0e" },
                      { name: "Comments", value: engagementMetrics.metrics.comments, fill: "#9467bd" },
                      { name: "Shares", value: engagementMetrics.metrics.shares, fill: "#2ca02c" },
                    ]}
                    margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis tickFormatter={fmtCompact} />
                    <Tooltip formatter={(v) => fmtInt(v as number)} />
                    <Bar dataKey="value" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Engagement Rates (%)</h2>
            <div className="h-96">
              {engagementMetrics?.rates ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={[
                      { name: "Like Rate", value: engagementMetrics.rates.like_rate, fill: "#ff7f0e" },
                      { name: "Comment Rate", value: engagementMetrics.rates.comment_rate, fill: "#9467bd" },
                      { name: "Share Rate", value: engagementMetrics.rates.share_rate, fill: "#2ca02c" },
                      {
                        name: "Overall Engagement",
                        value: engagementMetrics.rates.overall_engagement,
                        fill: "#8b4513",
                      },
                    ]}
                    margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip formatter={(v) => fmtPct(v as number)} />
                    <Bar dataKey="value" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>
        </section>

        {/* Engagement Rate Breakdown by Post Type */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Engagement Rate Breakdown by Post Type</h2>
          <div className="h-96">
            {postTypeEngagement.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={postTypeEngagement} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="post_type" />
                  <YAxis label={{ value: "Engagement Rate (%)", angle: -90, position: "insideLeft" }} />
                  <Tooltip formatter={(v) => fmtPct(v as number)} />
                  <Legend />
                  <Bar dataKey="reactions_rate" stackId="a" fill="#ff7f0e" name="Reactions" />
                  <Bar dataKey="comments_rate" stackId="a" fill="#9467bd" name="Comments" />
                  <Bar dataKey="shares_rate" stackId="a" fill="#2ca02c" name="Shares" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Top 10 Videos by Engagement */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Videos by Engagement</h2>
          <div className="overflow-x-auto">
            {topVideos.length > 0 ? (
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b-2 border-gray-300">
                    <th className="pb-3 text-gray-700 font-semibold">Title</th>
                    <th className="pb-3 text-gray-700 font-semibold text-right">Total Engagement</th>
                    <th className="pb-3 text-gray-700 font-semibold text-right">Reach</th>
                  </tr>
                </thead>
                <tbody>
                  {topVideos.map((video, index) => (
                    <tr key={index} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 text-gray-900">{video.title}</td>
                      <td className="py-3 text-gray-600 text-right">{fmtInt(video.total_engagement)}</td>
                      <td className="py-3 text-gray-600 text-right">{fmtInt(video.reach)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Temporal Patterns */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Posts Published Per Month</h2>
            <div className="h-96">
              {temporal?.monthly_posts && temporal.monthly_posts.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={temporal.monthly_posts.map((m: any) => ({
                      date: `${m.year}-${String(m.month).padStart(2, "0")}`,
                      count: m.post_count,
                    }))}
                    margin={{ top: 5, right: 30, left: 0, bottom: 30 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="count" stroke="#9b59b6" strokeWidth={2} dot={{ r: 3 }} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Average Engagement by Day of Week</h2>
            <div className="h-96">
              {temporal?.day_of_week && temporal.day_of_week.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={temporal.day_of_week} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="day_name" />
                    <YAxis />
                    <Tooltip formatter={(v) => fmtInt(v as number)} />
                    <Bar dataKey="avg_engagement" fill="#f39c12" />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Average Reach by Hour of Day</h2>
            <div className="h-96">
              {temporal?.hourly_reach && temporal.hourly_reach.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={temporal.hourly_reach} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="hour" />
                    <YAxis tickFormatter={fmtCompact} />
                    <Tooltip formatter={(v) => fmtInt(v as number)} />
                    <Line type="monotone" dataKey="avg_reach" stroke="#16a085" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Total Monthly Reach Trend</h2>
            <div className="h-96">
              {temporal?.monthly_reach && temporal.monthly_reach.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={temporal.monthly_reach.map((m: any) => ({
                      date: `${m.year}-${String(m.month).padStart(2, "0")}`,
                      reach: m.total_reach,
                    }))}
                    margin={{ top: 5, right: 30, left: 0, bottom: 30 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} />
                    <YAxis tickFormatter={fmtCompact} />
                    <Tooltip formatter={(v) => fmtInt(v as number)} />
                    <Area type="monotone" dataKey="reach" stroke="#c0392b" fill="#c0392b" fillOpacity={0.3} />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>
        </section>

        {/* Post Type Distribution */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Distribution of Post Types</h2>
          <div className="h-96">
            {postTypeDistribution.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                  <Pie
                    data={postTypeDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ post_type, percentage }) => `${post_type}: ${percentage}%`}
                    outerRadius={130}
                    fill="#8884d8"
                    dataKey="count"
                  >
                    {postTypeDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Cumulative Growth Charts */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Cumulative Reach Growth Over Time</h2>
            <div className="h-96">
              {cumulative.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={cumulative.filter((_, i) => i % Math.ceil(cumulative.length / 200) === 0)}
                    margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" hide />
                    <YAxis tickFormatter={fmtCompact} />
                    <Tooltip
                      formatter={(v) => fmtInt(v as number)}
                      labelFormatter={(d) => new Date(d).toDateString()}
                    />
                    <Area
                      type="monotone"
                      dataKey="cumulative_reach"
                      stroke="#1f77b4"
                      fill="#1f77b4"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Cumulative Engagement Rate Over Time</h2>
            <div className="h-96">
              {cumulative.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={cumulative.filter((_, i) => i % Math.ceil(cumulative.length / 200) === 0)}
                    margin={{ top: 5, right: 30, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" hide />
                    <YAxis />
                    <Tooltip
                      formatter={(v) => fmtPct(v as number)}
                      labelFormatter={(d) => new Date(d).toDateString()}
                    />
                    <Area
                      type="monotone"
                      dataKey="cumulative_engagement_rate"
                      stroke="#2ecc71"
                      fill="#2ecc71"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </div>
        </section>

        {/* Generate Report Buttons */}
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
