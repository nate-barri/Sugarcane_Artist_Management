"use client"

import Sidebar from "@/components/sidebar"
import { useEffect, useState } from "react"
import {
  ResponsiveContainer,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ComposedChart,
} from "recharts"
import jsPDF from "jspdf"

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
  const [generatingReport, setGeneratingReport] = useState(false)

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

  const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—"
    return `${n.toFixed(2)}%`
  }
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`

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

  const aggregateToYearly = (data: any[]) => {
    const yearlyMap: { [key: string]: number } = {}
    data.forEach((item) => {
      const year = item.publish_year || "Unknown"
      yearlyMap[year] = (yearlyMap[year] || 0) + (item.total_views || 0)
    })
    return Object.entries(yearlyMap)
      .sort()
      .map(([year, views]) => ({ year, views }))
  }

  const drawTable = (head: string[], body: (string | number)[][], startY: number, pdf: jsPDF, maxRows = 15) => {
    let currentY = startY
    const cellHeight = 6
    const colWidths = head.length === 2 ? [80, 50] : [15, 105, 40]
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

    body.slice(0, maxRows).forEach((row, idx) => {
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
    const yearlyTrend = aggregateToYearly(monthly)

    const csvContent = [
      ["YouTube Analytics Dashboard Report"],
      [`Generated on: ${new Date().toLocaleString()}`],
      [`Date Range: ${startDate} to ${endDate}`],
      [],
      ["OVERVIEW METRICS"],
      ["Total Videos", fmtInt(overview.total_videos)],
      ["Total Views", fmtCompact(overview.total_views)],
      ["Total Likes", fmtCompact(overview.total_likes)],
      ["Watch Time (hrs)", fmtCompact(overview.total_watch_time)],
      ["Engagement Rate", fmtPct(overview.engagement_rate)],
      [],
      ["TOP 10 VIDEOS BY VIEWS"],
      ["Rank", "Video Title", "Views"],
      ...topVideos.slice(0, 10).map((video, idx) => [idx + 1, video.title, fmtCompact(video.views)]),
      [],
      ["TOP 10 CATEGORIES"],
      ["Rank", "Category", "Views"],
      ...topCategories.slice(0, 10).map((cat, idx) => [idx + 1, cat.category, fmtCompact(cat.total_views)]),
      [],
      ["YEARLY VIEWS TREND"],
      ["Year", "Total Views"],
      ...yearlyTrend.map((item) => [item.year, fmtCompact(item.views)]),
      [],
      ["CONTENT TYPE PERFORMANCE"],
      ["Content Type", "Avg Views"],
      ...contentType.map((ct) => [ct.content_type, fmtCompact(ct.avg_views)]),
      [],
      ["AVERAGE ENGAGEMENT BY DURATION"],
      ["Duration", "Engagement Rate %"],
      ...duration.map((d) => [d.duration_bucket, fmtPct(d.avg_engagement_rate)]),
    ]

    const csvString = csvContent.map((row) => row.map((cell) => `"${cell}"`).join(",")).join("\n")
    const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" })
    const link = document.createElement("a")
    const url = URL.createObjectURL(blob)
    link.setAttribute("href", url)
    link.setAttribute("download", `youtube-report-${new Date().toISOString().split("T")[0]}.csv`)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const generatePDFReport = () => {
    const pdf = new jsPDF()
    const yearlyTrend = aggregateToYearly(monthly)
    let yPosition = 15

    pdf.setFontSize(18)
    pdf.setTextColor(18, 52, 88)
    pdf.setFont(undefined, "bold")
    pdf.text("YouTube Analytics Dashboard Report", 10, yPosition)
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
    pdf.text(`Total Videos: ${fmtInt(overview.total_videos)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Total Views: ${fmtCompact(overview.total_views)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Total Likes: ${fmtCompact(overview.total_likes)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Watch Time: ${fmtCompact(overview.total_watch_time)} hours`, 10, yPosition)
    yPosition += 5
    pdf.text(`Engagement Rate: ${fmtPct(overview.engagement_rate)}`, 10, yPosition)
    yPosition += 12

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Top 10 Videos by Views", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Rank", "Video Title", "Views"],
      topVideos.slice(0, 10).map((video, idx) => [idx + 1, video.title, fmtCompact(video.views)]),
      yPosition,
      pdf,
      10,
    )
    yPosition += 8

    if (yPosition > 240) {
      pdf.addPage()
      yPosition = 15
    }

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Top 10 Categories", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Rank", "Category", "Views"],
      topCategories.slice(0, 10).map((cat, idx) => [idx + 1, cat.category, fmtCompact(cat.total_views)]),
      yPosition,
      pdf,
      10,
    )
    yPosition += 8

    if (yPosition > 240) {
      pdf.addPage()
      yPosition = 15
    }

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Yearly Views Trend", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Year", "Total Views"],
      yearlyTrend.map((item) => [item.year, fmtCompact(item.views)]),
      yPosition,
      pdf,
    )
    yPosition += 8

    if (yPosition > 240) {
      pdf.addPage()
      yPosition = 15
    }

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Content Type Performance", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Content Type", "Avg Views"],
      contentType.map((ct) => [ct.content_type, fmtCompact(ct.avg_views)]),
      yPosition,
      pdf,
    )
    yPosition += 8

    if (yPosition > 240) {
      pdf.addPage()
      yPosition = 15
    }

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Engagement by Duration", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Duration", "Engagement Rate %"],
      duration.map((d) => [d.duration_bucket, fmtPct(d.avg_engagement_rate)]),
      yPosition,
      pdf,
    )

    pdf.save(`youtube-report-${new Date().toISOString().split("T")[0]}.pdf`)
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
    <div className="flex min-h-screen bg-[#D3D3D3] text-white">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">YouTube Analytics Dashboard</h1>
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

        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Videos</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {topVideos.length > 0 ? (
              <div className="space-y-2">
                {topVideos.slice(0, 10).map((video, index) => (
                  <div
                    key={video.video_id}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-gray-500 font-semibold text-sm">#{index + 1}</span>
                        <p className="text-gray-800 font-medium flex-1">{video.title}</p>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{fmtCompact(video.views)} views</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Categories</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {topCategories.length > 0 ? (
              <div className="space-y-2">
                {topCategories.slice(0, 10).map((cat, index) => (
                  <div
                    key={cat.category}
                    className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <span className="text-gray-500 font-semibold text-sm">#{index + 1}</span>
                        <p className="text-gray-800 font-medium capitalize flex-1">{cat.category}</p>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">{fmtCompact(cat.total_views)} views</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

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
