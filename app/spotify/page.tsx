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
import jsPDF from "jspdf"

export default function SpotifyDashboard() {
  const [overview, setOverview] = useState<any>({})
  const [topTracks, setTopTracks] = useState<any[]>([])
  const [monthly, setMonthly] = useState<any[]>([])
  const [engagement, setEngagement] = useState<any[]>([])
  const [dailyStreams, setDailyStreams] = useState<any[]>([])
  const [songReleases, setSongReleases] = useState<any[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generatingReport, setGeneratingReport] = useState(false)

  useEffect(() => {
    const fetchAllData = async () => {
      try {
        setLoading(true)
        setError(null)

        const [overviewRes, topTracksRes, monthlyRes, engagementRes, dailyStreamsRes] = await Promise.all([
          fetch("/api/analytics/spotify/overview"),
          fetch("/api/analytics/spotify/top-tracks"),
          fetch("/api/analytics/spotify/monthly"),
          fetch("/api/analytics/spotify/engagement"),
          fetch("/api/analytics/spotify/daily-streams-with-releases"),
        ])

        if (!overviewRes.ok) throw new Error("Failed to fetch overview")
        const overviewData = await overviewRes.json()

        setOverview({
          total_streams: Number(overviewData.overview.total_streams),
          total_followers: Number(overviewData.overview.total_followers),
          total_listeners: Number(overviewData.overview.total_listeners),
          top_tracks_count: Number(overviewData.overview.top_tracks_count),
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

  const aggregateToYearly = (data: any[]) => {
    const yearlyMap: { [key: string]: number } = {}
    data.forEach((item) => {
      const year = item.date ? item.date.substring(0, 4) : "Unknown"
      yearlyMap[year] = (yearlyMap[year] || 0) + (item.streams || 0)
    })
    return Object.entries(yearlyMap).map(([year, streams]) => ({ year, streams }))
  }

  const generateCSVReport = () => {
    const csvContent = [
      ["Spotify Analytics Dashboard Report"],
      [`Generated on: ${new Date().toLocaleString()}`],
      [
        `Date Range: ${dailyStreams.length > 0 ? dailyStreams[0].date : "N/A"} to ${dailyStreams.length > 0 ? dailyStreams[dailyStreams.length - 1].date : "N/A"}`,
      ],
      [],
      ["OVERVIEW METRICS"],
      ["Total Streams", fmtInt(overview.total_streams)],
      ["Total Listeners", fmtCompact(overview.total_listeners)],
      ["Total Followers", fmtInt(overview.total_followers)],
      ["Top Tracks Count", fmtInt(overview.top_tracks_count)],
      [],
      ["TOP 10 TRACKS BY STREAMS"],
      ["Rank", "Track Name", "Streams"],
      ...topTracks.slice(0, 10).map((track, idx) => [idx + 1, track.track, fmtInt(track.streams)]),
    ]

    const csvString = csvContent.map((row) => row.map((cell) => `"${cell}"`).join(",")).join("\n")
    const blob = new Blob([csvString], { type: "text/csv;charset=utf-8;" })
    const link = document.createElement("a")
    const url = URL.createObjectURL(blob)
    link.setAttribute("href", url)
    link.setAttribute("download", `spotify-report-${new Date().toISOString().split("T")[0]}.csv`)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const generatePDFReport = () => {
    const yearlyTrend = aggregateToYearly(monthly)
    const pdf = new jsPDF()
    const pageWidth = pdf.internal.pageSize.getWidth()
    const pageHeight = pdf.internal.pageSize.getHeight()
    let yPosition = 10

    const addNewPage = () => {
      pdf.addPage()
      yPosition = 15
    }

    const checkPageSpace = (spaceNeeded: number) => {
      if (yPosition + spaceNeeded > pageHeight - 15) {
        addNewPage()
      }
    }

    // Created manual table drawing function to replace autoTable
    const drawTable = (head: string[], body: (string | number)[][], startY: number, maxRows = 20) => {
      let currentY = startY
      const cellHeight = 6
      const colWidths = head.length === 3 ? [15, 105, 40] : [50, 80]
      const pageWidth = pdf.internal.pageSize.getWidth()

      // Draw header
      pdf.setFillColor(12, 77, 143)
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

      // Draw body rows
      pdf.setTextColor(0, 0, 0)
      pdf.setFont(undefined, "normal")
      pdf.setFontSize(9)

      body.slice(0, maxRows).forEach((row, idx) => {
        if (currentY + cellHeight > pageHeight - 15) {
          addNewPage()
          currentY = yPosition
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

    // Title
    pdf.setFontSize(18)
    pdf.setTextColor(12, 77, 143)
    pdf.setFont(undefined, "bold")
    pdf.text("Spotify Analytics Dashboard Report", 10, yPosition)
    yPosition += 12

    // Header info
    pdf.setFontSize(10)
    pdf.setTextColor(0, 0, 0)
    pdf.setFont(undefined, "normal")
    pdf.text(`Generated on: ${new Date().toLocaleString()}`, 10, yPosition)
    yPosition += 5
    pdf.text(
      `Date Range: ${dailyStreams.length > 0 ? dailyStreams[0].date : "N/A"} to ${
        dailyStreams.length > 0 ? dailyStreams[dailyStreams.length - 1].date : "N/A"
      }`,
      10,
      yPosition,
    )
    yPosition += 12

    // Overview Metrics
    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(12, 77, 143)
    pdf.text("Overview Metrics", 10, yPosition)
    yPosition += 8
    pdf.setFontSize(10)
    pdf.setFont(undefined, "normal")
    pdf.setTextColor(0, 0, 0)
    pdf.text(`Total Streams: ${fmtInt(overview.total_streams)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Total Listeners: ${fmtCompact(overview.total_listeners)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Total Followers: ${fmtInt(overview.total_followers)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Top Tracks Count: ${fmtInt(overview.top_tracks_count)}`, 10, yPosition)
    yPosition += 12

    // Top Tracks Table
    checkPageSpace(70)
    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(12, 77, 143)
    pdf.text("Top 10 Tracks by Streams", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Rank", "Track Name", "Streams"],
      topTracks.slice(0, 10).map((track, idx) => [idx + 1, track.track, fmtInt(track.streams)]),
      yPosition,
      10,
    )
    yPosition += 8

    pdf.save(`spotify-report-${new Date().toISOString().split("T")[0]}.pdf`)
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
          <h1 className="text-3xl font-bold">Spotify Analytics Dashboard</h1>
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
            <h3 className="text-sm font-medium text-gray-600">Top Tracks</h3>
            <p className="text-3xl font-bold text-gray-900 mt-2">{fmtInt(overview.top_tracks_count)}</p>
          </div>
        </section>

        {/* Top Tracks Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Top 10 Tracks by Streams</h2>
          <div className="h-96">
            {topTracks.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={topTracks} layout="vertical" margin={{ top: 5, right: 30, left: 200, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" tickFormatter={fmtCompact} />
                  <YAxis dataKey="track" type="category" width={190} tick={{ fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmtInt(v as number)} />
                  <Bar dataKey="streams" fill="#0c4d8f" />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="text-gray-500">No data available</div>
            )}
          </div>
        </section>

        {/* Daily Streams with Song Releases Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4 text-[#123458]">Daily Streams with Song Releases</h2>
          <div className="h-96">
            {dailyStreams.length > 0 && songReleases.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dailyStreams} margin={{ top: 20, right: 30, left: 40, bottom: 60 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="date"
                    tickFormatter={(value) =>
                      new Date(value).toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" })
                    }
                    angle={-45}
                    textAnchor="end"
                  />
                  <YAxis />
                  <Tooltip formatter={(value) => fmtInt(value)} />
                  <Legend />
                  <Line type="monotone" dataKey="streams" stroke="#1DB954" strokeWidth={2} name="Daily Streams" />

                  {/* Scatter plot for Song Releases */}
                  <ScatterChart>
                    <Scatter
                      name="Song Releases"
                      data={songReleases.map((release) => ({
                        date: release.release_date,
                        streams: dailyStreams.find((stream) => stream.date === release.release_date)?.streams || 0,
                      }))}
                      fill="#FF4444"
                      line
                      shape="circle"
                      strokeWidth={2}
                    />
                  </ScatterChart>
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
