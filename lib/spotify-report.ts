import jsPDF from "jspdf"

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
  return Object.entries(yearlyMap)
    .sort(([yearA], [yearB]) => yearA.localeCompare(yearB))
    .map(([year, streams]) => ({ year, streams }))
}

export const generateSpotifyCSV = (data: {
  overview: any
  topTracks: any[]
  monthly: any[]
  dailyListeners: any[]
  followerGrowth: any[]
  streamsGrowth: any[]
  followersGrowthPct: any[]
}) => {
  const csvContent = [
    ["Spotify Analytics Dashboard Report"],
    [`Generated on: ${new Date().toLocaleString()}`],
    [
      `Date Range: ${data.monthly.length > 0 ? data.monthly[0].date : "N/A"} to ${
        data.monthly.length > 0 ? data.monthly[data.monthly.length - 1].date : "N/A"
      }`,
    ],
    [],
    ["OVERVIEW METRICS"],
    ["Metric", "Value"],
    ["Total Streams", fmtInt(data.overview.total_streams)],
    ["Total Listeners", fmtCompact(data.overview.total_listeners)],
    ["Total Followers", fmtInt(data.overview.total_followers)],
    ["Top Song", data.overview.top_track || "—"],
    ["Top Song Streams", fmtInt(data.overview.top_track_streams)],
    [],
    ["TOP 10 TRACKS BY STREAMS"],
    ["Rank", "Track Name", "Streams"],
    ...data.topTracks.slice(0, 10).map((track, idx) => [idx + 1, track.track, fmtInt(track.streams)]),
    [],
    ["YEARLY STREAMS TREND"],
    ["Year", "Streams"],
    ...aggregateToYearly(data.monthly).map((item) => [item.year, fmtInt(item.streams)]),
    [],
    ["DAILY LISTENERS SUMMARY"],
    ["Date", "Listeners"],
    ...data.dailyListeners.slice(-30).map((item) => [item.date, fmtInt(item.listeners)]),
    [],
    ["FOLLOWER GROWTH"],
    ["Date", "Total Followers"],
    ...data.followerGrowth.slice(-30).map((item) => [item.date, fmtInt(item.total_followers)]),
    [],
    ["STREAMS GROWTH PERCENTAGE"],
    ["Date", "Growth %"],
    ...data.streamsGrowth.slice(-30).map((item) => [item.date, fmtPct(item.growth_pct)]),
    [],
    ["FOLLOWERS GROWTH PERCENTAGE"],
    ["Date", "Growth %"],
    ...data.followersGrowthPct.slice(-30).map((item) => [item.date, fmtPct(item.growth_pct)]),
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

export const generateSpotifyPDF = (data: {
  overview: any
  topTracks: any[]
  monthly: any[]
  dailyListeners: any[]
  followerGrowth: any[]
  streamsGrowth: any[]
  followersGrowthPct: any[]
}) => {
  const yearlyTrend = aggregateToYearly(data.monthly)
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

  const drawTable = (head: string[], body: (string | number)[][], startY: number, maxRows = 20) => {
    let currentY = startY
    const cellHeight = 6
    const colWidths =
      head.length === 3 ? [15, 105, 40] : head.length === 2 ? [50, 80] : Array(head.length).fill(180 / head.length)

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

  pdf.setFontSize(18)
  pdf.setTextColor(12, 77, 143)
  pdf.setFont(undefined, "bold")
  pdf.text("Spotify Analytics Dashboard Report", 10, yPosition)
  yPosition += 12

  pdf.setFontSize(10)
  pdf.setTextColor(0, 0, 0)
  pdf.setFont(undefined, "normal")
  pdf.text(`Generated on: ${new Date().toLocaleString()}`, 10, yPosition)
  yPosition += 5
  pdf.text(
    `Date Range: ${data.monthly.length > 0 ? data.monthly[0].date : "N/A"} to ${
      data.monthly.length > 0 ? data.monthly[data.monthly.length - 1].date : "N/A"
    }`,
    10,
    yPosition,
  )
  yPosition += 12

  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Overview Metrics", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Metric", "Value"],
    [
      ["Total Streams", fmtInt(data.overview.total_streams)],
      ["Total Listeners", fmtCompact(data.overview.total_listeners)],
      ["Total Followers", fmtInt(data.overview.total_followers)],
      ["Top Song", data.overview.top_track || "—"],
      ["Top Song Streams", fmtInt(data.overview.top_track_streams)],
    ],
    yPosition,
    5,
  )
  yPosition += 8

  checkPageSpace(70)
  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Top 10 Tracks by Streams", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Rank", "Track Name", "Streams"],
    data.topTracks.slice(0, 10).map((track, idx) => [idx + 1, track.track, fmtInt(track.streams)]),
    yPosition,
    10,
  )
  yPosition += 8

  checkPageSpace(60)
  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Yearly Streams Trend", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Year", "Streams"],
    yearlyTrend.map((item) => [item.year, fmtInt(item.streams)]),
    yPosition,
    yearlyTrend.length,
  )
  yPosition += 8

  checkPageSpace(60)
  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Daily Listeners (Last 30 Days)", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Date", "Listeners"],
    data.dailyListeners.slice(-30).map((item) => [item.date, fmtInt(item.listeners)]),
    yPosition,
    15,
  )
  yPosition += 8

  checkPageSpace(60)
  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Follower Growth (Last 30 Days)", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Date", "Total Followers"],
    data.followerGrowth.slice(-30).map((item) => [item.date, fmtInt(item.total_followers)]),
    yPosition,
    15,
  )
  yPosition += 8

  checkPageSpace(60)
  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Streams Growth Percentage (Last 30 Days)", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Date", "Growth %"],
    data.streamsGrowth.slice(-30).map((item) => [item.date, fmtPct(item.growth_pct)]),
    yPosition,
    15,
  )
  yPosition += 8

  checkPageSpace(60)
  pdf.setFontSize(14)
  pdf.setFont(undefined, "bold")
  pdf.setTextColor(12, 77, 143)
  pdf.text("Followers Growth Percentage (Last 30 Days)", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Date", "Growth %"],
    data.followersGrowthPct.slice(-30).map((item) => [item.date, fmtPct(item.growth_pct)]),
    yPosition,
    15,
  )

  pdf.save(`spotify-report-${new Date().toISOString().split("T")[0]}.pdf`)
}
