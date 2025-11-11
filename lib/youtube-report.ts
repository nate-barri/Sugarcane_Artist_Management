import jsPDF from "jspdf"

const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
const fmtCompact = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`
const fmtPct = (n?: number) => {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—"
  return `${n.toFixed(2)}%`
}

export async function generateYouTubeCSV(
  startDate: string,
  endDate: string,
  overview: any,
  topVideos: any[],
  topCategories: any[],
  monthly: any[],
  duration: any[],
  dayOfWeek: any[],
  contentType: any[],
  contentDistribution: any[],
  contentEngagement: any[],
) {
  let csv = "YouTube Analytics Report\n"
  csv += `Generated on: ${new Date().toISOString()}\n`
  csv += `Date Range: ${startDate} to ${endDate}\n\n`

  csv += "OVERVIEW METRICS\n"
  csv += "Metric,Value\n"
  csv += `Total Videos,${overview.total_videos || 0}\n`
  csv += `Total Views,${overview.total_views || 0}\n`
  csv += `Total Likes,${overview.total_likes || 0}\n`
  csv += `Watch Time (hrs),${overview.total_watch_time || 0}\n`
  csv += `Engagement Rate (%),${overview.engagement_rate || 0}\n\n`

  csv += "TOP 10 CATEGORIES\n"
  csv += "Rank,Category,Total Views\n"
  topCategories.slice(0, 10).forEach((cat, idx) => {
    csv += `${idx + 1},"${cat.category}",${cat.total_views}\n`
  })
  csv += "\n"

  csv += "TOP 15 VIDEOS\n"
  csv += "Rank,Video Title,Views,Likes,Watch Time,Category\n"
  topVideos.slice(0, 15).forEach((vid, idx) => {
    csv += `${idx + 1},"${vid.title}",${vid.views},${vid.likes || 0},${vid.watch_time || 0},"${vid.category}"\n`
  })
  csv += "\n"

  csv += "MONTHLY PERFORMANCE\n"
  csv += "Month,Videos Uploaded,Total Views\n"
  monthly.forEach((m) => {
    csv += `${m.publish_year}-${String(m.publish_month).padStart(2, "0")},${m.video_count},${m.total_views}\n`
  })
  csv += "\n"

  csv += "DURATION PERFORMANCE\n"
  csv += "Duration Bucket,Avg Engagement Rate (%)\n"
  duration.forEach((d) => {
    csv += `"${d.duration_bucket}",${d.avg_engagement_rate}\n`
  })
  csv += "\n"

  csv += "DAY OF WEEK PERFORMANCE\n"
  csv += "Day,Mean Engagement,Median Engagement\n"
  dayOfWeek.forEach((d) => {
    csv += `"${d.day_of_week}",${d.mean_engagement},${d.median_engagement}\n`
  })
  csv += "\n"

  csv += "CONTENT TYPE PERFORMANCE\n"
  csv += "Content Type,Avg Views,Avg Engagement Rate (%)\n"
  contentType.forEach((ct) => {
    const engagement = contentEngagement.find((ce) => ce.content_type === ct.content_type)
    csv += `"${ct.content_type}",${ct.avg_views},${engagement?.avg_engagement_rate || 0}\n`
  })
  csv += "\n"

  csv += "CONTENT DISTRIBUTION\n"
  csv += "Content Type,Video Count,Percentage (%)\n"
  contentDistribution.forEach((cd) => {
    csv += `"${cd.content_type}",${cd.video_count},${cd.percentage}\n`
  })

  return csv
}

export function generateYouTubePDF(
  startDate: string,
  endDate: string,
  overview: any,
  topVideos: any[],
  topCategories: any[],
  monthly: any[],
  duration: any[],
  dayOfWeek: any[],
  contentType: any[],
  contentDistribution: any[],
  contentEngagement: any[],
) {
  const doc = new jsPDF()
  let yPosition = 10

  const drawTable = (title: string, headers: string[], rows: any[][]) => {
    if (yPosition > 270) {
      doc.addPage()
      yPosition = 10
    }

    doc.setFontSize(12)
    doc.setFont("helvetica", "bold")
    doc.text(title, 10, yPosition)
    yPosition += 7

    doc.setFontSize(9)
    doc.setFont("helvetica", "bold")

    const colWidth = (190 - 20) / headers.length
    const headerHeight = 5

    // Draw header background
    doc.setDrawColor(0, 51, 102)
    doc.setFillColor(0, 51, 102)
    doc.rect(10, yPosition - 4, 190, headerHeight, "F")

    // Draw headers with white text
    doc.setTextColor(255, 255, 255)
    headers.forEach((header, i) => {
      doc.text(header, 11 + i * colWidth, yPosition, { maxWidth: colWidth - 2, align: "left" })
    })
    doc.setTextColor(0, 0, 0)
    yPosition += 6

    doc.setFont("helvetica", "normal")
    doc.setFontSize(8)
    rows.forEach((row, rowIdx) => {
      if (yPosition > 270) {
        doc.addPage()
        yPosition = 10
      }

      let maxLines = 1
      row.forEach((cell) => {
        const lines = doc.splitTextToSize(String(cell), colWidth - 2)
        maxLines = Math.max(maxLines, lines.length)
      })

      const rowHeight = maxLines * 3 + 2

      // Draw alternating row backgrounds
      if (rowIdx % 2 === 1) {
        doc.setFillColor(240, 240, 240)
        doc.rect(10, yPosition - 4, 190, rowHeight, "F")
      }

      // Draw cell text
      const cellY = yPosition
      row.forEach((cell, i) => {
        const lines = doc.splitTextToSize(String(cell), colWidth - 2)
        lines.forEach((line, lineIdx) => {
          doc.text(line, 11 + i * colWidth, cellY + lineIdx * 3)
        })
      })

      yPosition += rowHeight + 1
    })
    yPosition += 3
  }

  // Title and metadata
  doc.setFontSize(16)
  doc.setFont("helvetica", "bold")
  doc.text("YouTube Analytics Dashboard Report", 10, yPosition)
  yPosition += 10

  doc.setFontSize(9)
  doc.setFont("helvetica", "normal")
  doc.text(`Generated on: ${new Date().toISOString()}`, 10, yPosition)
  yPosition += 5
  doc.text(`Date Range: ${startDate} to ${endDate}`, 10, yPosition)
  yPosition += 10

  // Overview Metrics
  const overviewRows = [
    ["Total Videos", String(overview.total_videos || 0)],
    ["Total Views", String(overview.total_views || 0)],
    ["Total Likes", String(overview.total_likes || 0)],
    ["Watch Time (hrs)", String(overview.total_watch_time || 0)],
    ["Engagement Rate (%)", String((overview.engagement_rate || 0).toFixed(2))],
  ]
  drawTable("Overview Metrics", ["Metric", "Value"], overviewRows)

  // Top 10 Categories
  const categoryRows = topCategories
    .slice(0, 10)
    .map((cat, idx) => [String(idx + 1), cat.category, String(cat.total_views)])
  drawTable("Top 10 Categories", ["Rank", "Category", "Total Views"], categoryRows)

  // Top 15 Videos
  const videoRows = topVideos
    .slice(0, 15)
    .map((vid, idx) => [
      String(idx + 1),
      vid.title,
      String(vid.views),
      String(vid.likes || 0),
      String(vid.watch_time || 0),
      vid.category,
    ])
  drawTable("Top 15 Videos by Views", ["Rank", "Title", "Views", "Likes", "Watch Time", "Category"], videoRows)

  // Monthly Performance
  const monthlyRows = monthly.map((m) => [
    `${m.publish_year}-${String(m.publish_month).padStart(2, "0")}`,
    String(m.video_count),
    String((m.total_views / 1_000_000).toFixed(2)),
  ])
  drawTable("Monthly Upload Volume & Views", ["Month", "Videos Uploaded", "Total Views (M)"], monthlyRows)

  // Duration Performance
  const durationRows = duration.map((d) => [d.duration_bucket, String(d.avg_engagement_rate.toFixed(2))])
  drawTable("Average Engagement Rate by Duration", ["Duration Bucket", "Avg Engagement Rate (%)"], durationRows)

  // Day of Week Performance
  const dayRows = dayOfWeek.map((d) => [d.day_of_week, String(d.mean_engagement), String(d.median_engagement)])
  drawTable("Posting Day Performance", ["Day of Week", "Mean Engagement", "Median Engagement"], dayRows)

  // Content Type Performance
  const contentRows = contentType.map((ct) => {
    const engagement = contentEngagement.find((ce) => ce.content_type === ct.content_type)
    return [ct.content_type, String(ct.avg_views), String((engagement?.avg_engagement_rate || 0).toFixed(2))]
  })
  drawTable(
    "Average Views & Engagement Rate by Content Type",
    ["Content Type", "Avg Views", "Avg Engagement Rate (%)"],
    contentRows,
  )

  // Content Distribution
  const distRows = contentDistribution.map((cd) => [
    cd.content_type,
    String(cd.video_count),
    String(cd.percentage.toFixed(2)),
  ])
  drawTable("Content Type Distribution", ["Content Type", "Video Count", "Percentage (%)"], distRows)

  return doc
}
