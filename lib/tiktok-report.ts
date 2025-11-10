import { jsPDF } from "jspdf"

const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
const fmtPct = (n?: number) => {
  if (typeof n !== "number" || !Number.isFinite(n)) return "—"
  return `${n.toFixed(2)}%`
}
const fmtCompact = (n: number) =>
  n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`

export const generateTikTokCSV = (data: {
  startDate: string
  endDate: string
  overview: any
  topVideos: any[]
  monthly: any[]
  postType: any[]
  duration: any[]
  sound: any[]
  engagementDist: any
  engagementRates: any
}) => {
  const timestamp = new Date().toLocaleString()
  let csvContent = "TikTok Analytics Dashboard Report\n"
  csvContent += `Generated on: ${timestamp}\n`
  csvContent += `Date Range: ${data.startDate} to ${data.endDate}\n\n`

  csvContent += "OVERVIEW METRICS\n"
  csvContent += "Metric,Value\n"
  csvContent += `Total Videos,${data.overview.total_videos || 0}\n`
  csvContent += `Total Views,${data.overview.total_views || 0}\n`
  csvContent += `Total Likes,${data.overview.total_likes || 0}\n`
  csvContent += `Total Shares,${data.overview.total_shares || 0}\n`
  csvContent += `Engagement Rate,${data.overview.engagement_rate || 0}%\n\n`

  csvContent += "TOP 10 VIDEOS BY VIEWS\n"
  csvContent += "Rank,Video Title,Views\n"
  data.topVideos.slice(0, 10).forEach((video, idx) => {
    csvContent += `${idx + 1},"${video.title}",${video.views}\n`
  })
  csvContent += "\n"

  csvContent += "MONTHLY TRENDS\n"
  csvContent += "Month,Total Views\n"
  data.monthly.forEach((m) => {
    const month = `${m.publish_year}-${String(m.publish_month).padStart(2, "0")}`
    csvContent += `${month},${m.total_views}\n`
  })
  csvContent += "\n"

  const yearlyTrends = data.monthly.reduce((acc: any, m: any) => {
    const year = m.publish_year
    const existing = acc.find((item: any) => item.year === year)
    if (existing) {
      existing.total_views += m.total_views
    } else {
      acc.push({ year, total_views: m.total_views })
    }
    return acc
  }, [])

  csvContent += "YEARLY VIEWS TREND\n"
  csvContent += "Year,Views\n"
  yearlyTrends.forEach((item: any) => {
    csvContent += `${item.year},${item.total_views}\n`
  })
  csvContent += "\n"

  csvContent += "PERFORMANCE BY POST TYPE\n"
  csvContent += "Post Type,Average Views,Average Likes\n"
  data.postType.forEach((pt) => {
    csvContent += `${pt.post_type},${pt.avg_views},${pt.avg_likes}\n`
  })
  csvContent += "\n"

  csvContent += "PERFORMANCE BY DURATION\n"
  csvContent += "Duration Bucket,Video Count,Average Views,Average Likes,Average Engagement\n"
  data.duration.forEach((d) => {
    csvContent += `${d.duration_bucket},${d.video_count},${d.avg_views},${d.avg_likes},${d.avg_engagement}\n`
  })
  csvContent += "\n"

  csvContent += "TOP 20 SOUNDS BY VIEWS\n"
  csvContent += "Rank,Sound Category,Views\n"
  data.sound.slice(0, 20).forEach((s, idx) => {
    csvContent += `${idx + 1},"${s.sound_category}",${s.total_views}\n`
  })
  csvContent += "\n"

  csvContent += "ENGAGEMENT DISTRIBUTION\n"
  csvContent += "Metric,Value\n"
  csvContent += `Total Views,${data.engagementDist?.total_views || 0}\n`
  csvContent += `Total Likes,${data.engagementDist?.total_likes || 0}\n`
  csvContent += `Total Shares,${data.engagementDist?.total_shares || 0}\n`
  csvContent += `Total Comments,${data.engagementDist?.total_comments || 0}\n`
  csvContent += `Total Saves,${data.engagementDist?.total_saves || 0}\n\n`

  csvContent += "ENGAGEMENT RATE DISTRIBUTION\n"
  csvContent += "Rate Type,Percentage\n"
  csvContent += `Like Rate,${data.engagementRates?.like_rate || 0}%\n`
  csvContent += `Share Rate,${data.engagementRates?.share_rate || 0}%\n`
  csvContent += `Comment Rate,${data.engagementRates?.comment_rate || 0}%\n`
  csvContent += `Engagement Rate,${data.engagementRates?.engagement_rate || 0}%\n`

  const element = document.createElement("a")
  element.setAttribute("href", "data:text/csv;charset=utf-8," + encodeURIComponent(csvContent))
  element.setAttribute("download", `tiktok-report-${new Date().toISOString().split("T")[0]}.csv`)
  element.style.display = "none"
  document.body.appendChild(element)
  element.click()
  document.body.removeChild(element)
}

export const generateTikTokPDF = (data: {
  startDate: string
  endDate: string
  overview: any
  topVideos: any[]
  monthly: any[]
  postType: any[]
  duration: any[]
  sound: any[]
  engagementDist: any
  engagementRates: any
}) => {
  const doc = new jsPDF()
  const pageHeight = doc.internal.pageSize.getHeight()
  const timestamp = new Date().toLocaleString()
  let yPosition = 10

  const addNewPage = () => {
    doc.addPage()
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
      head.length === 2
        ? [80, 50]
        : head.length === 3
          ? [40, 90, 40]
          : head.length === 4
            ? [35, 75, 40, 40]
            : [20, 80, 50, 40]

    doc.setFillColor(12, 77, 143)
    doc.setTextColor(255, 255, 255)
    doc.setFontSize(10)
    doc.setFont(undefined, "bold")
    let xPos = 10
    head.forEach((h, i) => {
      doc.rect(xPos, currentY, colWidths[i], cellHeight, "F")
      doc.text(h, xPos + 2, currentY + 4)
      xPos += colWidths[i]
    })
    currentY += cellHeight

    doc.setTextColor(0, 0, 0)
    doc.setFont(undefined, "normal")
    doc.setFontSize(9)
    body.slice(0, maxRows).forEach((row, idx) => {
      if (currentY + cellHeight > pageHeight - 15) {
        addNewPage()
        currentY = yPosition
      }
      if (idx % 2 === 0) {
        doc.setFillColor(240, 240, 240)
        xPos = 10
        head.forEach((_, i) => {
          doc.rect(xPos, currentY, colWidths[i], cellHeight, "F")
          xPos += colWidths[i]
        })
      }
      xPos = 10
      row.forEach((cell, i) => {
        doc.text(String(cell), xPos + 2, currentY + 4)
        xPos += colWidths[i]
      })
      currentY += cellHeight
    })
    return currentY
  }

  doc.setFontSize(18)
  doc.setTextColor(12, 77, 143)
  doc.setFont(undefined, "bold")
  doc.text("TikTok Analytics Dashboard Report", 10, yPosition)
  yPosition += 12

  doc.setFontSize(10)
  doc.setTextColor(0, 0, 0)
  doc.setFont(undefined, "normal")
  doc.text(`Generated on: ${timestamp}`, 10, yPosition)
  yPosition += 5
  doc.text(`Date Range: ${data.startDate} to ${data.endDate}`, 10, yPosition)
  yPosition += 12

  doc.setFontSize(14)
  doc.setFont(undefined, "bold")
  doc.setTextColor(12, 77, 143)
  doc.text("Overview Metrics", 10, yPosition)
  yPosition += 8
  doc.setFontSize(10)
  doc.setFont(undefined, "normal")
  doc.setTextColor(0, 0, 0)
  doc.text(`Total Videos: ${fmtInt(data.overview.total_videos)}`, 10, yPosition)
  yPosition += 5
  doc.text(`Total Views: ${fmtCompact(data.overview.total_views)}`, 10, yPosition)
  yPosition += 5
  doc.text(`Total Likes: ${fmtCompact(data.overview.total_likes)}`, 10, yPosition)
  yPosition += 5
  doc.text(`Total Shares: ${fmtCompact(data.overview.total_shares)}`, 10, yPosition)
  yPosition += 5
  doc.text(`Engagement Rate: ${fmtPct(data.overview.engagement_rate)}`, 10, yPosition)
  yPosition += 12

  checkPageSpace(70)
  doc.setFontSize(14)
  doc.setFont(undefined, "bold")
  doc.setTextColor(12, 77, 143)
  doc.text("Top 10 Videos by Views", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Rank", "Video Title", "Views"],
    data.topVideos.slice(0, 10).map((video, idx) => [idx + 1, video.title.substring(0, 40), fmtInt(video.views)]),
    yPosition,
    10,
  )
  yPosition += 8

  const yearlyTrends = data.monthly.reduce((acc: any, m: any) => {
    const year = m.publish_year
    const existing = acc.find((item: any) => item.year === year)
    if (existing) {
      existing.total_views += m.total_views
    } else {
      acc.push({ year, total_views: m.total_views })
    }
    return acc
  }, [])

  checkPageSpace(70)
  doc.setFontSize(14)
  doc.setFont(undefined, "bold")
  doc.setTextColor(12, 77, 143)
  doc.text("Yearly Views Trend", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Year", "Views"],
    yearlyTrends.map((item: any) => [item.year, fmtCompact(item.total_views)]),
    yPosition,
  )
  yPosition += 8

  if (data.postType.length > 0) {
    checkPageSpace(70)
    doc.setFontSize(14)
    doc.setFont(undefined, "bold")
    doc.setTextColor(12, 77, 143)
    doc.text("Performance by Post Type", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Post Type", "Avg Views", "Avg Likes"],
      data.postType.map((pt: any) => [pt.post_type, fmtInt(pt.avg_views), fmtInt(pt.avg_likes)]),
      yPosition,
    )
    yPosition += 8
  }

  if (data.duration.length > 0) {
    checkPageSpace(70)
    doc.setFontSize(14)
    doc.setFont(undefined, "bold")
    doc.setTextColor(12, 77, 143)
    doc.text("Performance by Duration", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Duration", "Videos", "Avg Views", "Avg Likes"],
      data.duration.map((d: any) => [d.duration_bucket, `${d.video_count}`, fmtInt(d.avg_views), fmtInt(d.avg_likes)]),
      yPosition,
    )
    yPosition += 8
  }

  if (data.sound.length > 0) {
    checkPageSpace(70)
    doc.setFontSize(14)
    doc.setFont(undefined, "bold")
    doc.setTextColor(12, 77, 143)
    doc.text("Top 20 Sounds by Views", 10, yPosition)
    yPosition += 8
    yPosition = drawTable(
      ["Rank", "Sound", "Views"],
      data.sound
        .slice(0, 20)
        .map((s: any, idx: number) => [idx + 1, s.sound_category.substring(0, 40), fmtInt(s.total_views)]),
      yPosition,
      20,
    )
    yPosition += 8
  }

  checkPageSpace(70)
  doc.setFontSize(14)
  doc.setFont(undefined, "bold")
  doc.setTextColor(12, 77, 143)
  doc.text("Engagement Distribution", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Metric", "Value"],
    [
      ["Total Views", fmtCompact(data.engagementDist?.total_views || 0)],
      ["Total Likes", fmtCompact(data.engagementDist?.total_likes || 0)],
      ["Total Shares", fmtCompact(data.engagementDist?.total_shares || 0)],
      ["Total Comments", fmtCompact(data.engagementDist?.total_comments || 0)],
      ["Total Saves", fmtCompact(data.engagementDist?.total_saves || 0)],
    ],
    yPosition,
  )
  yPosition += 8

  checkPageSpace(50)
  doc.setFontSize(14)
  doc.setFont(undefined, "bold")
  doc.setTextColor(12, 77, 143)
  doc.text("Engagement Rates", 10, yPosition)
  yPosition += 8
  yPosition = drawTable(
    ["Rate Type", "Percentage"],
    [
      ["Like Rate", `${fmtPct(data.engagementRates?.like_rate)}`],
      ["Share Rate", `${fmtPct(data.engagementRates?.share_rate)}`],
      ["Comment Rate", `${fmtPct(data.engagementRates?.comment_rate)}`],
      ["Engagement Rate", `${fmtPct(data.engagementRates?.engagement_rate)}`],
    ],
    yPosition,
  )
  doc.save(`tiktok-report-${new Date().toISOString().split("T")[0]}.pdf`)
}
