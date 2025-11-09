"use client"

import Sidebar from "@/components/sidebar"
import { useState } from "react"
import jsPDF from "jspdf"

export default function FacebookDashboard() {
  const [generatingReport, setGeneratingReport] = useState(false)

  // Mock data for Facebook dashboard
  const overview = {
    page_likes: 125430,
    post_reach: 856000,
    engagement_rate: 4.2,
    page_views: 542000,
  }

  const topPosts = [
    { rank: 1, title: "Summer Announcement", engagement: 8500 },
    { rank: 2, title: "New Product Launch", engagement: 7200 },
    { rank: 3, title: "Customer Testimonial", engagement: 6100 },
    { rank: 4, title: "Behind the Scenes", engagement: 5800 },
    { rank: 5, title: "Weekly Tips", engagement: 5200 },
    { rank: 6, title: "Community Update", engagement: 4900 },
    { rank: 7, title: "Flash Sale", engagement: 4600 },
    { rank: 8, title: "Event Recap", engagement: 4300 },
    { rank: 9, title: "User Generated Content", engagement: 3900 },
    { rank: 10, title: "Holiday Special", engagement: 3600 },
  ]

  const demographicsData = [
    { age_group: "18-24", percentage: 22 },
    { age_group: "25-34", percentage: 35 },
    { age_group: "35-44", percentage: 28 },
    { age_group: "45-54", percentage: 12 },
    { age_group: "55+", percentage: 3 },
  ]

  const fmtInt = (n?: number) => (typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—")
  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—"
    return `${n.toFixed(2)}%`
  }
  const fmtCompact = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)}M` : n >= 1_000 ? `${(n / 1_000).toFixed(1)}K` : `${n}`

  const drawTable = (head: string[], body: (string | number)[][], startY: number, pdf: jsPDF) => {
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
      [],
      ["OVERVIEW METRICS"],
      ["Page Likes", fmtInt(overview.page_likes)],
      ["Post Reach", fmtCompact(overview.post_reach)],
      ["Engagement Rate", fmtPct(overview.engagement_rate)],
      ["Page Views", fmtCompact(overview.page_views)],
      [],
      ["TOP 10 POSTS BY ENGAGEMENT"],
      ["Rank", "Post Title", "Engagement"],
      ...topPosts.map((post) => [post.rank, post.title, fmtInt(post.engagement)]),
      [],
      ["AUDIENCE DEMOGRAPHICS"],
      ["Age Group", "Percentage"],
      ...demographicsData.map((demo) => [demo.age_group, fmtPct(demo.percentage)]),
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
    yPosition += 12

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Overview Metrics", 10, yPosition)
    yPosition += 8

    pdf.setFontSize(10)
    pdf.setFont(undefined, "normal")
    pdf.setTextColor(0, 0, 0)
    pdf.text(`Page Likes: ${fmtInt(overview.page_likes)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Post Reach: ${fmtCompact(overview.post_reach)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Engagement Rate: ${fmtPct(overview.engagement_rate)}`, 10, yPosition)
    yPosition += 5
    pdf.text(`Page Views: ${fmtCompact(overview.page_views)}`, 10, yPosition)
    yPosition += 12

    pdf.setFontSize(14)
    pdf.setFont(undefined, "bold")
    pdf.setTextColor(18, 52, 88)
    pdf.text("Top 10 Posts by Engagement", 10, yPosition)
    yPosition += 8

    yPosition = drawTable(
      ["Rank", "Post Title", "Engagement"],
      topPosts.map((post) => [post.rank, post.title, fmtInt(post.engagement)]),
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
    pdf.text("Audience Demographics", 10, yPosition)
    yPosition += 8

    yPosition = drawTable(
      ["Age Group", "Percentage"],
      demographicsData.map((demo) => [demo.age_group, fmtPct(demo.percentage)]),
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

  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-gray-800">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">Meta</h1>
        </header>

        {/* Key Metrics Section for Facebook */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Page Likes</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtInt(overview.page_likes)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Post Reach</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtCompact(overview.post_reach)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Engagement Rate</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtPct(overview.engagement_rate)}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Page Views</h2>
            <p className="text-3xl font-bold text-gray-900">{fmtCompact(overview.page_views)}</p>
          </div>
        </section>

        {/* Charts Section for Facebook */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Page Likes Growth</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Page Likes Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Post Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Bar+Chart"
                alt="Post Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Reach & Impressions</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Area+Chart"
                alt="Reach & Impressions Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Audience Demographics</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/150x150/123458/ffffff?text=Pie+Chart"
                alt="Audience Demographics Chart Placeholder"
                className="rounded"
              />
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
