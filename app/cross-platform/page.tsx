"use client"
import { useEffect, useState } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, LabelList, ResponsiveContainer, Pie, PieChart, Tooltip, Cell } from "recharts"
import AppSidebar from "@/components/sidebar"
import type { NextRequest } from "next/server"
import { NextResponse } from "next/server"

export default function CrossPlatformDashboard() {
  const [tempStartDate, setTempStartDate] = useState<string>("2021-01-01")
  const [tempEndDate, setTempEndDate] = useState<string>("2025-12-31")
  const [startDate, setStartDate] = useState<string>("2019-01-01")
  const [endDate, setEndDate] = useState<string>("2025-12-31")
  const [distribution, setDistribution] = useState<any[]>([])
  const [totals, setTotals] = useState<any>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [platformTotalsData, setPlatformTotalsData] = useState<any[]>([])
  const [engagementRateData, setEngagementRateData] = useState<any[]>([])
  const [postsTotals, setPostsTotals] = useState<{ platform: string; total: number }[]>([])
  const [postsTotalsError, setPostsTotalsError] = useState<string | null>(null)
  const [loadingPostsTotals, setLoadingPostsTotals] = useState<boolean>(true)
  const [reachTotals, setReachTotals] = useState<{ platform: string; total: number }[]>([])
  const [loadingReachTotals, setLoadingReachTotals] = useState<boolean>(true)
  const [reachTotalsError, setReachTotalsError] = useState<string | null>(null)
  const [avgEngPerPost, setAvgEngPerPost] = useState<{ platform: string; avg: number }[]>([])
  const [loadingAvgEng, setLoadingAvgEng] = useState<boolean>(true)
  const [avgEngError, setAvgEngError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const res = await fetch(
          `/api/analytics/cross-platform/engagement-distribution?startDate=${startDate}&endDate=${endDate}`,
          { cache: "no-store" }
        )
        const data = await res.json()
        if (!res.ok) {
          throw new Error(data?.detail || data?.error || "Failed to fetch engagement distribution")
        }

        const resTotals = await fetch(
          `/api/analytics/cross-platform/engagement-totals?startDate=${startDate}&endDate=${endDate}`,
          { cache: "no-store" }
        )
        const totalsJson = await resTotals.json()
        if (!resTotals.ok) {
          throw new Error(totalsJson?.detail || totalsJson?.error || "Failed to fetch engagement totals")
        }
        setPlatformTotalsData(totalsJson.data || [])

        const resRate = await fetch(
          `/api/analytics/cross-platform/engagement-rate?startDate=${startDate}&endDate=${endDate}`,
          { cache: "no-store" }
        )
        const rateJson = await resRate.json()
        if (!resRate.ok) {
          throw new Error(rateJson?.detail || rateJson?.error || "Failed to fetch engagement rate")
        }
        setEngagementRateData(rateJson.data || [])

        // --- Total Posts by Platform ---
        try {
          const url = `/api/analytics/cross-platform/post-totals?startDate=${startDate}&endDate=${endDate}&_=${Date.now()}`
          const res = await fetch(url, { cache: "no-store" })
          if (!res.ok) {
            // If you still see <!DOCTYPE html..., middleware or wrong path is the culprit.
            const txt = await res.text()
            throw new Error(`HTTP ${res.status} — ${txt.slice(0, 180)}`)
          }
          const json = await res.json()
          setPostsTotals(Array.isArray(json?.data) ? json.data : [])
          setPostsTotalsError(null)
        } catch (e: any) {
          setPostsTotals([])
          setPostsTotalsError(e?.message || "Failed to load posts totals")
        } finally {
          setLoadingPostsTotals(false)
        }

        // --- Total Reach/Views by Platform ---
        try {
          const url = `/api/analytics/cross-platform/reach-totals?startDate=${startDate}&endDate=${endDate}&_=${Date.now()}`
          const res = await fetch(url, { cache: "no-store" })
          if (!res.ok) {
            const txt = await res.text()
            throw new Error(`HTTP ${res.status} — ${txt.slice(0, 180)}`)
          }
          const json = await res.json()
          setReachTotals(Array.isArray(json?.data) ? json.data : [])
          setReachTotalsError(null)
        } catch (e: any) {
          setReachTotals([])
          setReachTotalsError(e?.message || "Failed to load reach totals")
        } finally {
          setLoadingReachTotals(false)
        }

        // --- Avg Engagement per Post ---
        try {
          const url = `/api/analytics/cross-platform/avg-engagement-per-post?startDate=${startDate}&endDate=${endDate}&_=${Date.now()}`
          const res = await fetch(url, { cache: "no-store" })
          if (!res.ok) {
            const txt = await res.text()
            throw new Error(`HTTP ${res.status} — ${txt.slice(0, 180)}`)
          }
          const json = await res.json()
          setAvgEngPerPost(Array.isArray(json?.data) ? json.data : [])
          setAvgEngError(null)
        } catch (e: any) {
          setAvgEngPerPost([])
          setAvgEngError(e?.message || "Failed to load avg engagement per post")
        } finally {
          setLoadingAvgEng(false)
        }

        // --- NEW: trust only raw totals and compute % on the client (YT/FB/TT only)
        const cleanTotals = {
          YouTube: Number((data.totals && data.totals.YouTube) || 0),
          Facebook: Number((data.totals && data.totals.Facebook) || 0),
          TikTok: Number((data.totals && data.totals.TikTok) || 0),
        }
        const grand =
          cleanTotals.YouTube + cleanTotals.Facebook + cleanTotals.TikTok

        const pct = (n: number) => (grand > 0 ? (n / grand) * 100 : 0)

        const computed = [
          { platform: "YouTube", value: pct(cleanTotals.YouTube) },
          { platform: "Facebook", value: pct(cleanTotals.Facebook) },
          { platform: "TikTok", value: pct(cleanTotals.TikTok) },
        ]

        setTotals(cleanTotals)
        setDistribution(computed)
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load data")
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [startDate, endDate])

  const fmtPct = (n?: number) => {
    if (typeof n !== "number" || !Number.isFinite(n)) return "—"
    return `${n.toFixed(1)}%`
  }
  const fmtInt = (n?: number) =>
    typeof n === "number" && Number.isFinite(n) ? n.toLocaleString() : "—"

  const COLORS = {
    YouTube: "#e23535ff",   // red
    Facebook: "#3273d4ff",  // orange (to match your screenshot)
    TikTok: "#d13b7aff",    // pink
  }

  if (loading) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3] text-white">
        <AppSidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <p className="text-xl text-[#123458]">Loading dashboard data...</p>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3] text-white">
        <AppSidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <div className="bg-red-900 p-6 rounded-lg">
            <p className="text-lg font-semibold text-[#FFFFFF]">Error loading dashboard</p>
            <p className="text-sm mt-2">{error}</p>
          </div>
        </main>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-white">
      <AppSidebar />
      <main className="flex-1 p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">Cross-Platform Social Media Analytics </h1>
        </header>

        {/* SIDE-BY-SIDE: Engagement Distribution (Pie) + Total Engagement by Platform (Bar) */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Left: Pie + Totals inside the same card */}
          <section className="bg-white p-6 rounded-lg shadow-md text-[#123458]">
            <h2 className="text-xl font-semibold mb-4">Engagement Distribution Across Platforms</h2>

            <div className="h-80">
              {distribution.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={distribution}
                      dataKey="value"
                      nameKey="platform"
                      cx="50%"
                      cy="50%"
                      outerRadius={110}
                      paddingAngle={2}
                      label={({ platform, value }) => `${platform}: ${(value as number).toFixed(1)}%`
}
                    >
                      {distribution.map((entry, idx) => (
                        <Cell
                          key={idx}
                          fill={
                            (entry.platform === "YouTube"  && "#e23535ff") ||
                            (entry.platform === "Facebook" && "#3273d4ff") ||
                            (entry.platform === "TikTok"   && "#d13b7aff") ||
                            "#8884d8"
                          }
                        />
                      ))}
                    </Pie>
                    <Tooltip formatter={(v) => `${(v as number).toFixed(1)}%`} />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>

            {/* Totals inside same card */}
            <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-gray-50 rounded px-4 py-3">
                <p className="text-xs uppercase text-gray-500">YouTube Engagement</p>
                <p className="text-2xl font-bold text-gray-900">{fmtInt(totals?.YouTube)}</p>
              </div>
              <div className="bg-gray-50 rounded px-4 py-3">
                <p className="text-xs uppercase text-gray-500">Facebook Engagement</p>
                <p className="text-2xl font-bold text-gray-900">{fmtInt(totals?.Facebook)}</p>
              </div>
              <div className="bg-gray-50 rounded px-4 py-3">
                <p className="text-xs uppercase text-gray-500">TikTok Engagement</p>
                <p className="text-2xl font-bold text-gray-900">{fmtInt(totals?.TikTok)}</p>
              </div>
            </div>
          </section>

          {/* Right: Bar */}
          <section className="bg-white p-6 rounded-lg shadow-md text-[#123458]">
            <h2 className="text-xl font-semibold mb-4">Total Engagement by Platform</h2>
            <div className="h-80">
              {platformTotalsData?.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={platformTotalsData}
                    margin={{ top: 56, right: 24, left: 32, bottom: 40 }}  // ⬅️ more space
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="platform" angle={-30} textAnchor="end" interval={0} />
                    <YAxis tickFormatter={(v) => Number(v).toLocaleString()} />
                    <Tooltip formatter={(v) => Number(v as number).toLocaleString()} />
                    <Bar dataKey="total" name="Total Engagement">
                      <LabelList
                        dataKey="total"
                        position="top"
                        content={({ x, y, width, value }) => {
                          const v = Array.isArray(value) ? value[1] : value
                          const cx = Number(x) + Number(width) / 2
                          const cy = Number(y) - 8  // ⬅️ nudge label up a bit
                          return (
                            <text x={cx} y={cy} textAnchor="middle" fontSize={12} fill="#111">
                              {Number(v).toLocaleString()}
                            </text>
                          )
                        }}
                      />
                      {platformTotalsData.map((entry: any, idx: number) => (
                        <Cell
                          key={idx}
                          fill={
                            (entry.platform === "YouTube"  && "#e23535ff") ||
                            (entry.platform === "Facebook" && "#3273d4ff") ||
                            (entry.platform === "TikTok"   && "#d13b7aff") ||
                            "#8884d8"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </section>
          
          {/* Engagement Rate (%) – horizontal bars */}
          <section className="bg-white p-6 rounded-lg shadow-md text-[#123458]">
            <h2 className="text-xl font-semibold mb-4">Engagement Rate (%)</h2>
            <div className="h-72">
              {engagementRateData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={engagementRateData}
                    layout="vertical"
                    margin={{ top: 16, right: 24, left: 48, bottom: 16 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <YAxis
                      type="category"
                      dataKey="platform"
                      width={90}
                    />
                    <XAxis
                      type="number"
                      domain={[0, "auto"]}
                      tickFormatter={(v) => `${Number(v).toFixed(2)}%`}
                    />
                    <Tooltip formatter={(v) => `${Number(v as number).toFixed(2)}%`} />
                    <Bar dataKey="rate" name="Engagement Rate (%)">
                      <LabelList
                        dataKey="rate"
                        position="right"
                        content={({ x, y, value }) => {
                          const v = Array.isArray(value) ? value[1] : value
                          return (
                            <text x={Number(x) + 8} y={Number(y) + 4} fontSize={12} fill="#111">
                              {`${Number(v).toFixed(2)}%`}
                            </text>
                          )
                        }}
                      />
                      {engagementRateData.map((d: any, i: number) => (
                        <Cell
                          key={i}
                          fill={
                            (d.platform === "YouTube"  && "#e23535ff") ||
                            (d.platform === "Facebook" && "#3273d4ff") ||
                            (d.platform === "TikTok"   && "#d13b7aff") ||
                            "#8884d8"
                          }
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="text-gray-500">No data available</div>
              )}
            </div>
          </section>
          
          <section aria-labelledby="posts-by-platform-heading" className="bg-white p-6 rounded-lg shadow-md text-[#123458]">
          <h2 id="posts-by-platform-heading" className="text-xl font-semibold mb-4">
            Total Posts by Platform
          </h2>

          {loadingPostsTotals ? (
            <p className="text-sm text-[#123458]/70 text-center">Loading…</p>
          ) : postsTotalsError ? (
            <p className="text-sm text-red-700 text-center">{postsTotalsError}</p>
          ) : (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={postsTotals} margin={{ top: 16, right: 24, left: 24, bottom: 36 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="platform" angle={-30} textAnchor="end" interval={0} />
                  <YAxis tickFormatter={(v) => Number(v).toLocaleString()} />
                  <Tooltip formatter={(v) => Number(v as number).toLocaleString()} />
                  <Bar dataKey="total" name="Posts">
                    {/* Facebook / TikTok / YouTube colors to match your theme */}
                    {postsTotals.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={
                          (entry.platform === "Facebook" && "#3273d4ff") ||
                          (entry.platform === "TikTok"   && "#d13b7aff") ||
                          (entry.platform === "YouTube"  && "#e23535ff") ||
                          "#8884d8"
                        }
                      />
                    ))}
                    <LabelList
                      dataKey="total"
                      position="top"
                      content={({ x, y, width, value }) => {
                        const v = Array.isArray(value) ? value[1] : value
                        const cx = Number(x) + Number(width) / 2
                        const cy = Number(y) - 8
                        return (
                          <text x={cx} y={cy} textAnchor="middle" fontSize={12} fill="#111">
                            {Number(v).toLocaleString()}
                          </text>
                        )
                      }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        <section aria-labelledby="reach-by-platform-heading" className="bg-white p-6 rounded-lg shadow-md text-[#123458]">
        <h2 id="reach-by-platform-heading" className="text-xl font-semibold mb-4">
          Total Reach/Views by Platform
        </h2>

        {loadingReachTotals ? (
          <p className="text-sm text-[#123458]/70 text-center">Loading…</p>
        ) : reachTotalsError ? (
          <p className="text-sm text-red-700 text-center">{reachTotalsError}</p>
        ) : (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={reachTotals} margin={{ top: 16, right: 24, left: 56, bottom: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="platform" angle={-30} textAnchor="end" interval={0} />
                <YAxis tickFormatter={(v) => Number(v).toLocaleString()} label={{angle: -90, position: "insideLeft" }} />
                <Tooltip
                  formatter={(v) => Number(v as number).toLocaleString()}
                  labelFormatter={(l) => `${l}`}
                />
                <Bar dataKey="total">
                  {reachTotals.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={
                        (entry.platform === "Facebook" && "#3273d4ff") ||
                        (entry.platform === "TikTok"   && "#d13b7aff") ||
                        (entry.platform === "YouTube"  && "#e23535ff") ||
                        "#8884d8"
                      }
                    />
                  ))}
                  <LabelList
                    dataKey="total"
                    position="top"
                    content={({ x, y, width, value }) => {
                      const n = Number(Array.isArray(value) ? value[1] : value)
                      let label = n.toLocaleString()
                      if (n >= 1_000_000) label = `${(n/1_000_000).toFixed(1)}M`
                      else if (n >= 1_000) label = `${Math.round(n/1_000)}K`
                      const cx = Number(x) + Number(width) / 2
                      const cy = Number(y) - 8
                      return (
                        <text x={cx} y={cy} textAnchor="middle" fontSize={12} fill="#111">
                          {label}
                        </text>
                      )
                    }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </section>
        <section aria-labelledby="avg-engagement-heading" className="bg-white p-6 rounded-lg shadow-md text-[#123458]">
          <h2 id="avg-engagement-heading" className="text-xl font-semibold mb-4">
            Avg Engagement per Post
          </h2>

          {loadingAvgEng ? (
            <p className="text-sm text-[#123458]/70 text-center">Loading…</p>
          ) : avgEngError ? (
            <p className="text-sm text-red-700 text-center">{avgEngError}</p>
          ) : (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={avgEngPerPost} margin={{ top: 16, right: 24, left: 56, bottom: 80 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="platform" angle={-30} textAnchor="end" interval={0} />
                  <YAxis tickFormatter={(v) => Number(v).toLocaleString()} label={{ angle: -90, position: "insideLeft" }} />
                  <Tooltip formatter={(v) => Number(v as number).toLocaleString()} />
                  <Bar dataKey="avg" name="Avg Engagement per Post">
                    {avgEngPerPost.map((entry, i) => (
                      <Cell
                        key={i}
                        fill={
                          (entry.platform === "Facebook" && "#3273d4ff") ||
                          (entry.platform === "TikTok"   && "#d13b7aff") ||
                          (entry.platform === "YouTube"  && "#e23535ff") ||
                          "#8884d8"
                        }
                      />
                    ))}
                    <LabelList
                      dataKey="avg"
                      position="top"
                      content={({ x, y, width, value }) => {
                        const n = Number(Array.isArray(value) ? value[1] : value)
                        const cx = Number(x) + Number(width) / 2
                        const cy = Number(y) - 8
                        return (
                          <text x={cx} y={cy} textAnchor="middle" fontSize={12} fill="#111">
                            {n.toLocaleString()}
                          </text>
                        )
                      }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>
        </div>
      </main>
    </div>
  )
}
