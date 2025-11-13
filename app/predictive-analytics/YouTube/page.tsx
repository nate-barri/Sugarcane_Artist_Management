"use client"

import { useState, useEffect } from "react"
import Sidebar from "@/components/sidebar"
import type { ModelData } from "@/types/modelData"
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts"
import { MetricsCard } from "@/app/predictive-analytics/metrics-card"

function MetricsBox({ metrics, variant = "blue" }: { metrics: any; variant?: string }) {
  if (!metrics) return null

  const bgColor = {
    blue: "bg-blue-50 border-blue-200",
    green: "bg-green-50 border-green-200",
    orange: "bg-orange-50 border-orange-200",
  }[variant]

  return (
    <div className={`absolute bottom-2 right-2 ${bgColor} border rounded p-2 text-xs space-y-1`}>
      {metrics.mape !== undefined && (
        <div className="font-semibold text-gray-700">MAPE: {metrics.mape.toFixed(1)}%</div>
      )}
      {metrics.r2 !== undefined && <div className="text-gray-700">R²: {metrics.r2.toFixed(3)}</div>}
      {metrics.mase !== undefined && <div className="text-gray-700">MASE: {metrics.mase.toFixed(2)}</div>}
      {metrics.mae !== undefined && <div className="text-gray-700">MAE: {metrics.mae.toFixed(2)}</div>}
      {metrics.n !== undefined && <div className="text-gray-700">n = {metrics.n}</div>}
    </div>
  )
}

function ChartWrapper({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full flex flex-col" style={{ height: "300px", minHeight: "300px" }}>
      {children}
    </div>
  )
}

export default function YouTubePredictive() {
  const [data, setData] = useState<ModelData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setError(null)
        const response = await fetch("/api/analytics/predictive/models-data")
        if (!response.ok) {
          throw new Error(`Failed to fetch data: ${response.status}`)
        }
        const json = await response.json()
        setData(json)
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to load YouTube models"
        setError(errorMessage)
        console.error("[v0] YouTube analytics error:", err)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3]">
        <Sidebar />
        <main className="flex-1 p-8 flex items-center justify-center">
          <div className="text-center">
            <div className="text-lg font-semibold text-gray-700 mb-2">Loading YouTube predictive models...</div>
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          </div>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3]">
        <Sidebar />
        <main className="flex-1 p-8">
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-800">
            <h2 className="font-bold mb-2">Error Loading Data</h2>
            <p>{error}</p>
          </div>
        </main>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="flex min-h-screen bg-[#D3D3D3]">
        <Sidebar />
        <main className="flex-1 p-8">
          <div className="text-center text-gray-600">No data available</div>
        </main>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen bg-[#D3D3D3]">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">YouTube Predictive Models</h1>
          <p className="text-gray-600 mt-2">6 YouTube Models: Engagement + Catalog + Growth Breakdown</p>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* YouTube Model 1: Historical Cumulative */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-sm font-bold mb-1">{data?.youtube.cumulativeModel.title}</h2>
            <p className="text-xs text-gray-600 mb-4">Full historical timeline with model validation</p>
            <ChartWrapper>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data?.youtube.cumulativeModel.part1?.data}
                  margin={{ top: 5, right: 10, left: 0, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 9 }} width={50} />
                  <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Line
                    type="monotone"
                    dataKey="actual"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    name="Actual Cumulative"
                    connectNulls={false}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="model"
                    stroke="#f97316"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Model Estimate"
                    connectNulls={true}
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </ChartWrapper>
          </div>

          {/* YouTube Model 2: 6-Month Cumulative Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-sm font-bold mb-1">
              {data?.youtube.cumulativeModel.part2?.label || "6-Month Cumulative View Forecast"}
            </h2>
            <p className="text-xs text-gray-600 mb-4">Last 6 months historical + Next 6 months forecast</p>
            <ChartWrapper>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={data?.youtube.cumulativeModel.part2?.data}
                  margin={{ top: 5, right: 10, left: 0, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 9 }} width={50} />
                  <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    fill="#86efac"
                    fillOpacity={0.1}
                    stroke="none"
                    name="Confidence Range"
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="upper"
                    fill="#86efac"
                    fillOpacity={0.2}
                    stroke="none"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="historical"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    name="Historical (Last 6M)"
                    connectNulls={false}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#22c55e"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Baseline (6mo)"
                    connectNulls={true}
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartWrapper>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-2">
            <h2 className="text-sm font-bold mb-1">{data?.youtube.catalogViews.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.catalogViews.description}</p>
            <ChartWrapper>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data?.youtube.catalogViews.data} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                  <YAxis tick={{ fontSize: 9 }} width={50} />
                  <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                  <Legend wrapperStyle={{ fontSize: 10 }} />
                  <Area
                    type="monotone"
                    dataKey="lower"
                    fill="#fbbf24"
                    fillOpacity={0.1}
                    stroke="none"
                    name="Confidence Range"
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="upper"
                    fill="#fbbf24"
                    fillOpacity={0.2}
                    stroke="none"
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="historical"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    name="Historical (Backcast)"
                    connectNulls={false}
                    dot={false}
                    isAnimationActive={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="forecast"
                    stroke="#f97316"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    name="Forecast (Baseline 6mo)"
                    connectNulls={true}
                    dot={false}
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </ChartWrapper>
          </div>
        </section>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* YouTube Model 5: Existing Catalog */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-violet-100 text-violet-700 text-xs px-2 py-1 rounded">
              Catalog
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.youtube.existingCatalog.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.existingCatalog.description}</p>
            <div className="space-y-3">
              <div className="p-3 bg-gray-50 rounded">
                <div className="text-xs text-gray-600">Current</div>
                <div className="font-semibold text-gray-900">
                  {data?.youtube.existingCatalog.metrics.current?.toLocaleString()}
                </div>
              </div>
              <div className="p-3 bg-green-50 rounded">
                <div className="text-xs text-gray-600">Projected</div>
                <div className="font-semibold text-green-900">
                  {data?.youtube.existingCatalog.metrics.projected?.toLocaleString()}
                </div>
              </div>
              <div className="p-3 bg-blue-50 rounded">
                <div className="text-xs text-gray-600">Growth</div>
                <div className="font-semibold text-blue-900">
                  +{data?.youtube.existingCatalog.metrics.growth?.toLocaleString()} (
                  {data?.youtube.existingCatalog.metrics.growth_percent}%)
                </div>
                <div className="text-xs text-blue-700 mt-1">
                  +{data?.youtube.existingCatalog.metrics.monthly}% monthly
                </div>
              </div>
            </div>
          </div>

          {/* YouTube Model 6: Growth Breakdown */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-amber-100 text-amber-700 text-xs px-2 py-1 rounded">
              Breakdown
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.youtube.growthBreakdown.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.growthBreakdown.description}</p>
            <div className="space-y-3">
              {data?.youtube.growthBreakdown.data?.map((item, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded ${idx === 0 ? "bg-emerald-50" : idx === 1 ? "bg-amber-50" : "bg-slate-50"}`}
                >
                  <div className="flex justify-between items-center">
                    <span className="text-sm text-gray-600">{item.category} Content</span>
                    <span
                      className={`font-semibold ${idx === 0 ? "text-emerald-900" : idx === 1 ? "text-amber-900" : "text-slate-900"}`}
                    >
                      +{item.percent}%
                    </span>
                  </div>
                  <div className="text-xs text-gray-600 mt-1">+{item.growth?.toLocaleString()} views</div>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-cyan-100 text-cyan-700 text-xs px-2 py-1 rounded">
              New Content
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.youtube.growthFromNewContent.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.growthFromNewContent.description}</p>
            <div className="p-4 bg-cyan-50 rounded-lg">
              <div className="text-xs text-gray-600 mb-2">Growth from New Content</div>
              <div className="font-bold text-2xl text-cyan-900">
                +{data?.youtube.growthFromNewContent.metrics.growth?.toLocaleString()}
              </div>
              <div className="text-xs text-cyan-700 mt-2">views from new content</div>
            </div>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded">
              Projection
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.youtube.sixMonthProjection.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.sixMonthProjection.description}</p>
            <div className="space-y-2">
              <div className="p-3 bg-purple-50 rounded">
                <div className="text-xs text-gray-600">Baseline (6mo)</div>
                <div className="font-semibold text-purple-900">
                  {data?.youtube.sixMonthProjection.metrics.projectedTotal?.toLocaleString()}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div className="p-2 bg-red-50 rounded">
                  <div className="text-xs text-gray-600">Conservative</div>
                  <div className="font-semibold text-red-700 text-sm">
                    {data?.youtube.sixMonthProjection.metrics.conservative?.toLocaleString()}
                  </div>
                </div>
                <div className="p-2 bg-green-50 rounded">
                  <div className="text-xs text-gray-600">Optimistic</div>
                  <div className="font-semibold text-green-700 text-sm">
                    {data?.youtube.sixMonthProjection.metrics.optimistic?.toLocaleString()}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <div className="space-y-4">
          <div className="h-1 bg-black/20 rounded-full"></div>

          <div className="flex flex-wrap gap-4 justify-center">
            <MetricsCard
              title="Historical Cumulative"
              metrics={data?.youtube.cumulativeModel.part1?.metrics}
              variant="blue"
            />
            <MetricsCard
              title="6-Month Forecast"
              metrics={data?.youtube.cumulativeModel.part2?.metrics}
              variant="green"
            />
            <MetricsCard title="Catalog Views" metrics={data?.youtube.catalogViews.metrics} variant="orange" />
          </div>
        </div>
      </main>
    </div>
  )
}
