"use client"

import { useState, useEffect } from "react"
import Sidebar from "@/components/sidebar"
import type { ModelData } from "@/types/modelData"
import {
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ReferenceLine,
  Cell,
} from "recharts"
import { MetricsCard } from "@/app/predictive-analytics/metrics-card"

function MetricsBox({ metrics, variant = "yellow" }: { metrics: any; variant?: string }) {
  if (!metrics) return null

  const bgColor = {
    yellow: "bg-yellow-50 border-yellow-200",
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

export default function TikTokPredictive() {
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
        const errorMessage = err instanceof Error ? err.message : "Failed to load TikTok models"
        setError(errorMessage)
        console.error("[v0] TikTok analytics error:", err)
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
            <div className="text-lg font-semibold text-gray-700 mb-2">Loading TikTok predictive models...</div>
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
          <h1 className="text-3xl font-bold text-[#123458]">TikTok Predictive Models</h1>
          <p className="text-gray-600 mt-2">
            3 TikTok Models: Channel Views + Prediction Accuracy + Cumulative Forecast
          </p>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* TikTok Model 1: Channel Views */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.channelViews.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.channelViews.description}</p>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={data?.tiktok.channelViews.data} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} width={50} />
                <Tooltip formatter={(v) => (typeof v === "number" ? (v / 1e6).toFixed(1) + "M" : "N/A")} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area type="monotone" dataKey="lower" fill="#fbbf24" fillOpacity={0.1} stroke="none" />
                <Area
                  type="monotone"
                  dataKey="upper"
                  fill="#fbbf24"
                  fillOpacity={0.25}
                  stroke="none"
                  name="Confidence Range"
                />
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Historical"
                  connectNulls={false}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Predicted"
                  connectNulls={true}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* TikTok Model 2: Prediction Accuracy (Scatter) */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.predictionAccuracy.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.predictionAccuracy.description}</p>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 60, left: 40, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  type="number"
                  dataKey="actual"
                  name="Actual Engagement Rate (%)"
                  tick={{ fontSize: 9 }}
                  domain={[0, 25]}
                />
                <YAxis
                  type="number"
                  dataKey="predicted"
                  name="Predicted Engagement Rate (%)"
                  tick={{ fontSize: 9 }}
                  domain={[0, 25]}
                />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ fontSize: 10 }} />
                <Legend wrapperStyle={{ fontSize: 10, paddingTop: 10 }} />

                <ReferenceLine
                  stroke="#dc2626"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  segment={[
                    { x: 0, y: 0 },
                    { x: 25, y: 25 },
                  ]}
                  name="Perfect Prediction"
                  label={{ value: "Perfect Prediction", fontSize: 9, fill: "#dc2626" }}
                />

                <ReferenceLine
                  stroke="#22c55e"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                  segment={[
                    { x: 0, y: 3 },
                    { x: 22, y: 25 },
                  ]}
                  name="±3% Zone"
                />
                <ReferenceLine
                  stroke="#22c55e"
                  strokeDasharray="3 3"
                  strokeWidth={1}
                  segment={[
                    { x: 3, y: 0 },
                    { x: 25, y: 22 },
                  ]}
                />

                <Scatter name="Predictions" data={data?.tiktok.predictionAccuracy.data || []} fill="#8884d8">
                  {data?.tiktok.predictionAccuracy.data.map((entry, index) => {
                    let fillColor = "#22c55e" // default green
                    if (entry.color === "red") fillColor = "#ef4444"
                    else if (entry.color === "orange") fillColor = "#f97316"
                    else if (entry.color === "yellow") fillColor = "#eab308"

                    return <Cell key={`cell-${index}`} fill={fillColor} />
                  })}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* TikTok Model 3: Cumulative Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.cumulativeForecast.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.cumulativeForecast.description}</p>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart
                data={data?.tiktok.cumulativeForecast.data}
                margin={{ top: 5, right: 10, left: 0, bottom: 40 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} width={50} />
                <Tooltip formatter={(v) => (typeof v === "number" ? (v / 1e6).toFixed(1) + "M" : "N/A")} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area type="monotone" dataKey="lower" fill="#fbbf24" fillOpacity={0.1} stroke="none" />
                <Area
                  type="monotone"
                  dataKey="upper"
                  fill="#fbbf24"
                  fillOpacity={0.25}
                  stroke="none"
                  name="Confidence Range"
                />
                <Line
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Historical (Last 6M)"
                  connectNulls={false}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Projected (6-Month)"
                  connectNulls={true}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <div className="space-y-4">
          <div className="h-1 bg-black/20 rounded-full"></div>

          <div className="flex flex-wrap gap-4 justify-center">
            <MetricsCard title="Channel Views" metrics={data?.tiktok.channelViews.metrics} variant="yellow" />
            <MetricsCard
              title="Prediction Accuracy"
              metrics={data?.tiktok.predictionAccuracy.metrics}
              variant="orange"
            />
            <MetricsCard
              title="Cumulative Forecast"
              metrics={data?.tiktok.cumulativeForecast.metrics}
              variant="yellow"
            />
          </div>
        </div>
      </main>
    </div>
  )
}
