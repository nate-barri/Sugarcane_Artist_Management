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
  ScatterChart,
  Scatter,
  ReferenceLine,
  Cell,
} from "recharts"

function MetricsBox({ metrics, variant = "blue" }: { metrics: any; variant?: string }) {
  if (!metrics) return null

  const bgColor = {
    blue: "bg-blue-50 border-blue-200",
    yellow: "bg-yellow-50 border-yellow-200",
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

export default function PredictiveAnalyticsDashboard() {
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
        const errorMessage = err instanceof Error ? err.message : "Failed to load predictive models"
        setError(errorMessage)
        console.error("[v0] Predictive analytics error:", err)
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
            <div className="text-lg font-semibold text-gray-700 mb-2">Loading predictive models...</div>
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
        <h1 className="text-3xl font-bold text-[#123458] mb-2">Predictive Analytics</h1>
        <p className="text-gray-600 mb-8">9 Predictive Models: 3 Facebook + 3 YouTube + 3 TikTok</p>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* ============ META MODELS (3) ============ */}

          {/* Meta Model 1: Backtest */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <div className="absolute top-3 right-3 bg-red-100 text-red-700 text-xs px-2 py-1 rounded">
              {data?.meta.backtest.label || "Backtesting"}
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.meta.backtest.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.meta.backtest.description}</p>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={data?.meta.backtest.data} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
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
                  name="Actual Reach"
                  connectNulls={false}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Predicted Reach"
                  connectNulls={true}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.meta.backtest.metrics} variant="blue" />
          </div>

          {/* Meta Model 2: 6-Month Reach Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <div className="absolute top-3 right-3 bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">
              {data?.meta.reach6m.videoType || "New Videos"}
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.meta.reach6m.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.meta.reach6m.description}</p>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={data?.meta.reach6m.data} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} width={50} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area
                  type="monotone"
                  dataKey="lower"
                  fill="#93c5fd"
                  fillOpacity={0.1}
                  stroke="none"
                  name="Confidence Range"
                />
                <Area type="monotone" dataKey="upper" fill="#93c5fd" fillOpacity={0.2} stroke="none" />
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
                  stroke="#22c55e"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  name="Forecasted"
                  connectNulls={true}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.meta.reach6m.metrics} variant="blue" />
          </div>

          {/* Meta Model 3: Existing Posts Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <div className="absolute top-3 right-3 bg-green-100 text-green-700 text-xs px-2 py-1 rounded">
              {data?.meta.existingPostsForecast.videoType || "Existing Videos"}
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.meta.existingPostsForecast.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.meta.existingPostsForecast.description}</p>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart
                data={data?.meta.existingPostsForecast.data}
                margin={{ top: 5, right: 10, left: 0, bottom: 40 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} width={50} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area type="monotone" dataKey="lower" fill="#86efac" fillOpacity={0.1} stroke="none" />
                <Area
                  type="monotone"
                  dataKey="upper"
                  fill="#86efac"
                  fillOpacity={0.3}
                  stroke="none"
                  name="Confidence Range"
                />
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Existing (Historical)"
                  connectNulls={false}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Existing (Forecast)"
                  connectNulls={true}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.meta.existingPostsForecast.metrics} variant="green" />
          </div>

          {/* ============ YOUTUBE MODELS (3) - EXPANDED ============ */}

          {/* YouTube Model 1: Historical Cumulative - REPLACED with ADD_back style */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <h2 className="text-sm font-bold mb-1">{data?.youtube.cumulativeModel.title}</h2>
            <p className="text-xs text-gray-600 mb-4">Full historical timeline with model validation</p>
            <ResponsiveContainer width="100%" height={250}>
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
                />
              </LineChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.youtube.cumulativeModel.part1?.metrics} variant="blue" />
          </div>

          {/* YouTube Model 2: NEW - 6-Month Cumulative Forecast (9th graph) */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <h2 className="text-sm font-bold mb-1">
              {data?.youtube.cumulativeModel.part2?.label || "6-Month Cumulative View Forecast"}
            </h2>
            <p className="text-xs text-gray-600 mb-4">Last 6 months historical + Next 6 months forecast</p>
            <ResponsiveContainer width="100%" height={250}>
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
                />
                <Area type="monotone" dataKey="upper" fill="#86efac" fillOpacity={0.2} stroke="none" />
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Historical (Last 6M)"
                  connectNulls={false}
                  dot={false}
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
                />
              </AreaChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.youtube.cumulativeModel.metrics} variant="green" />
          </div>

          {/* YouTube Model 3: Catalog Views */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <h2 className="text-sm font-bold mb-1">{data?.youtube.catalogViews.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.catalogViews.description}</p>
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={data?.youtube.catalogViews.data} margin={{ top: 5, right: 10, left: 0, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={60} tick={{ fontSize: 9 }} />
                <YAxis tick={{ fontSize: 9 }} width={50} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Area type="linear" dataKey="lower" fill="#f97316" fillOpacity={0.1} stroke="none" />
                <Area
                  type="linear"
                  dataKey="upper"
                  fill="#f97316"
                  fillOpacity={0.2}
                  stroke="none"
                  name="Confidence Range"
                />
                <Line
                  type="linear"
                  dataKey="historical"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Historical (Backcast)"
                  connectNulls={false}
                  dot={false}
                />
                <Line
                  type="linear"
                  dataKey="forecast"
                  stroke="#f97316"
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Forecast (Baseline 6mo)"
                  connectNulls={true}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.youtube.catalogViews.metrics} variant="orange" />
          </div>

          {/* ============ TIKTOK MODELS (3) ============ */}

          {/* TikTok Model 1: Channel Views */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.channelViews.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.channelViews.description}</p>
            <ResponsiveContainer width="100%" height={250}>
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
            <MetricsBox metrics={data?.tiktok.channelViews.metrics} variant="yellow" />
          </div>

          {/* TikTok Model 2: Prediction Accuracy (Scatter) - ENHANCED */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.predictionAccuracy.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.predictionAccuracy.description}</p>
            <ResponsiveContainer width="100%" height={250}>
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
            <MetricsBox metrics={data?.tiktok.predictionAccuracy.metrics} variant="yellow" />
          </div>

          {/* TikTok Model 3: Cumulative Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-1 relative">
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.cumulativeForecast.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.cumulativeForecast.description}</p>
            <ResponsiveContainer width="100%" height={250}>
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
            <MetricsBox metrics={data?.tiktok.cumulativeForecast.metrics} variant="yellow" />
          </div>
        </section>
      </main>
    </div>
  )
}
