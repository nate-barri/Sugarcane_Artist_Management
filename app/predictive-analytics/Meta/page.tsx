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

function MetricsBox({ metrics, variant = "blue" }: { metrics: any; variant?: string }) {
  if (!metrics) return null

  const bgColor = {
    blue: "bg-blue-50 border-blue-200",
    red: "bg-red-50 border-red-200",
    green: "bg-green-50 border-green-200",
  }[variant]

  return (
    <div className={`absolute bottom-2 right-2 ${bgColor} border rounded p-2 text-xs space-y-1`}>
      {metrics.mape !== undefined && (
        <div className="font-semibold text-gray-700">MAPE: {metrics.mape.toFixed(1)}%</div>
      )}
      {metrics.r2 !== undefined && <div className="text-gray-700">R²: {metrics.r2.toFixed(3)}</div>}
      {metrics.mase !== undefined && <div className="text-gray-700">MASE: {metrics.mase.toFixed(2)}</div>}
      {metrics.mae !== undefined && <div className="text-gray-700">MAE: {metrics.mae.toFixed(2)}</div>}
    </div>
  )
}

export default function MetaPredictive() {
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
        const errorMessage = err instanceof Error ? err.message : "Failed to load Meta models"
        setError(errorMessage)
        console.error("[v0] Meta analytics error:", err)
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
            <div className="text-lg font-semibold text-gray-700 mb-2">Loading Meta predictive models...</div>
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
          <h1 className="text-3xl font-bold text-[#123458]">Meta Predictive Models</h1>
          <p className="text-gray-600 mt-2">3 Meta Models: Backtest + 6-Month Reach + Existing Posts Forecast</p>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Meta Model 1: Backtest */}
          <div className="bg-white p-6 rounded-lg shadow-md relative">
            <div className="absolute top-3 right-3 bg-red-100 text-red-700 text-xs px-2 py-1 rounded">
              {data?.meta.backtest.label || "Backtesting"}
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.meta.backtest.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.meta.backtest.description}</p>
            <ResponsiveContainer width="100%" height={300}>
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
            <MetricsBox metrics={data?.meta.backtest.metrics} variant="red" />
          </div>

          {/* Meta Model 2: 6-Month Reach Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md relative">
            <div className="absolute top-3 right-3 bg-blue-100 text-blue-700 text-xs px-2 py-1 rounded">
              {data?.meta.reach6m.videoType || "New Videos"}
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.meta.reach6m.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.meta.reach6m.description}</p>
            <ResponsiveContainer width="100%" height={300}>
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
          <div className="bg-white p-6 rounded-lg shadow-md relative">
            <div className="absolute top-3 right-3 bg-green-100 text-green-700 text-xs px-2 py-1 rounded">
              {data?.meta.existingPostsForecast.videoType || "Existing Videos"}
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.meta.existingPostsForecast.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.meta.existingPostsForecast.description}</p>
            <ResponsiveContainer width="100%" height={300}>
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
                  stroke="#22c55e"
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
        </section>
      </main>
    </div>
  )
}
