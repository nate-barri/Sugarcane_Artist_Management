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
          <p className="text-gray-600 mt-2">
            3 YouTube Models: Historical Cumulative + 6-Month Forecast + Catalog Views
          </p>
        </header>

        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* YouTube Model 1: Historical Cumulative */}
          <div className="bg-white p-6 rounded-lg shadow-md relative">
            <h2 className="text-sm font-bold mb-1">{data?.youtube.cumulativeModel.title}</h2>
            <p className="text-xs text-gray-600 mb-4">Full historical timeline with model validation</p>
            <ResponsiveContainer width="100%" height={300}>
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

          {/* YouTube Model 2: 6-Month Cumulative Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md relative">
            <h2 className="text-sm font-bold mb-1">
              {data?.youtube.cumulativeModel.part2?.label || "6-Month Cumulative View Forecast"}
            </h2>
            <p className="text-xs text-gray-600 mb-4">Last 6 months historical + Next 6 months forecast</p>
            <ResponsiveContainer width="100%" height={300}>
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
            <MetricsBox metrics={data?.youtube.cumulativeModel.part2?.metrics} variant="green" />
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md relative lg:col-span-2">
            <h2 className="text-sm font-bold mb-1">{data?.youtube.catalogViews.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.catalogViews.description}</p>
            <ResponsiveContainer width="100%" height={300}>
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
                />
                <Area type="monotone" dataKey="upper" fill="#fbbf24" fillOpacity={0.2} stroke="none" />
                <Line
                  type="monotone"
                  dataKey="historical"
                  stroke="#2563eb"
                  strokeWidth={2.5}
                  name="Historical (Backcast)"
                  connectNulls={false}
                  dot={false}
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
                />
              </AreaChart>
            </ResponsiveContainer>
            <MetricsBox metrics={data?.youtube.catalogViews.metrics} variant="orange" />
          </div>
        </section>
      </main>
    </div>
  )
}
