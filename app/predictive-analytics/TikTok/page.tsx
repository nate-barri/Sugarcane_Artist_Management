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
  LineChart,
  ScatterChart,
  Scatter,
} from "recharts"

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
          <p className="text-gray-600 mt-2">6 TikTok Models: New & Existing Videos + Channel Views + Forecasts</p>
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

          {/* TikTok Model 2: Engagement Rate (moved from YouTube) */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-sky-100 text-sky-700 text-xs px-2 py-1 rounded">Engagement</div>
            <h2 className="text-sm font-bold mb-1">{data?.youtube.engagementRate.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.youtube.engagementRate.description}</p>
            <div className="space-y-4">
              <div className="p-3 bg-blue-50 rounded">
                <div className="text-xs text-gray-600">Current Rate</div>
                <div className="text-2xl font-bold text-blue-600">{data?.youtube.engagementRate.metrics.current}%</div>
              </div>
              <div className="p-3 bg-cyan-50 rounded">
                <div className="text-xs text-gray-600">Trend Rate</div>
                <div className="text-2xl font-bold text-cyan-600">{data?.youtube.engagementRate.metrics.trend}%</div>
              </div>
            
            </div>
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

        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* TikTok Model 4: New Videos Backtest */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-pink-100 text-pink-700 text-xs px-2 py-1 rounded">New Videos</div>
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.newVideosBacktest.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.newVideosBacktest.description}</p>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">Pseudo-R²</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.newVideosBacktest.metrics.r2_pseudo.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">MASE</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.newVideosBacktest.metrics.mase.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">MAE</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.newVideosBacktest.metrics.mae.toLocaleString()}
                </span>
              </div>
              <div className="p-3 bg-pink-50 rounded text-center">
                <div className="text-xs text-gray-600">New Video Performance Model</div>
              </div>
            </div>
          </div>

          {/* TikTok Model 5: New Videos Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-rose-100 text-rose-700 text-xs px-2 py-1 rounded">Forecast</div>
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.newVideosForecast.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.newVideosForecast.description}</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={data?.tiktok.newVideosForecast.data} margin={{ top: 5, right: 10, left: 0, bottom: 30 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis dataKey="week" angle={-45} textAnchor="end" height={40} tick={{ fontSize: 8 }} />
                <YAxis tick={{ fontSize: 9 }} width={40} />
                <Tooltip formatter={(v) => (typeof v === "number" ? v.toLocaleString() : "N/A")} />
                <Line
                  type="monotone"
                  dataKey="views"
                  stroke="#ec4899"
                  strokeWidth={2}
                  name="Weekly Views"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-3 p-2 bg-pink-50 rounded text-xs">
              <div className="font-semibold text-gray-900">
                Total: {data?.tiktok.newVideosForecast.metrics.total_views?.toLocaleString()} views
              </div>
              <div className="text-gray-600">
                Growth: {data?.tiktok.newVideosForecast.metrics.growth_start?.toLocaleString()} →{" "}
                {data?.tiktok.newVideosForecast.metrics.growth_end?.toLocaleString()}
              </div>
            </div>
          </div>

          {/* TikTok Model 6: Existing Videos Forecast */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-teal-100 text-teal-700 text-xs px-2 py-1 rounded">
              Existing Videos
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.existingVideosForecast.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.existingVideosForecast.description}</p>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">R²</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.existingVideosBacktest.metrics.r2.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">MASE</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.existingVideosBacktest.metrics.mase.toFixed(3)}
                </span>
              </div>
              <div className="flex justify-between items-center p-3 bg-gray-50 rounded">
                <span className="text-sm text-gray-600">MAPE</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.existingVideosBacktest.metrics.mape}%</span>
              </div>
              <div className="p-3 bg-teal-50 rounded">
                <div className="font-semibold text-teal-900 text-sm">Total Views Forecast</div>
                <div className="text-teal-700">
                  {data?.tiktok.existingVideosForecast.metrics.total_views?.toLocaleString()}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <div className="absolute top-3 right-3 bg-purple-100 text-purple-700 text-xs px-2 py-1 rounded">
              Model Performance
            </div>
            <h2 className="text-sm font-bold mb-1">{data?.tiktok.engagementRateModel.title}</h2>
            <p className="text-xs text-gray-600 mb-4">{data?.tiktok.engagementRateModel.description}</p>

            <ResponsiveContainer width="100%" height={350}>
              <ScatterChart margin={{ top: 20, right: 20, bottom: 60, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                <XAxis
                  type="number"
                  dataKey="x"
                  name="Actual Engagement Rate (%)"
                  label={{ value: "Actual (%)", position: "insideBottomRight", offset: -10 }}
                  domain={[0, 25]}
                  tick={{ fontSize: 9 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  name="Predicted Engagement Rate (%)"
                  label={{ value: "Predicted (%)", angle: -90, position: "insideLeft" }}
                  domain={[0, 25]}
                  tick={{ fontSize: 9 }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  contentStyle={{ fontSize: 12 }}
                  formatter={(value) => (typeof value === "number" ? value.toFixed(2) : value)}
                />

                {/* Perfect Prediction Line */}
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  strokeWidth={2}
                  dot={false}
                  name="Perfect Prediction"
                  isAnimationActive={false}
                  data={[
                    { x: 0, y: 0 },
                    { x: 25, y: 25 },
                  ]}
                />

                {/* ±3% Confidence Zone - Upper Bound */}
                <Line
                  type="monotone"
                  data={Array.from({ length: 26 }, (_, i) => ({
                    x: i,
                    y: i + 3,
                  }))}
                  dataKey="y"
                  stroke="#86efac"
                  strokeWidth={12}
                  dot={false}
                  isAnimationActive={false}
                  name="±3% Zone"
                  strokeOpacity={0.3}
                />

                {/* ±3% Confidence Zone - Lower Bound */}
                <Line
                  type="monotone"
                  data={Array.from({ length: 26 }, (_, i) => ({
                    x: i,
                    y: Math.max(0, i - 3),
                  }))}
                  dataKey="y"
                  stroke="#86efac"
                  strokeWidth={12}
                  dot={false}
                  isAnimationActive={false}
                  strokeOpacity={0.3}
                />

                {/* Data Points - Color Coded by Error */}
                {data?.tiktok.predictionAccuracy.data && (
                  <>
                    <Scatter
                      name="Dark Green (≤1% error)"
                      data={data.tiktok.predictionAccuracy.data
                        .filter((p: any) => p.color === "dark green")
                        .map((p: any) => ({ x: p.actual, y: p.predicted }))}
                      fill="#15803d"
                      fillOpacity={0.8}
                    />
                    <Scatter
                      name="Light Green (≤3% error)"
                      data={data.tiktok.predictionAccuracy.data
                        .filter((p: any) => p.color === "light green")
                        .map((p: any) => ({ x: p.actual, y: p.predicted }))}
                      fill="#84cc16"
                      fillOpacity={0.7}
                    />
                    <Scatter
                      name="Yellow (≤5% error)"
                      data={data.tiktok.predictionAccuracy.data
                        .filter((p: any) => p.color === "yellow")
                        .map((p: any) => ({ x: p.actual, y: p.predicted }))}
                      fill="#facc15"
                      fillOpacity={0.7}
                    />
                    <Scatter
                      name="Orange (≤7% error)"
                      data={data.tiktok.predictionAccuracy.data
                        .filter((p: any) => p.color === "orange")
                        .map((p: any) => ({ x: p.actual, y: p.predicted }))}
                      fill="#f97316"
                      fillOpacity={0.7}
                    />
                    <Scatter
                      name="Red (>7% error)"
                      data={data.tiktok.predictionAccuracy.data
                        .filter((p: any) => p.color === "red")
                        .map((p: any) => ({ x: p.actual, y: p.predicted }))}
                      fill="#ef4444"
                      fillOpacity={0.7}
                    />
                  </>
                )}

                <Legend wrapperStyle={{ fontSize: 10 }} />
              </ScatterChart>
            </ResponsiveContainer>

            <div className="bg-yellow-50 border border-yellow-200 p-4 rounded-lg mt-6">
              <div className="font-bold text-gray-800 mb-3">Engagement Rate</div>
              <div className="space-y-2">
                <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                  <span className="text-sm text-gray-700">R²</span>
                  <span className="font-semibold text-gray-900">{data?.tiktok.engagementRateModel.metrics.r2}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                  <span className="text-sm text-gray-700">MASE</span>
                  <span className="font-semibold text-gray-900">{data?.tiktok.engagementRateModel.metrics.mase}</span>
                </div>
                <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                  <span className="text-sm text-gray-700">MAE</span>
                  <span className="font-semibold text-gray-900">{data?.tiktok.engagementRateModel.metrics.mae}%</span>
                </div>
                <div className="flex justify-between items-center py-2">
                  <span className="text-sm text-gray-700">Zone</span>
                  <span className="font-semibold text-gray-900">{data?.tiktok.engagementRateModel.metrics.zone}</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Bottom Metrics Cards */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Channel Views Metrics */}
          <div className="bg-yellow-50 border border-yellow-200 p-6 rounded-lg">
            <div className="font-bold text-gray-800 mb-3">Channel Views</div>
            <div className="space-y-2">
              <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                <span className="text-sm text-gray-700">MAPE</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.channelViews.metrics.mape}%</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                <span className="text-sm text-gray-700">R²</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.channelViews.metrics.r2}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                <span className="text-sm text-gray-700">MASE</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.channelViews.metrics.mase}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-sm text-gray-700">MAE</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.channelViews.metrics.mae.toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Cumulative Forecast Metrics */}
          <div className="bg-yellow-50 border border-yellow-200 p-6 rounded-lg">
            <div className="font-bold text-gray-800 mb-3">Cumulative Forecast</div>
            <div className="space-y-2">
              <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                <span className="text-sm text-gray-700">MAPE</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.cumulativeForecast.metrics.mape}%</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                <span className="text-sm text-gray-700">R²</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.cumulativeForecast.metrics.r2}</span>
              </div>
              <div className="flex justify-between items-center py-2 border-b border-yellow-100">
                <span className="text-sm text-gray-700">MASE</span>
                <span className="font-semibold text-gray-900">{data?.tiktok.cumulativeForecast.metrics.mase}</span>
              </div>
              <div className="flex justify-between items-center py-2">
                <span className="text-sm text-gray-700">MAE</span>
                <span className="font-semibold text-gray-900">
                  {data?.tiktok.cumulativeForecast.metrics.mae.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
