"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import Sidebar from "@/components/sidebar"
import { generateReport } from "@/utils/reportGenerator"
import { useAuth } from "@/components/auth-provider"
import { useDashboardData } from "@/hooks/use-api"

export default function Dashboard() {
  const { user, loading: authLoading } = useAuth()
  const { data: dashboardData, isLoading: dataLoading, error, refresh } = useDashboardData()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !user) {
      router.push("/login")
    }
  }, [user, authLoading, router])

  if (authLoading || dataLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl text-red-600">Error loading dashboard data</div>
      </div>
    )
  }

  const metrics = dashboardData || {
    total_subscribers: 0,
    total_views: 0,
    total_watch_time: 0,
    total_spotify_streams: 0,
    audience_growth: 0,
    top_performing_platform: "N/A",
  }

  return (
    <div className="flex min-h-screen bg-[#123458] text-[#123458]">
      <Sidebar />

      {/* Main Content Area */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">Dashboard Overview</h1>
          <button
            onClick={() => refresh()}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-[#FFFFFF] font-medium py-2 px-4 rounded-lg transition-colors duration-200"
          >
            Refresh Data
          </button>
        </header>

        {/* Key Metrics Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Subscribers</h2>
            <p className="text-3xl font-bold text-gray-900">{metrics.total_subscribers.toLocaleString()}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Views</h2>
            <p className="text-3xl font-bold text-gray-900">{metrics.total_views.toLocaleString()}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Watch time</h2>
            <p className="text-3xl font-bold text-gray-900">{metrics.total_watch_time.toLocaleString()} HRS</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Spotify Streams</h2>
            <p className="text-3xl font-bold text-gray-900">{metrics.total_spotify_streams.toLocaleString()}</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Audience Growth</h2>
            <p className="text-3xl font-bold text-gray-900">{metrics.audience_growth}%</p>
          </div>
        </section>

        {/* Charts Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Overall Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/150x150/123458/ffffff?text=Donut+Chart"
                alt="Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Spotify Streams Overtime</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Spotify Streams Overtime Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Impressions & Reach</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Impressions & Reach Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Audience Retention Overtime</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Audience Retention Overtime Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Views Overtime</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Views Overtime Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
        </section>

        {/* Top Performing Platform Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Top Performing Platform</h2>
          <p className="text-2xl font-bold text-gray-900">{metrics.top_performing_platform}</p>
        </section>

        {/* Generate Report Button */}
        <div className="flex justify-end">
          <button
            onClick={() => generateReport("Dashboard Overview")}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-[#FFFFFF] font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200"
          >
            GENERATE REPORT
          </button>
        </div>
      </main>
    </div>
  )
}
