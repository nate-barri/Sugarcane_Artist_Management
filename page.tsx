"use client";

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import Sidebar from "@/components/sidebar"
import { generateReport } from "@/utils/reportGenerator"

export default function Dashboard() {
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [loading, setLoading] = useState(true)
  const router = useRouter()

  useEffect(() => {
    console.log("[v0] Authentication check starting...")
    const authToken = localStorage.getItem("authToken")
    console.log("[v0] AuthToken found:", authToken)

    if (!authToken) {
      console.log("[v0] No auth token, redirecting to login...")
      router.push("/login")
    } else {
      console.log("[v0] Auth token exists, setting authenticated to true")
      setIsAuthenticated(true)
    }
    setLoading(false)
    console.log("[v0] Loading set to false")
  }, [router])

  console.log("[v0] Render - Loading:", loading, "Authenticated:", isAuthenticated)

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    )
  }

  if (!isAuthenticated) {
    console.log("[v0] Not authenticated, returning null")
    return null
  }

  return (
    <div className="flex min-h-screen bg-[#123458] text-[#123458]">
      <Sidebar />

    
      {/* Main Content Area */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">Dashboard Overview</h1>
        </header>

        {/* Key Metrics Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Subscribers</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Views</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Watch time</h2>
            <p className="text-3xl font-bold text-gray-900">0 HRS</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Total Spotify Streams</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Audience Growth</h2>
            <p className="text-3xl font-bold text-gray-900">0%</p>
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
          <p className="text-2xl font-bold text-gray-900">Spotify</p>
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
