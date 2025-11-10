"use client";

import Sidebar from "@/components/sidebar"
import { generateReport } from "@/utils/reportGenerator"

export default function CrossPlatformDashboard() {
  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-[#123458]">
      <Sidebar />

      {/* Main Content Area for Cross-Platform Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">Cross-Platform Analytics</h1>
        </header>

        {/* Platform Comparison Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">YouTube Performance</h2>
            <p className="text-3xl font-bold text-[#FF0000]">85%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Facebook Performance</h2>
            <p className="text-3xl font-bold text-[#4267B2]">72%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Spotify Performance</h2>
            <p className="text-3xl font-bold text-[#1DB954]">91%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">TikTok Performance</h2>
            <p className="text-3xl font-bold text-[#FE2C55]">88%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
        </section>

        {/* Cross-Platform Charts Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Platform Comparison</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Radar+Chart"
                alt="Platform Comparison Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Unified Audience Growth</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Multi-Line+Chart"
                alt="Unified Audience Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Content Performance Across Platforms</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Stacked+Bar+Chart"
                alt="Content Performance Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Cross-Platform Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Area+Chart"
                alt="Cross-Platform Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
        </section>

        {/* Generate Report Button */}
        <div className="flex justify-end">
          <button
            onClick={() => generateReport("Crossplatform")}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-[#FFFFFF] font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200"
          >
            GENERATE REPORT
          </button>
        </div>
      </main>
    </div>
  )
}
