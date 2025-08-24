"use client";

import Sidebar from "@/components/sidebar"
import { generateReport } from "@/utils/reportGenerator"

export default function YouTubeDashboard() {
  return (
    <div className="flex min-h-screen bg-[#123458] text-[#123458]">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">YouTube Dashboard</h1>
        </header>

        {/* Key Metrics Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-lg font-medium text-[#123458]">Total Views</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-lg font-medium text-[#123458]">Total Watch time</h2>
            <p className="text-3xl font-bold text-gray-900">0 Hours</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-lg font-medium text-[#123458]">Subscribers Growth</h2>
            <p className="text-3xl font-bold text-gray-900">0%</p>
          </div>
        </section>

        {/* Charts Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-xl font-semibold text-[#1B3C53]">Views Overtime</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF] rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Bar+Chart"
                alt="Views Overtime Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-xl font-semibold text-[#1B3C53]">Audience Retention</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF] rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Audience Retention Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-xl font-semibold text-[#1B3C53]">Subscribers Growth</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF] rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Subscribers Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md border border-[#123458]">
            <h2 className="text-xl font-semibold text-[#1B3C53]">Watch Time</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF] rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Multi-Line+Chart"
                alt="Watch Time Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
        </section>

        {/* Generate Report Button */}
        <div className="flex justify-end">
          <button
            onClick={() => generateReport("YouTube")}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-[#FFFFFF] font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200"
          >
            GENERATE REPORT
          </button>
        </div>
      </main>
    </div>
  )
}