"use client";

import Sidebar from "@/components/sidebar"
import { generateReport } from "@/utils/reportGenerator"

export default function FacebookDashboard() {
  return (
    <div className="flex min-h-screen bg-[#123458] text-gray-800">
      <Sidebar />
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">Meta</h1>
        </header>

        {/* Key Metrics Section for Facebook */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Page Likes</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Post Reach</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Engagement Rate</h2>
            <p className="text-3xl font-bold text-gray-900">0%</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Page Views</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
        </section>

        {/* Charts Section for Facebook */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Page Likes Growth</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Page Likes Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Post Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Bar+Chart"
                alt="Post Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Reach & Impressions</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Area+Chart"
                alt="Reach & Impressions Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-[#123458]">Audience Demographics</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]/10 rounded-lg">
              <img
                src="https://placehold.co/150x150/123458/ffffff?text=Pie+Chart"
                alt="Audience Demographics Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
        </section>

        {/* Generate Report Button */}
        <div className="flex justify-end">
          <button
            onClick={() => generateReport("Facebook")}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-white font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200"
          >
            GENERATE REPORT
          </button>
        </div>
      </main>
    </div>
  )
}
