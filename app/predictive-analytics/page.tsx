"use client";

import Sidebar from "@/components/sidebar"
import { generateReport } from "@/utils/reportGenerator"


export default function PredictiveAnalyticsDashboard() {
  return (
    <div className="flex min-h-screen bg-[#123458] text-[#123458]">
      <Sidebar />

      {/* Main Content Area for Predictive Analytics Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">Predictive Analytics</h1>
        </header>

        {/* Best Time to Post Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Best Time to Post | Youtube</h2>
            <p className="text-xl font-bold text-gray-900">2:00 PM - 4:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Best Time to Post | Facebook</h2>
            <p className="text-xl font-bold text-gray-900">1:00 PM - 3:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Best Time to Post | Spotify</h2>
            <p className="text-xl font-bold text-gray-900">6:00 PM - 8:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Best Time to Post | Instagram</h2>
            <p className="text-xl font-bold text-gray-900">11:00 AM - 1:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-[#123458]">Best Time to Post | Tiktok</h2>
            <p className="text-xl font-bold text-gray-900">7:00 PM - 9:00 PM</p>
          </div>
        </section>

        {/* Charts Section for Predictive Analytics */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/150x150/123458/ffffff?text=Donut+Chart"
                alt="Engagement Chart Placeholder"
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
            <h2 className="text-xl font-semibold mb-4">Views</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Views Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Audience Growth</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Bar+Chart"
                alt="Audience Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Growth</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Line+Chart"
                alt="Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-[#FFFFFF]">
              <img
                src="https://placehold.co/250x150/123458/ffffff?text=Bar+Chart"
                alt="Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
        </section>

         {/* Generate Report Button */}
                <div className="flex justify-end">
                  <button
                    onClick={() => generateReport("Predictive Analytics")}
                    className="bg-[#123458] hover:bg-[#001F3F] text-[#FFFFFF] font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200"
                  >
                    GENERATE REPORT
                  </button>
                </div>
              </main>
            </div>
          )
        }
        