import Sidebar from "@/components/sidebar"

export default function PredictiveAnalyticsDashboard() {
  return (
    <div className="flex min-h-screen bg-gray-100 text-gray-800">
      <Sidebar />

      {/* Main Content Area for Predictive Analytics Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Predictive Analytics</h1>
        </header>

        {/* Best Time to Post Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Best Time to Post | Youtube</h2>
            <p className="text-xl font-bold text-gray-900">2:00 PM - 4:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Best Time to Post | Facebook</h2>
            <p className="text-xl font-bold text-gray-900">1:00 PM - 3:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Best Time to Post | Spotify</h2>
            <p className="text-xl font-bold text-gray-900">6:00 PM - 8:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Best Time to Post | Instagram</h2>
            <p className="text-xl font-bold text-gray-900">11:00 AM - 1:00 PM</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Best Time to Post | Tiktok</h2>
            <p className="text-xl font-bold text-gray-900">7:00 PM - 9:00 PM</p>
          </div>
        </section>

        {/* Charts Section for Predictive Analytics */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/150x150/fca5a5/ffffff?text=Donut+Chart"
                alt="Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Impressions & Reach</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Line+Chart"
                alt="Impressions & Reach Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Views</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Line+Chart"
                alt="Views Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Audience Growth</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Bar+Chart"
                alt="Audience Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Growth</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Line+Chart"
                alt="Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Bar+Chart"
                alt="Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
        </section>

        {/* Generate Report Button */}
        <div className="flex justify-end">
          <button className="bg-red-600 hover:bg-red-700 text-white font-bold py-3 px-6 rounded-lg shadow-lg flex items-center transition-colors duration-200">
            GENERATE REPORT
            <svg className="w-5 h-5 ml-2" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10.293 15.707a1 1 0 010-1.414L14.586 10l-4.293-4.293a1 1 0 111.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z"
                clipRule="evenodd"
              ></path>
              <path
                fillRule="evenodd"
                d="M4.293 15.707a1 1 0 010-1.414L8.586 10 4.293 5.707a1 1 0 011.414-1.414l5 5a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0z"
                clipRule="evenodd"
              ></path>
            </svg>
          </button>
        </div>
      </main>
    </div>
  )
}
