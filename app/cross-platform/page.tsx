import Sidebar from "@/components/sidebar"

export default function CrossPlatformDashboard() {
  return (
    <div className="flex min-h-screen bg-gray-100 text-gray-800">
      <Sidebar />

      {/* Main Content Area for Cross-Platform Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Cross-Platform Analytics</h1>
        </header>

        {/* Platform Comparison Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">YouTube Performance</h2>
            <p className="text-3xl font-bold text-red-600">85%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Facebook Performance</h2>
            <p className="text-3xl font-bold text-blue-600">72%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Spotify Performance</h2>
            <p className="text-3xl font-bold text-green-600">91%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Instagram Performance</h2>
            <p className="text-3xl font-bold text-purple-600">78%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">TikTok Performance</h2>
            <p className="text-3xl font-bold text-pink-600">88%</p>
            <p className="text-sm text-gray-500">Engagement Score</p>
          </div>
        </section>

        {/* Cross-Platform Charts Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Platform Comparison</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Radar+Chart"
                alt="Platform Comparison Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Unified Audience Growth</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Multi-Line+Chart"
                alt="Unified Audience Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Content Performance Across Platforms</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Stacked+Bar+Chart"
                alt="Content Performance Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Cross-Platform Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Area+Chart"
                alt="Cross-Platform Engagement Chart Placeholder"
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
