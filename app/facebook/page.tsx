import Sidebar from "@/components/sidebar"

export default function FacebookDashboard() {
  return (
    <div className="flex min-h-screen bg-gray-100 text-gray-800">
      <Sidebar />

      {/* Main Content Area for Facebook Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Meta Facebook Dashboard</h1>
        </header>

        {/* Key Metrics Section for Facebook */}
        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Page Likes</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Post Reach</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Engagement Rate</h2>
            <p className="text-3xl font-bold text-gray-900">0%</p>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col items-start">
            <h2 className="text-lg font-medium text-gray-600 mb-2">Page Views</h2>
            <p className="text-3xl font-bold text-gray-900">000,000.00</p>
          </div>
        </section>

        {/* Charts Section for Facebook */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Page Likes Growth</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Line+Chart"
                alt="Page Likes Growth Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Post Engagement</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Bar+Chart"
                alt="Post Engagement Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Reach & Impressions</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/250x150/fca5a5/ffffff?text=Area+Chart"
                alt="Reach & Impressions Chart Placeholder"
                className="rounded"
              />
            </div>
          </div>
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Audience Demographics</h2>
            <div className="flex justify-center items-center h-48 bg-gray-50 rounded-lg">
              <img
                src="https://placehold.co/150x150/fca5a5/ffffff?text=Pie+Chart"
                alt="Audience Demographics Chart Placeholder"
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
