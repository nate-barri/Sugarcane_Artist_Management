import Sidebar from "@/components/sidebar"

export default function ImportDashboard() {
  return (
    <div className="flex min-h-screen bg-[#123458] text-[#123458]">
      <Sidebar />

      {/* Main Content Area for Import Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">Import Data</h1>
        </header>

                {/* Import Options Section */}
        <section className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* CSV Import Card */}
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-between">
            <div>
              <div className="flex items-center mb-4">
                <div className="bg-[#E6F4FA] p-2 rounded-full">
                  <svg className="w-6 h-6 text-[#008000]" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"></path>
                  </svg>
                </div>
                <h2 className="ml-3 text-xl font-semibold text-gray-900">CSV Import</h2>
              </div>
              <p className="text-gray-600 mb-6">Upload your analytics data using CSV format.</p>
            </div>
            <button className="w-full bg-[#3396D3] hover:bg-[#2A75A4] text-white font-semibold py-2 px-4 rounded-lg transition duration-200">
              Choose CSV File
            </button>
          </div>

          {/* Excel Import Card */}
          <div className="bg-white p-6 rounded-lg shadow-md flex flex-col justify-between">
            <div>
              <div className="flex items-center mb-4">
                <div className="bg-[#E6F4FA] p-2 rounded-full">
                  <svg className="w-6 h-6 text-[#008000]" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M3 4a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1V4zM3 10a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H4a1 1 0 01-1-1v-6zM14 9a1 1 0 00-1 1v6a1 1 0 001 1h2a1 1 0 001-1v-6a1 1 0 00-1-1h-2z"></path>
                  </svg>
                </div>
                <h2 className="ml-3 text-xl font-semibold text-gray-900">Excel Import</h2>
              </div>
              <p className="text-gray-600 mb-6">Upload your data using Excel spreadsheets (.xlsx).</p>
            </div>
            <button className="w-full bg-[#3396D3] hover:bg-[#2A75A4] text-white font-semibold py-2 px-4 rounded-lg transition duration-200">
              Choose Excel File
            </button>
          </div>
        </section>

        {/* Import History Section */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Recent Imports</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="pb-3 text-gray-600 font-medium">File Name</th>
                  <th className="pb-3 text-gray-600 font-medium">Type</th>
                  <th className="pb-3 text-gray-600 font-medium">Date</th>
                  <th className="pb-3 text-gray-600 font-medium">Status</th>
                  <th className="pb-3 text-gray-600 font-medium">Records</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100">
                  <td className="py-3 text-gray-900">youtube_analytics_2024.csv</td>
                  <td className="py-3 text-gray-600">CSV</td>
                  <td className="py-3 text-gray-600">2024-01-15</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-sm">Success</span>
                  </td>
                  <td className="py-3 text-gray-600">1,250</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 text-gray-900">instagram_data.xlsx</td>
                  <td className="py-3 text-gray-600">Excel</td>
                  <td className="py-3 text-gray-600">2024-01-14</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-sm">Success</span>
                  </td>
                  <td className="py-3 text-gray-600">890</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-3 text-gray-900">spotify_metrics.json</td>
                  <td className="py-3 text-gray-600">JSON</td>
                  <td className="py-3 text-gray-600">2024-01-13</td>
                  <td className="py-3">
                    <span className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded-full text-sm">Processing</span>
                  </td>
                  <td className="py-3 text-gray-600">2,100</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* Import Guidelines */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Import Guidelines</h2>
          <div className="space-y-3 text-gray-600">
            <p>• Ensure your data files are properly formatted with headers</p>
            <p>• Maximum file size: 50MB per upload</p>
            <p>• Supported date formats: YYYY-MM-DD, MM/DD/YYYY, DD/MM/YYYY</p>
            <p>• Required columns: date, platform, metric_name, metric_value</p>
            <p>• Data will be processed and available in your dashboard within 5-10 minutes</p>
          </div>
        </section>
      </main>
    </div>
  )
}