import Sidebar from "@/components/sidebar"

export default function FAQDashboard() {
  return (
    <div className="flex min-h-screen bg-[#123458] text-[#123458]">
      <Sidebar />

      {/* Main Content Area for FAQ Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#FFFFFF]">Frequently Asked Questions</h1>
        </header>

        {/* FAQ Section */}
        <section className="space-y-6">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">How do I add my social media data?</h2>
            <p className="text-gray-600 leading-relaxed">
              Navigate to the Import page from the sidebar and upload your analytics data files. The system supports CSV
              and XLSX formats and will automatically detect whether your data is from YouTube, TikTok, Meta (Facebook),
              or Spotify. Simply select your file and the system will process and load it into the database.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">How often is the data updated?</h2>
            <p className="text-gray-600 leading-relaxed">
              Data is updated whenever you import new files through the Import page. The system analyzes historical data
              from your uploaded CSV/Excel files and displays comprehensive analytics across all supported platforms.
              There is no automatic real-time syncing - you control when data is imported.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">What platforms are supported?</h2>
            <p className="text-gray-600 leading-relaxed">
              The platform currently supports YouTube and TikTok with full analytics dashboards. You can also import
              data from Meta (Facebook) and Spotify. Each platform has dedicated analytics pages showing metrics like
              views, engagement rates, content performance, top videos/songs, duration analysis, and content type
              distribution.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">What data format do I need for uploads?</h2>
            <p className="text-gray-600 leading-relaxed">
              Your CSV or Excel files should include platform-specific columns: YouTube requires Video ID, Video title,
              and Video publish time; TikTok needs tiktok_video_id, content_link, and video_title; Meta requires
              post_id, page_name, and publish_time; Spotify needs song, listeners, and streams columns. The system will
              automatically detect the platform based on these columns.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">What analytics can I see?</h2>
            <p className="text-gray-600 leading-relaxed">
              The dashboards display comprehensive metrics including total views, engagement rates, likes, shares,
              comments, watch time, and unique viewers. You can analyze performance trends over time, view
              top-performing content, compare content types, examine duration-based engagement patterns, and track
              monthly/yearly performance through interactive charts and visualizations.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">How is my data processed and stored?</h2>
            <p className="text-gray-600 leading-relaxed">
              When you upload a file, the system runs ETL (Extract, Transform, Load) scripts that clean and process your
              data, removing emojis from text fields, parsing dates and durations, calculating engagement metrics, and
              loading everything into a PostgreSQL database. The data is then queried in real-time to generate the
              analytics visualizations you see on each platform dashboard.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Are there any file size limitations?</h2>
            <p className="text-gray-600 leading-relaxed">
              Yes, the maximum file size per upload is 50MB. For larger datasets, consider splitting your data into
              multiple files or filtering to include only the most relevant time periods. The system can handle
              thousands of records per import and uses efficient batch processing to load data quickly.
            </p>
          </div>
        </section>
      </main>
    </div>
  )
}
