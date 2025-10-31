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
            <h2 className="text-xl font-semibold mb-4 text-gray-900">How do I connect my social media accounts?</h2>
            <p className="text-gray-600 leading-relaxed">
              To connect your social media accounts, navigate to the Settings page and click on "Connect Accounts".
              Follow the authentication process for each platform you want to integrate with your dashboard.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">How often is the data updated?</h2>
            <p className="text-gray-600 leading-relaxed">
              Data is automatically updated every 4 hours for most platforms. Premium accounts receive real-time updates
              for critical metrics like engagement rates and follower counts.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">What does the predictive analytics feature do?</h2>
            <p className="text-gray-600 leading-relaxed">
              Our predictive analytics uses machine learning algorithms to analyze your historical data and predict
              optimal posting times, content performance, and audience growth trends to help maximize your engagement.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Can I export my analytics data?</h2>
            <p className="text-gray-600 leading-relaxed">
              Yes! You can generate and download comprehensive reports in PDF or CSV format using the "Generate Report"
              button available on each dashboard page.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">How do I interpret the engagement metrics?</h2>
            <p className="text-gray-600 leading-relaxed">
              Engagement metrics include likes, comments, shares, and saves across all platforms. Higher engagement
              rates typically indicate better content performance and audience connection.
            </p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4 text-gray-900">Is my data secure?</h2>
            <p className="text-gray-600 leading-relaxed">
              Absolutely. We use enterprise-grade encryption and follow industry best practices for data security. Your
              social media credentials are never stored on our servers, and all data transmission is encrypted.
            </p>
          </div>
        </section>
      </main>
    </div>
  )
}