"use client"

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/components/auth-provider"
import { useIntegrations } from "@/hooks/use-api"
import Sidebar from "@/components/sidebar"
import IntegrationCard from "@/components/integration-card"

export default function IntegrationsPage() {
  const { user, loading: authLoading } = useAuth()
  const { integrations, summary, isLoading, error, refresh } = useIntegrations()
  const router = useRouter()

  useEffect(() => {
    if (!authLoading && !user) {
      router.push("/login")
    }
  }, [user, authLoading, router])

  if (authLoading || isLoading) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-xl text-gray-600">Loading...</div>
      </div>
    )
  }

  if (!user) {
    return null
  }

  const availablePlatforms = [
    { name: "youtube", displayName: "YouTube" },
    { name: "spotify", displayName: "Spotify" },
    { name: "instagram", displayName: "Instagram" },
    { name: "tiktok", displayName: "TikTok" },
    { name: "facebook", displayName: "Facebook" },
    { name: "twitter", displayName: "Twitter" },
  ]

  const getIntegrationStatus = (platformName: string) => {
    const integration = integrations.find((int: any) => int.platform === platformName)
    return integration || { platform: platformName, status: "disconnected" }
  }

  return (
    <div className="flex min-h-screen bg-[#123458]">
      <Sidebar />

      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white">Social Media Integrations</h1>
            <p className="text-gray-300 mt-2">Connect your social media accounts to sync analytics data</p>
          </div>
          <button
            onClick={() => refresh()}
            className="bg-[#0f2946] hover:bg-[#001F3F] text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200"
          >
            Refresh Status
          </button>
        </header>

        {/* Summary Stats */}
        <div className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Integration Summary</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-[#123458]">{summary.total_connected || 0}</div>
              <div className="text-sm text-gray-600">Connected Platforms</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-600">{availablePlatforms.length}</div>
              <div className="text-sm text-gray-600">Available Platforms</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {Math.round(((summary.total_connected || 0) / availablePlatforms.length) * 100)}%
              </div>
              <div className="text-sm text-gray-600">Integration Rate</div>
            </div>
          </div>
        </div>

        {/* Integration Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {availablePlatforms.map((platform) => {
            const integration = getIntegrationStatus(platform.name)
            return (
              <IntegrationCard
                key={platform.name}
                platform={platform.name}
                displayName={platform.displayName}
                status={integration.status}
                username={integration.platform_username}
                onStatusChange={refresh}
              />
            )
          })}
        </div>

        {error && (
          <div className="mt-8 bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
            Error loading integrations: {error.message}
          </div>
        )}
      </main>
    </div>
  )
}
