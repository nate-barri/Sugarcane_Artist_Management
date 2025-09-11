"use client"

// Component for displaying social media integration status

import { useState } from "react"
import { apiService } from "@/lib/api"

interface IntegrationCardProps {
  platform: string
  displayName: string
  status: "connected" | "disconnected" | "error" | "pending"
  username?: string
  onStatusChange?: () => void
}

export default function IntegrationCard({
  platform,
  displayName,
  status,
  username,
  onStatusChange,
}: IntegrationCardProps) {
  const [loading, setLoading] = useState(false)

  const handleConnect = async () => {
    setLoading(true)
    try {
      const response = await apiService.connectPlatform(platform)
      if (response.data) {
        onStatusChange?.()
      }
    } catch (error) {
      console.error(`Failed to connect ${platform}:`, error)
    }
    setLoading(false)
  }

  const handleDisconnect = async () => {
    setLoading(true)
    try {
      const response = await apiService.disconnectPlatform(platform)
      if (response.data) {
        onStatusChange?.()
      }
    } catch (error) {
      console.error(`Failed to disconnect ${platform}:`, error)
    }
    setLoading(false)
  }

  const getStatusColor = () => {
    switch (status) {
      case "connected":
        return "text-green-600 bg-green-100"
      case "error":
        return "text-red-600 bg-red-100"
      case "pending":
        return "text-yellow-600 bg-yellow-100"
      default:
        return "text-gray-600 bg-gray-100"
    }
  }

  const getStatusText = () => {
    switch (status) {
      case "connected":
        return "Connected"
      case "error":
        return "Error"
      case "pending":
        return "Pending"
      default:
        return "Not Connected"
    }
  }

  return (
    <div className="bg-white p-6 rounded-lg shadow-md border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-900">{displayName}</h3>
        <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor()}`}>{getStatusText()}</span>
      </div>

      {username && <p className="text-sm text-gray-600 mb-4">@{username}</p>}

      <div className="flex space-x-2">
        {status === "connected" ? (
          <button
            onClick={handleDisconnect}
            disabled={loading}
            className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 text-sm"
          >
            {loading ? "Disconnecting..." : "Disconnect"}
          </button>
        ) : (
          <button
            onClick={handleConnect}
            disabled={loading}
            className="px-4 py-2 bg-[#123458] text-white rounded-md hover:bg-[#0e2742] disabled:opacity-50 text-sm"
          >
            {loading ? "Connecting..." : "Connect"}
          </button>
        )}
      </div>
    </div>
  )
}
