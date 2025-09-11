import useSWR from "swr"
import { apiService } from "@/lib/api"

// Enhanced fetcher with better error handling
const fetcher = async (url: string) => {
  const response = await apiService.getDashboardOverview()
  if (response.error) {
    throw new Error(response.error)
  }
  return response.data
}

const integrationsFetcher = async () => {
  const response = await apiService.getIntegrations()
  if (response.error) {
    // Return mock data if backend is not available
    console.warn("Backend not available, using mock data:", response.error)
    return {
      integrations: [],
      summary: { total_connected: 0 },
    }
  }
  return response.data
}

const platformFetcher = async (platform: string) => {
  const response = await apiService.getPlatformAnalytics(platform)
  if (response.error) {
    // Return mock data if backend is not available
    console.warn("Backend not available, using mock data:", response.error)
    return {
      platform,
      metrics: {
        followers: 0,
        engagement_rate: 0,
        total_posts: 0,
        avg_likes: 0,
      },
      recent_posts: [],
    }
  }
  return response.data
}

export function useDashboard() {
  const { data, error, isLoading, mutate } = useSWR("/dashboard", fetcher, {
    refreshInterval: 30000, // Refresh every 30 seconds
    revalidateOnFocus: false,
    fallbackData: {
      // Provide fallback data structure
      total_followers: 0,
      total_engagement: 0,
      total_posts: 0,
      platforms: [],
      recent_activity: [],
    },
    onError: (error) => {
      console.warn("Dashboard API error:", error.message)
    },
  })

  return {
    dashboard: data,
    isLoading,
    error,
    refresh: mutate,
  }
}

export function useIntegrations() {
  const { data, error, isLoading, mutate } = useSWR("/integrations", integrationsFetcher, {
    refreshInterval: 60000, // Refresh every minute
    revalidateOnFocus: false,
    fallbackData: {
      integrations: [],
      summary: { total_connected: 0 },
    },
    onError: (error) => {
      console.warn("Integrations API error:", error.message)
    },
  })

  return {
    integrations: data?.integrations || [],
    summary: data?.summary || { total_connected: 0 },
    isLoading,
    error,
    refresh: mutate,
  }
}

export function usePlatformAnalytics(platform: string) {
  const { data, error, isLoading, mutate } = useSWR(
    platform ? `/platform/${platform}` : null,
    () => platformFetcher(platform),
    {
      refreshInterval: 60000,
      revalidateOnFocus: false,
      fallbackData: {
        platform,
        metrics: {
          followers: 0,
          engagement_rate: 0,
          total_posts: 0,
          avg_likes: 0,
        },
        recent_posts: [],
      },
      onError: (error) => {
        console.warn(`${platform} API error:`, error.message)
      },
    },
  )

  return {
    analytics: data,
    isLoading,
    error,
    refresh: mutate,
  }
}

export function useReports() {
  const { data, error, isLoading, mutate } = useSWR(
    "/reports",
    async () => {
      const response = await apiService.getReports()
      if (response.error) {
        console.warn("Reports API error:", response.error)
        return []
      }
      return response.data
    },
    {
      refreshInterval: 120000, // Refresh every 2 minutes
      fallbackData: [],
      onError: (error) => {
        console.warn("Reports API error:", error.message)
      },
    },
  )

  return {
    reports: data || [],
    isLoading,
    error,
    refresh: mutate,
  }
}
