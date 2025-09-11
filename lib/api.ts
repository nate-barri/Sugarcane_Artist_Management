const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api"

interface ApiResponse<T = any> {
  data?: T
  error?: string
  message?: string
}

class ApiService {
  private baseURL: string
  private token: string | null = null

  constructor() {
    this.baseURL = API_BASE_URL
    // Initialize token from localStorage if available
    if (typeof window !== "undefined") {
      this.token = localStorage.getItem("authToken")
    }
  }

  setToken(token: string) {
    this.token = token
    if (typeof window !== "undefined") {
      localStorage.setItem("authToken", token)
    }
  }

  clearToken() {
    this.token = null
    if (typeof window !== "undefined") {
      localStorage.removeItem("authToken")
      localStorage.removeItem("refreshToken")
      localStorage.removeItem("currentUser")
    }
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<ApiResponse<T>> {
    const url = `${this.baseURL}${endpoint}`

    const headers: HeadersInit = {
      "Content-Type": "application/json",
      ...options.headers,
    }

    if (this.token) {
      headers.Authorization = `Bearer ${this.token}`
    }

    try {
      const response = await fetch(url, {
        ...options,
        headers,
        // Add timeout to prevent hanging requests
        signal: AbortSignal.timeout(10000), // 10 second timeout
      })

      // Check if response is ok before trying to parse JSON
      if (!response.ok) {
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`

        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorData.error || errorMessage
        } catch {
          // If JSON parsing fails, use the HTTP status message
        }

        // Handle token refresh if needed
        if (response.status === 401 && this.token) {
          const refreshed = await this.refreshToken()
          if (refreshed) {
            // Retry the original request with new token
            headers.Authorization = `Bearer ${this.token}`
            const retryResponse = await fetch(url, {
              ...options,
              headers,
              signal: AbortSignal.timeout(10000),
            })

            if (retryResponse.ok) {
              const retryData = await retryResponse.json()
              return { data: retryData }
            }
          } else {
            this.clearToken()
            if (typeof window !== "undefined") {
              window.location.href = "/login"
            }
          }
        }

        return {
          error: errorMessage,
          data: null,
        }
      }

      const data = await response.json()
      return { data }
    } catch (error) {
      console.error("API request failed:", error)

      // Provide more specific error messages
      let errorMessage = "Network error occurred"

      if (error instanceof TypeError && error.message.includes("fetch")) {
        errorMessage = "Unable to connect to server. Please check if the backend is running."
      } else if (error instanceof Error && error.name === "AbortError") {
        errorMessage = "Request timed out. Please try again."
      } else if (error instanceof Error) {
        errorMessage = error.message
      }

      return {
        error: errorMessage,
        data: null,
      }
    }
  }

  private async refreshToken(): Promise<boolean> {
    const refreshToken = localStorage.getItem("refreshToken")
    if (!refreshToken) return false

    try {
      const response = await fetch(`${this.baseURL}/auth/refresh/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ refresh: refreshToken }),
        signal: AbortSignal.timeout(5000), // 5 second timeout for refresh
      })

      if (response.ok) {
        const data = await response.json()
        this.setToken(data.access)
        localStorage.setItem("refreshToken", data.refresh)
        return true
      }
    } catch (error) {
      console.error("Token refresh failed:", error)
    }

    return false
  }

  // Authentication endpoints
  async login(email: string, password: string): Promise<ApiResponse<any>> {
    const response = await this.request("/auth/login/", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    })

    if (response.data) {
      this.setToken(response.data.access)
      localStorage.setItem("refreshToken", response.data.refresh)
      localStorage.setItem("currentUser", JSON.stringify(response.data.user))
    }

    return response
  }

  async register(userData: {
    email: string
    username: string
    full_name: string
    password: string
    password_confirm: string
  }): Promise<ApiResponse<any>> {
    const response = await this.request("/auth/register/", {
      method: "POST",
      body: JSON.stringify(userData),
    })

    if (response.data) {
      this.setToken(response.data.access)
      localStorage.setItem("refreshToken", response.data.refresh)
      localStorage.setItem("currentUser", JSON.stringify(response.data.user))
    }

    return response
  }

  async logout(): Promise<ApiResponse<any>> {
    const refreshToken = localStorage.getItem("refreshToken")
    if (refreshToken) {
      await this.request("/auth/logout/", {
        method: "POST",
        body: JSON.stringify({ refresh: refreshToken }),
      })
    }

    this.clearToken()
    return { data: { message: "Logged out successfully" } }
  }

  async getProfile(): Promise<ApiResponse<any>> {
    return this.request("/auth/profile/")
  }

  async updateProfile(profileData: any): Promise<ApiResponse<any>> {
    return this.request("/auth/profile/update/", {
      method: "PUT",
      body: JSON.stringify(profileData),
    })
  }

  // Analytics endpoints
  async getDashboardOverview(): Promise<ApiResponse<any>> {
    return this.request("/analytics/dashboard/")
  }

  async getPlatformAnalytics(platform: string): Promise<ApiResponse<any>> {
    return this.request(`/analytics/platform/${platform}/`)
  }

  async syncPlatformData(platform: string): Promise<ApiResponse<any>> {
    return this.request(`/analytics/sync/${platform}/`, {
      method: "POST",
    })
  }

  // Integration endpoints
  async getIntegrations(): Promise<ApiResponse<any>> {
    return this.request("/integrations/")
  }

  async getIntegrationStatus(): Promise<ApiResponse<any>> {
    return this.request("/integrations/status/")
  }

  async connectPlatform(platform: string): Promise<ApiResponse<any>> {
    return this.request(`/integrations/connect/${platform}/`, {
      method: "POST",
    })
  }

  async disconnectPlatform(platform: string): Promise<ApiResponse<any>> {
    return this.request(`/integrations/disconnect/${platform}/`, {
      method: "POST",
    })
  }

  // Reports endpoints
  async getReports(): Promise<ApiResponse<any>> {
    return this.request("/reports/")
  }

  async generateReport(reportData: any): Promise<ApiResponse<any>> {
    return this.request("/reports/generate/", {
      method: "POST",
      body: JSON.stringify(reportData),
    })
  }

  async downloadReport(reportId: number): Promise<ApiResponse<any>> {
    return this.request(`/reports/download/${reportId}/`)
  }
}

export const apiService = new ApiService()
export default apiService
