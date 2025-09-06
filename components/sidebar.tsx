"use client"

import type React from "react"
import { useState, useEffect } from "react"

// A simple mock for the useAuth hook to avoid external dependencies
const useAuth = () => {
  // Explicitly set the type of the user state to allow it to be null
  const [user, setUser] = useState<{ name: string; email: string } | null>({
    name: "John Doe",
    email: "john.doe@example.com",
  })
  const logout = () => {
    console.log("Logged out.")
    setUser(null)
  }
  return { user, logout }
}

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isDashboardOpen, setIsDashboardOpen] = useState(true)
  // Mock pathname state to replace usePathname from next/navigation
  const [pathname, setPathname] = useState("/")
  const { user, logout } = useAuth()

  // Load sidebar state from localStorage on initial render
  useEffect(() => {
    const storedCollapsedState = localStorage.getItem("sidebarCollapsedState")
    if (storedCollapsedState === "true") {
      setIsCollapsed(true)
      setIsDashboardOpen(false)
    }
  }, [])

  // Save sidebar state to localStorage whenever it changes
  useEffect(() => {
    localStorage.setItem("sidebarCollapsedState", isCollapsed.toString())
    if (isCollapsed) {
      setIsDashboardOpen(false)
    }
  }, [isCollapsed])

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed)
  }

  const toggleDashboard = (e: React.MouseEvent) => {
    // Only toggle if the target is the arrow icon, or if it's the dashboard link itself
    if ((e.target as HTMLElement).closest("#dashboard-arrow")) {
      e.preventDefault()
      if (!isCollapsed) {
        setIsDashboardOpen(!isDashboardOpen)
      }
    }
  }

  // Check if a link is active based on a mock pathname
  const isActiveLink = (href: string) => {
    if (href === "/" && pathname === "/") return true
    if (href !== "/" && pathname.includes(href.replace(".html", ""))) return true
    return false
  }

  const dashboardPages = [
    { href: "/youtube", label: "YouTube" },
    { href: "/facebook", label: "Meta Facebook" },
    { href: "/spotify", label: "Spotify" },
    { href: "/instagram", label: "Instagram" },
    { href: "/tiktok", label: "TikTok" },
  ]

  // Determine if the dashboard parent link should be active
  const isDashboardActive = pathname === "/" || dashboardPages.some((page) => pathname.includes(page.href))

  return (
    <aside
      // Changed background color to white and text color to a dark gray as requested
      className={`sidebar bg-white text-gray-800 shadow-lg p-6 flex flex-col rounded-r-lg transition-all duration-300 ease-in-out ${
        isCollapsed ? "w-20 items-center" : "w-64"
      }`}
    >
      <div className={`mb-8 flex items-center ${isCollapsed ? 'justify-center' : 'justify-between'} logo-container`}>
        {/* Hamburger Icon */}
        <button onClick={toggleSidebar} className="text-gray-800 focus:outline-none p-2 rounded-lg hover:bg-gray-200">
          <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M3 5a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM3 15a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
              clipRule="evenodd"
            ></path>
          </svg>
        </button>

        {/* Logo */}
        {!isCollapsed && (
          // Updated logo placeholder to have dark text on a light background
          <img src="https://placehold.co/100x32/ffffff/000000?text=SUGARCANE+LOGO" alt="Sugarcane Logo" className="h-8 w-auto rounded logo-text" />
        )}
      </div>

      {user && (
        <div className={`mb-6 p-3 bg-gray-200 rounded-lg ${isCollapsed ? 'hidden' : ''}`}>
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-800 truncate">{user.name}</p>
              <p className="text-xs text-gray-600 truncate">{user.email}</p>
            </div>
          </div>
        </div>
      )}

      <nav className="flex-1">
        <ul>
          {/* Dashboard */}
          <li className="mb-4">
            <a
              href="#"
              onClick={(e) => {
                setPathname("/")
                toggleDashboard(e)
              }}
              className={`flex items-center p-3 rounded-lg font-semibold shadow-sm cursor-pointer nav-item transition-colors duration-200 ${
                isDashboardActive ? "bg-gray-200 text-[#123458]" : "text-gray-800 hover:bg-gray-200"
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z"></path>
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>Dashboard</span>
              {!isCollapsed && (
                <svg
                  id="dashboard-arrow"
                  className={`w-4 h-4 ml-auto transform transition-transform duration-200 ${
                    isDashboardOpen ? "rotate-180" : ""
                  }`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  ></path>
                </svg>
              )}
            </a>

            {!isCollapsed && (
              <ul className={`submenu pl-8 pt-2 transition-all duration-200 ${isDashboardOpen ? "active" : "hidden"}`}>
                {dashboardPages.map((page) => (
                  <li key={page.href} className="mb-2">
                    <a
                      href="#"
                      onClick={(e) => {
                        e.preventDefault()
                        setPathname(page.href)
                      }}
                      className={`flex items-center p-2 rounded-lg transition-colors duration-200 nav-item ${
                        isActiveLink(page.href)
                          ? "bg-gray-300 text-gray-900 font-semibold"
                          : "text-gray-600 hover:bg-gray-200 hover:text-gray-900"
                      }`}
                    >
                      <span className="nav-text">{page.label}</span>
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </li>

          {/* Other Items */}
          <li className="mb-4">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                setPathname("/predictive-analytics")
              }}
              className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                isActiveLink("/predictive-analytics")
                  ? "bg-gray-200 text-[#123458] font-semibold"
                  : "text-gray-800 hover:bg-gray-200"
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>Predictive Analytics</span>
            </a>
          </li>

          <li className="mb-4">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                setPathname("/notifications")
              }}
              className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                isActiveLink("/notifications")
                  ? "bg-gray-200 text-[#123458] font-semibold"
                  : "text-gray-800 hover:bg-gray-200"
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"></path>
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>Notifications</span>
            </a>
          </li>

          <li className="mb-4">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                setPathname("/faq")
              }}
              className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                isActiveLink("/faq") ? "bg-gray-200 text-[#123458] font-semibold" : "text-gray-800 hover:bg-gray-200"
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5L6 11H5a1 1 0 100 2h1a1 1 0 00.867.5L10 13h1a1 1 0 000-2h-1a1 1 0 00-.867-.5z"
                  clipRule="evenodd"
                ></path>
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>FAQ</span>
            </a>
          </li>

          <li className="mb-4">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                setPathname("/cross-platform")
              }}
              className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                isActiveLink("/cross-platform")
                  ? "bg-gray-200 text-[#123458] font-semibold"
                  : "text-gray-800 hover:bg-gray-200"
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"></path>
                <path
                  fillRule="evenodd"
                  d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 0a1 1 0 00-1 1v5a1 1 0 102 0V6a1 1 0 00-1-1zm3 0a1 1 0 00-1 1v5a1 1 0 102 0V6a1 1 0 00-1-1zm3 0a1 1 0 00-1 1v5a1 1 0 102 0V6a1 1 0 00-1-1z"
                  clipRule="evenodd"
                ></path>
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>Cross-platform</span>
            </a>
          </li>

          <li className="mb-4">
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault()
                setPathname("/import")
              }}
              className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                isActiveLink("/import") ? "bg-gray-200 text-[#123458] font-semibold" : "text-gray-800 hover:bg-gray-200"
              } ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>Import</span>
            </a>
          </li>

          {/* Logout Menu Item */}
          <li className="mt-auto">
            <button
              onClick={logout}
              className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item text-gray-800 hover:bg-gray-200 w-full text-left ${isCollapsed ? 'justify-center' : ''}`}
            >
              <svg className={`flex-shrink-0 ${isCollapsed ? 'w-8 h-8' : 'w-5 h-5 mr-3'}`} fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M3 3a1 1 0 00-1 1v12a1 1 0 102 0V4a1 1 0 00-1-1zm10.293 9.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3a1 1 0 10-1.414 1.414L14.586 9H7a1 1 0 100 2h7.586l-1.293 1.293z"
                  clipRule="evenodd"
                />
              </svg>
              <span className={`${isCollapsed ? 'hidden' : 'nav-text'}`}>Logout</span>
            </button>
          </li>
        </ul>
      </nav>
    </aside>
  )
}
