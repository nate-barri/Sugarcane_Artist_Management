"use client"

import type React from "react"

import { useState, useEffect } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useAuth } from "@/components/auth-provider"

export default function Sidebar() {
  const [isCollapsed, setIsCollapsed] = useState(false)
  const [isDashboardOpen, setIsDashboardOpen] = useState(true)
  const pathname = usePathname()
  const { user, logout } = useAuth()

  // Load sidebar state from localStorage
  useEffect(() => {
    const storedState = localStorage.getItem("sidebarCollapsedState")
    if (storedState === "true") {
      setIsCollapsed(true)
      setIsDashboardOpen(false)
    }
  }, [])

  // Save sidebar state to localStorage
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
    if ((e.target as HTMLElement).closest("#dashboard-arrow")) {
      e.preventDefault()
      if (!isCollapsed) {
        setIsDashboardOpen(!isDashboardOpen)
      }
    }
  }

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

  const isDashboardActive =
    pathname === "/" || dashboardPages.some((page) => pathname.includes(page.href))

  return (
    <aside
      className={`sidebar bg-[#0f2946] shadow-lg px-6 pt-6 flex flex-col rounded-r-lg transition-all duration-300 ease-in-out h-screen justify-between ${
        isCollapsed ? "w-20" : "w-64"
      }`}
    >
      {/* Top Section (Logo + Nav) */}
      <div>
        <div className="mb-8 flex items-center justify-between logo-container">
          {/* Hamburger Icon */}
          <button
            onClick={toggleSidebar}
            className="text-white focus:outline-none p-2 rounded-lg hover:bg-gray-700"
          >
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
            <img
              src="/SUGARCANE-LOGO.png"
              alt="Sugarcane Logo"
              className="h-8 w-auto rounded logo-text"
            />
          )}
        </div>

        {/* Navigation */}
        <nav className="flex-1">
          <ul>
            {/* Dashboard */}
            <li className="mb-4">
              <Link
                href="/"
                onClick={toggleDashboard}
                className={`flex items-center justify-between p-3 rounded-lg font-semibold shadow-sm cursor-pointer nav-item transition-colors duration-200 ${
                  isDashboardActive
                    ? "bg-[#123458] text-white"
                    : "text-white hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center">
                  <svg
                    className={`w-5 h-5 ${!isCollapsed ? "mr-3" : "mx-auto"}`}
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z"></path>
                  </svg>
                  {!isCollapsed && <span className="nav-text">Dashboard</span>}
                </div>

                {/* Dropdown Arrow */}
                {!isCollapsed && (
                  <svg
                    id="dashboard-arrow"
                    className={`w-4 h-4 ml-2 transition-transform duration-200 ${
                      isDashboardOpen ? "rotate-90" : ""
                    }`}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                  </svg>
                )}
              </Link>

              {!isCollapsed && (
                <ul
                  className={`submenu pl-8 pt-2 transition-all duration-200 ${
                    isDashboardOpen ? "active" : "hidden"
                  }`}
                >
                  {dashboardPages.map((page) => (
                    <li key={page.href} className="mb-2">
                      <Link
                        href={page.href}
                        className={`flex items-center p-2 rounded-lg transition-colors duration-200 nav-item ${
                          isActiveLink(page.href)
                            ? "bg-gray-600 text-white font-semibold"
                            : "text-gray-300 hover:bg-gray-700 hover:text-white"
                        }`}
                      >
                        <span className="nav-text">{page.label}</span>
                      </Link>
                    </li>
                  ))}
                </ul>
              )}
            </li>

            {/* Other Items */}
            <li className="mb-4">
              <Link
                href="/predictive-analytics"
                className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                  isActiveLink("/predictive-analytics")
                    ? "bg-[#123458] text-white font-semibold"
                    : "text-white hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center">
                <svg
                  className={`w-5 h-5 ${!isCollapsed ? "mr-3" : "mx-auto"}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                {!isCollapsed && <span className="nav-text">Predictive Analytics</span>}
                </div>
              </Link>
            </li>

            <li className="mb-4">
              <Link
                href="/notifications"
                className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                  isActiveLink("/notifications")
                    ? "bg-[#123458] text-white font-semibold"
                    : "text-white hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center">
                <svg
                  className={`w-5 h-5 ${!isCollapsed ? "mr-3" : "mx-auto"}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M10 2a6 6 0 00-6 6v3.586l-.707.707A1 1 0 004 14h12a1 1 0 00.707-1.707L16 11.586V8a6 6 0 00-6-6zM10 18a3 3 0 01-3-3h6a3 3 0 01-3 3z"></path>
                </svg>
                {!isCollapsed && <span className="nav-text">Notifications</span>}
                </div>
              </Link>
            </li>

            <li className="mb-4">
              <Link
                href="/faq"
                className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                  isActiveLink("/faq")
                    ? "bg-[#123458] text-white font-semibold"
                    : "text-white hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center">
                <svg
                  className={`w-5 h-5 ${!isCollapsed ? "mr-3" : "mx-auto"}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-8-3a1 1 0 00-.867.5L6 11H5a1 1 0 100 2h1a1 1 0 00.867.5L10 13h1a1 1 0 000-2h-1a1 1 0 00-.867-.5z"
                    clipRule="evenodd"
                  ></path>
                </svg>
                {!isCollapsed && <span className="nav-text">FAQ</span>}
                </div>
              </Link>
            </li>

            <li className="mb-4">
              <Link
                href="/cross-platform"
                className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                  isActiveLink("/cross-platform")
                    ? "bg-[#123458] text-white font-semibold"
                    : "text-white hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center">
                <svg
                  className={`w-5 h-5 ${!isCollapsed ? "mr-3" : "mx-auto"}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"></path>
                  <path
                    fillRule="evenodd"
                    d="M4 5a2 2 0 012-2 3 3 0 003 3h2a3 3 0 003-3 2 2 0 012 2v11a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 0a1 1 0 00-1 1v5a1 1 0 102 0V6a1 1 0 00-1-1zm3 0a1 1 0 00-1 1v5a1 1 0 102 0V6a1 1 0 00-1-1zm3 0a1 1 0 00-1 1v5a1 1 0 102 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  ></path>
                </svg>
                {!isCollapsed && <span className="nav-text">Cross-platform</span>}
                </div>
              </Link>
            </li>

            <li className="mb-4">
              <Link 
                href="/import"
                className={`flex items-center p-3 rounded-lg transition-colors duration-200 nav-item ${
                  isActiveLink("/import")
                    ? "bg-[#123458] text-white font-semibold"
                    : "text-white hover:bg-gray-700"
                }`}
              >
                <div className="flex items-center">
                <svg
                  className={`w-5 h-5 ${!isCollapsed ? "mr-3" : "mx-auto"}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
                {!isCollapsed && <span className="nav-text">Import</span>}
                </div>
              </Link>
            </li>
          </ul>
        </nav>
      </div>

      {/* Logout Button at Bottom */}
      <div className="mb-4">
        <button
          onClick={logout}
          className="flex items-center w-full p-3 rounded-lg text-white hover:bg-gray-700 transition-colors duration-200"
        >
          <svg
            className="w-5 h-5"
            fill="currentColor"
            viewBox="0 0 20 20"
          >
            <path
              fillRule="evenodd"
              d="M3 3a1 1 0 00-1 1v12a1 1 0 102 0V4a1 1 0 00-1-1zm10.293 9.293a1 1 0 001.414 1.414l3-3a1 1 0 000-1.414l-3-3a1 1 0 10-1.414 1.414L14.586 9H7a1 1 0 100 2h7.586l-1.293 1.293z"
              clipRule="evenodd"
            />
          </svg>
          {!isCollapsed && <span className="ml-2">Log out</span>}
        </button>
      </div>
    </aside>
  )
}
