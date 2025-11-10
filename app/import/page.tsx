"use client"

import type React from "react"

import { useState } from "react"
import Sidebar from "@/components/sidebar"

interface ImportRecord {
  fileName: string
  type: string
  date: string
  status: "Success" | "Processing" | "Failed"
  records: number
  dataType?: string
}

export default function ImportDashboard() {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<string>("")
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [recentImports, setRecentImports] = useState<ImportRecord[]>([])

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setIsUploading(true)
    setUploadStatus("Uploading and processing...")
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append("file", file)

      const response = await new Promise<{ ok: boolean; data: any }>((resolve, reject) => {
        const xhr = new XMLHttpRequest()

        // Track upload progress
        xhr.upload.addEventListener("progress", (event) => {
          if (event.lengthComputable) {
            const percentComplete = (event.loaded / event.total) * 100
            setUploadProgress(Math.min(percentComplete, 90))
          }
        })

        xhr.addEventListener("load", () => {
          if (xhr.status === 200 || xhr.status === 201) {
            try {
              const data = JSON.parse(xhr.responseText)
              resolve({ ok: true, data })
            } catch {
              reject(new Error("Failed to parse response"))
            }
          } else {
            try {
              const data = JSON.parse(xhr.responseText)
              resolve({ ok: false, data })
            } catch {
              reject(new Error(`Upload failed with status ${xhr.status}`))
            }
          }
        })

        xhr.addEventListener("error", () => {
          reject(new Error("Upload failed"))
        })

        xhr.open("POST", "/api/upload")
        xhr.send(formData)
      })

      if (response.ok) {
        setUploadProgress(100)
        setUploadStatus(`Success! Processed ${response.data.records} records from ${response.data.dataType} data.`)

        const newImport: ImportRecord = {
          fileName: response.data.fileName,
          type: file.name.endsWith(".csv") ? "CSV" : "Excel",
          date: new Date().toISOString().split("T")[0],
          status: "Success",
          records: response.data.records,
          dataType: response.data.dataType,
        }
        setRecentImports([newImport, ...recentImports])
      } else {
        setUploadProgress(0)
        setUploadStatus(`Error: ${response.data.error}`)
      }
    } catch (error) {
      setUploadProgress(0)
      setUploadStatus(`Error: ${error instanceof Error ? error.message : "Upload failed"}`)
    } finally {
      setIsUploading(false)
      event.target.value = ""
    }
  }

  return (
    <div className="flex min-h-screen bg-[#D3D3D3] text-[#123458]">
      <Sidebar />

      {/* Main Content Area for Import Dashboard */}
      <main className="flex-1 p-8">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold text-[#123458]">Import Data</h1>
        </header>

        {uploadStatus && (
          <div
            className={`mb-6 p-4 rounded-lg ${
              uploadStatus.startsWith("Success")
                ? "bg-green-100 text-green-800"
                : uploadStatus.startsWith("Error")
                  ? "bg-red-100 text-red-800"
                  : "bg-blue-100 text-blue-800"
            }`}
          >
            {uploadStatus}
            {isUploading && uploadProgress > 0 && (
              <div className="mt-3 w-full bg-gray-300 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            )}
          </div>
        )}

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
              <p className="text-gray-600 mb-6">
                Upload your analytics data in CSV format. The system will automatically detect the platform (YouTube,
                TikTok, Spotify, Meta).
              </p>
            </div>
            <label className="w-full bg-[#3396D3] hover:bg-[#2A75A4] text-white font-semibold py-2 px-4 rounded-lg transition duration-200 text-center cursor-pointer">
              {isUploading ? "Processing..." : "Choose CSV File"}
              <input type="file" accept=".csv" onChange={handleFileUpload} disabled={isUploading} className="hidden" />
            </label>
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
              <p className="text-gray-600 mb-6">
                Upload your data using Excel spreadsheets (.xlsx). The system will automatically detect the platform.
              </p>
            </div>
            <label className="w-full bg-[#3396D3] hover:bg-[#2A75A4] text-white font-semibold py-2 px-4 rounded-lg transition duration-200 text-center cursor-pointer">
              {isUploading ? "Processing..." : "Choose Excel File"}
              <input type="file" accept=".xlsx" onChange={handleFileUpload} disabled={isUploading} className="hidden" />
            </label>
          </div>
        </section>

        {/* Import History Section */}
        {recentImports.length > 0 && (
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
                    <th className="pb-3 text-gray-600 font-medium">Data Type</th>
                  </tr>
                </thead>
                <tbody>
                  {recentImports.map((record, index) => (
                    <tr key={index} className="border-b border-gray-100">
                      <td className="py-3 text-gray-900">{record.fileName}</td>
                      <td className="py-3 text-gray-600">{record.type}</td>
                      <td className="py-3 text-gray-600">{record.date}</td>
                      <td className="py-3">
                        <span
                          className={`px-2 py-1 rounded-full text-sm ${
                            record.status === "Success"
                              ? "bg-green-100 text-green-800"
                              : record.status === "Processing"
                                ? "bg-yellow-100 text-yellow-800"
                                : "bg-red-100 text-red-800"
                          }`}
                        >
                          {record.status}
                        </span>
                      </td>
                      <td className="py-3 text-gray-600">{record.records.toLocaleString()}</td>
                      <td className="py-3 text-gray-600">{record.dataType || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        {/* Import Guidelines */}
        <section className="bg-white p-6 rounded-lg shadow-md mb-8">
          <h2 className="text-xl font-semibold mb-4">Import Guidelines</h2>
          <div className="space-y-3 text-gray-600">
            <p>• Ensure your data files are properly formatted with headers</p>
            <p>• Maximum file size: 50MB per upload</p>
            <p>• Supported formats: CSV and XLSX files</p>
            <p>• System automatically detects META (Facebook), YOUTUBE, TIKTOK, or SPOTIFY data</p>
            <p>• META data requires: post_id, page_name, publish_time columns</p>
            <p>• YOUTUBE data requires: Video title, Video ID, Video publish time columns</p>
            <p>• TIKTOK data requires: tiktok_video_id, content_link, video_title columns</p>
            <p>
              • SPOTIFY data requires: song, listeners, streams columns (optional: saves, release_date, date, followers)
            </p>
            <p>• Data will be processed and loaded into the database immediately</p>
          </div>
        </section>
      </main>
    </div>
  )
}
