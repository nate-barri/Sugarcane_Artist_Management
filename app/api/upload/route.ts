import { type NextRequest, NextResponse } from "next/server"
import { writeFile, unlink } from "fs/promises"
import { join } from "path"
import { spawn } from "child_process"
import { tmpdir } from "os"

// Helper function to detect data type from CSV headers
async function detectDataType(filePath: string): Promise<"META" | "YOUTUBE" | "TIKTOK" | "SPOTIFY" | "UNKNOWN"> {
  const fs = require("fs").promises
  const content = await fs.readFile(filePath, "utf-8")
  const firstLine = content.split("\n")[0].toLowerCase()

  if (firstLine.includes("tiktok_video_id") || firstLine.includes("content_link")) {
    return "TIKTOK"
  }

  // Check for Facebook/Meta specific columns
  if (firstLine.includes("post_id") || firstLine.includes("page_name") || firstLine.includes("permalink")) {
    return "META"
  }

  // Check for YouTube specific columns
  if (firstLine.includes("video title") || firstLine.includes("video id") || firstLine.includes("video publish time")) {
    return "YOUTUBE"
  }

  if (
    firstLine.includes("date") &&
    firstLine.includes("listeners") &&
    firstLine.includes("streams") &&
    firstLine.includes("followers")
  ) {
    return "SPOTIFY"
  }

  return "UNKNOWN"
}

// Helper function to find available Python command on Windows/Mac/Linux
async function findPythonCommand(): Promise<string> {
  const commands = ["py", "python", "python3"]

  for (const cmd of commands) {
    try {
      const result = await new Promise<boolean>((resolve) => {
        const testProcess = spawn(cmd, ["--version"])
        testProcess.on("error", () => resolve(false))
        testProcess.on("close", (code) => resolve(code === 0))
      })

      if (result) {
        console.log("[v0] Found Python command:", cmd)
        return cmd
      }
    } catch {
      continue
    }
  }

  throw new Error("Python not found. Please install Python and ensure it's in your PATH.")
}

// Helper function to execute Python script
function executePythonScript(
  scriptName: string,
  filePath: string,
  pythonCmd: string,
): Promise<{ success: boolean; records: number; error?: string }> {
  return new Promise((resolve) => {
    const scriptPath = join(process.cwd(), "scripts", scriptName)

    const env = {
      ...process.env,
      DATABASE_URL: process.env.DATABASE_URL,
    }

    const pythonProcess = spawn(pythonCmd, [scriptPath, filePath], {
      env,
      shell: true,
    })

    let stdout = ""
    let stderr = ""

    pythonProcess.stdout.on("data", (data) => {
      stdout += data.toString()
      console.log("[v0] Python stdout:", data.toString())
    })

    pythonProcess.stderr.on("data", (data) => {
      stderr += data.toString()
      console.error("[v0] Python stderr:", data.toString())
    })

    pythonProcess.on("close", (code) => {
      console.log("[v0] Python process exited with code:", code)

      if (code === 0) {
        // Extract record count from output
        const recordMatch = stdout.match(/RECORDS: (\d+)/)
        const records = recordMatch ? Number.parseInt(recordMatch[1]) : 0

        resolve({ success: true, records })
      } else {
        resolve({ success: false, records: 0, error: stderr || "Unknown error occurred" })
      }
    })

    pythonProcess.on("error", (error) => {
      console.error("[v0] Failed to start Python process:", error)
      resolve({ success: false, records: 0, error: error.message })
    })
  })
}

// Helper function to ensure Python dependencies are installed
function installPythonDependencies(pythonCmd: string): Promise<boolean> {
  return new Promise((resolve) => {
    const requiredPackages = ["pandas", "psycopg2-binary", "emoji", "numpy"]

    const installProcess = spawn(pythonCmd, ["-m", "pip", "install", ...requiredPackages], {
      shell: true,
      stdio: "pipe", // Suppress output
    })

    installProcess.on("close", (code) => {
      if (code === 0) {
        console.log("[v0] Python dependencies installed successfully")
        resolve(true)
      } else {
        console.log("[v0] Warning: Could not verify all dependencies, attempting to run anyway")
        resolve(true) // Still proceed - dependencies may already be installed
      }
    })

    installProcess.on("error", () => {
      console.log("[v0] Warning: Could not install dependencies, attempting to run anyway")
      resolve(true) // Still proceed
    })
  })
}

export async function POST(request: NextRequest) {
  try {
    const pythonCmd = await findPythonCommand()

    await installPythonDependencies(pythonCmd)

    const formData = await request.formData()
    const file = formData.get("file") as File

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 })
    }

    // Validate file type
    const fileName = file.name.toLowerCase()
    if (!fileName.endsWith(".csv") && !fileName.endsWith(".xlsx")) {
      return NextResponse.json({ error: "Only CSV and XLSX files are supported" }, { status: 400 })
    }

    const tempDir = tmpdir()
    const tempFilePath = join(tempDir, `upload_${Date.now()}_${file.name}`)

    // Save file temporarily
    const bytes = await file.arrayBuffer()
    const buffer = Buffer.from(bytes)
    await writeFile(tempFilePath, buffer)

    console.log("[v0] File saved to:", tempFilePath)

    // Detect data type
    const dataType = await detectDataType(tempFilePath)
    console.log("[v0] Detected data type:", dataType)

    if (dataType === "UNKNOWN") {
      await unlink(tempFilePath)
      return NextResponse.json(
        {
          error:
            "Unable to detect data type. Please ensure your CSV has the correct headers for META, YOUTUBE, TIKTOK, or SPOTIFY data.",
        },
        { status: 400 },
      )
    }

    const scriptMap: Record<string, string> = {
      META: "facebook_etl.py",
      YOUTUBE: "youtube_etl.py",
      TIKTOK: "tiktok_etl.py",
      SPOTIFY: "spotify_etl.py",
    }
    const scriptName = scriptMap[dataType]
    const result = await executePythonScript(scriptName, tempFilePath, pythonCmd)

    // Clean up temp file
    await unlink(tempFilePath)

    if (result.success) {
      return NextResponse.json({
        success: true,
        message: `Successfully processed ${result.records} records`,
        dataType,
        records: result.records,
        fileName: file.name,
        timestamp: new Date().toISOString(),
      })
    } else {
      return NextResponse.json(
        {
          error: result.error || "ETL processing failed",
          dataType,
        },
        { status: 500 },
      )
    }
  } catch (error) {
    console.error("[v0] Upload error:", error)
    return NextResponse.json(
      {
        error: error instanceof Error ? error.message : "Upload failed",
      },
      { status: 500 },
    )
  }
}
