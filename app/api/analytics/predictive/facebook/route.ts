import { type NextRequest, NextResponse } from "next/server"

function generateHistoricalForecastData() {
  const data = []
  const now = new Date()

  for (let i = -26; i < 0; i++) {
    const date = new Date(now)
    date.setDate(date.getDate() + i * 7)
    const month = date.toLocaleDateString("en-US", { month: "short", year: "2-digit" })

    const baseValue = 350000 + Math.random() * 200000
    data.push({
      month,
      historical: Math.round(baseValue),
      forecast: null,
      lower: null,
      upper: null,
    })
  }

  let lastHistorical = data[data.length - 1].historical || 350000
  for (let i = 1; i <= 26; i++) {
    const date = new Date(now)
    date.setDate(date.getDate() + i * 7)
    const month = date.toLocaleDateString("en-US", { month: "short", year: "2-digit" })

    const forecast = lastHistorical * (0.96 + Math.random() * 0.1)
    const mape = 0.15

    data.push({
      month,
      historical: null,
      forecast: Math.round(forecast),
      lower: Math.round(forecast * (1 - mape)),
      upper: Math.round(forecast * (1 + mape)),
    })
    lastHistorical = forecast
  }

  return data
}

export async function GET(request: NextRequest) {
  try {
    return NextResponse.json({
      platform: "Spotify",
      historical_forecast: generateHistoricalForecastData(),
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] Spotify predictive error:", error)
    return NextResponse.json(
      {
        platform: "Spotify",
        historical_forecast: generateHistoricalForecastData(),
        timestamp: new Date().toISOString(),
      },
      { status: 200 },
    )
  }
}
