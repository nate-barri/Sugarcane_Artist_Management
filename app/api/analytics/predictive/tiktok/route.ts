import { type NextRequest, NextResponse } from "next/server"

function generateHistoricalForecastData() {
  // Generate 6 months historical + 6 months forecast
  const data = []
  const now = new Date()

  // Historical data (past 6 months)
  for (let i = -26; i < 0; i++) {
    const date = new Date(now)
    date.setDate(date.getDate() + i * 7)
    const month = date.toLocaleDateString("en-US", { month: "short", year: "2-digit" })

    const baseValue = 150000 + Math.random() * 100000
    data.push({
      month,
      historical: Math.round(baseValue),
      forecast: null,
      lower: null,
      upper: null,
    })
  }

  // Forecast data (next 6 months)
  let lastHistorical = data[data.length - 1].historical || 150000
  for (let i = 1; i <= 26; i++) {
    const date = new Date(now)
    date.setDate(date.getDate() + i * 7)
    const month = date.toLocaleDateString("en-US", { month: "short", year: "2-digit" })

    const forecast = lastHistorical * (0.95 + Math.random() * 0.1)
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
      platform: "TikTok",
      historical_forecast: generateHistoricalForecastData(),
      timestamp: new Date().toISOString(),
    })
  } catch (error) {
    console.error("[v0] TikTok predictive error:", error)
    return NextResponse.json(
      {
        platform: "TikTok",
        historical_forecast: generateHistoricalForecastData(),
        timestamp: new Date().toISOString(),
      },
      { status: 200 },
    )
  }
}
