import { NextResponse } from "next/server"

export async function GET() {
  // Chart 1: Catalog Views
  const catalogViews = generateCatalogViewsData()

  // Chart 2: Facebook Reach
  const facebookReach = generateFacebookReachData()

  // Chart 3: Channel Views Cumulative
  const channelViewsCumulative = generateChannelViewsCumulativeData()

  // Chart 4: Channel Views Alternative
  const channelViewsAlt = generateChannelViewsAltData()

  // Chart 5: Predicted vs Actual
  const predictedVsActual = generatePredictedVsActualData()

  // Chart 6: Historical Cumulative
  const historicalCumulative = generateHistoricalCumulativeData()

  // Chart 7: Backtest Forecast
  const backtestForecast = generateBacktestForecastData()

  // Chart 8: Existing Posts Reach
  const existingPostsReach = generateExistingPostsReachData()

  return NextResponse.json({
    catalogViews,
    facebookReach,
    channelViewsCumulative,
    channelViewsAlt,
    predictedVsActual,
    historicalCumulative,
    backtestForecast,
    existingPostsReach,
  })
}

function generateCatalogViewsData() {
  const months = ["6m", "5m", "4m", "3m", "2m", "1m", "Now", "+1m", "+2m", "+3m", "+4m", "+5m", "+6m"]
  const baseValue = 87394937
  const growth = 1.02

  return months.map((m, i) => {
    const value = baseValue * Math.pow(growth, i - 6)
    return {
      month: m,
      historical: i <= 6 ? value : null,
      forecast: i >= 6 ? value : null,
      confidenceLower: i >= 6 ? value * 0.7 : null,
      confidenceUpper: i >= 6 ? value * 1.3 : null,
    }
  })
}

function generateFacebookReachData() {
  const months = [
    "Mar 2025",
    "Apr 2025",
    "May 2025",
    "Jun 2025",
    "Jul 2025",
    "Aug 2025",
    "Sep 2025",
    "Oct 2025",
    "Nov 2025",
    "Dec 2025",
    "Jan 2026",
    "Feb 2026",
  ]
  const baseReach = 210000

  return months.map((m, i) => {
    const actual = i < 6 ? baseReach + Math.sin(i) * 50000 : null
    const forecast = i >= 5 ? baseReach * 0.6 + Math.sin(i * 0.5) * 30000 : null
    return {
      month: m,
      actual,
      forecast,
      lower: forecast ? forecast * 0.8 : null,
      upper: forecast ? forecast * 1.2 : null,
    }
  })
}

function generateChannelViewsCumulativeData() {
  const months = ["May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar"]
  return months.map((m, i) => ({
    month: m,
    historical: i < 5 ? 100000 + i * 50000 : null,
    forecast: i >= 4 ? 1500000 + (i - 4) * 300000 : null,
  }))
}

function generateChannelViewsAltData() {
  const months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"]
  return months.map((m, i) => {
    const base = 400000
    return {
      month: m,
      historical: i < 6 ? base + Math.random() * 100000 : null,
      forecast: i >= 5 ? base : null,
      lower: i >= 5 ? base * 0.85 : null,
      upper: i >= 5 ? base * 1.15 : null,
    }
  })
}

function generatePredictedVsActualData() {
  const points = []
  for (let i = 0; i < 77; i++) {
    const actual = Math.random() * 25
    const predicted = actual + (Math.random() - 0.5) * 10
    points.push({
      actual: Number.parseFloat(actual.toFixed(2)),
      predicted: Number.parseFloat(Math.max(0, predicted).toFixed(2)),
    })
  }
  return points
}

function generateHistoricalCumulativeData() {
  const dates = []
  const startDate = new Date(2019, 11, 1)

  for (let i = 0; i < 60; i++) {
    const date = new Date(startDate)
    date.setMonth(date.getMonth() + i)
    const actual = Math.pow(1.03, i) * 5000000
    const estimate = Math.pow(1.02, i) * 4000000
    dates.push({
      date: date.toISOString().split("T")[0],
      actual: Math.round(actual),
      estimate: Math.round(estimate),
    })
  }
  return dates
}

function generateBacktestForecastData() {
  return [
    { split: 0, actual: 300000, predicted: 400000 },
    { split: 0.5, actual: 900000, predicted: 600000 },
    { split: 1, actual: 1000000, predicted: 900000 },
    { split: 1.5, actual: 1200000, predicted: 1050000 },
    { split: 2, actual: 1300000, predicted: 1000000 },
    { split: 2.5, actual: 300000, predicted: 300000 },
    { split: 3, actual: 100000, predicted: 100000 },
    { split: 3.5, actual: 80000, predicted: 150000 },
    { split: 4, actual: 100000, predicted: 120000 },
  ]
}

function generateExistingPostsReachData() {
  const months = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"]
  return months.map((m, i) => {
    const base = 100000
    const variation = Math.sin(i * 0.5) * 30000
    return {
      month: m,
      historical: i < 6 ? base + variation : null,
      forecast: i >= 5 ? base * 0.4 : null,
      lower: i >= 5 ? base * 0.4 * 0.8 : null,
      upper: i >= 5 ? base * 0.4 * 1.2 : null,
    }
  })
}
