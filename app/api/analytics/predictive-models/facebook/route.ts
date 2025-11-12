import { NextResponse } from "next/server"

// These values are extracted from facebook_historical_forecast.py model outputs

export async function GET() {
  // Meta.png: Backtest Forecast - Actual vs Predicted Reach (3-Month Rolling)
  const backtestForecast = {
    title: "Backtest Forecast: Actual vs Predicted Reach (3-Month Rolling)",
    subtitle: "Model validation on test set",
    data: [
      { x: 0.0, actual: 0.35, predicted: 0.4 },
      { x: 0.5, actual: 0.6, predicted: 0.9 },
      { x: 1.0, actual: 0.9, predicted: 1.5 },
      { x: 1.5, actual: 1.1, predicted: 1.4 },
      { x: 2.0, actual: 1.3, predicted: 1.05 },
      { x: 2.5, actual: 0.3, predicted: 0.35 },
      { x: 3.0, actual: 0.25, predicted: 0.3 },
      { x: 3.5, actual: 0.15, predicted: 0.2 },
      { x: 4.0, actual: 0.1, predicted: 0.12 },
    ],
    xAxis: "Backtest Split (Time Order)",
    yAxis: "Monthly Reach",
    lines: [
      { key: "actual", name: "Actual Reach", color: "#2563eb", lineStyle: "solid" },
      { key: "predicted", name: "Predicted Reach", color: "#ea580c", lineStyle: "solid" },
    ],
  }

  // Meta2.png: Facebook Reach - Last 6 Months + Next 6 Months Forecast
  const facebookReachForecast = {
    title: "Facebook Reach: Last 6 Months (Actual) + Next 6 Months (Forecast)",
    subtitle: "Monthly reach projections",
    data: [
      { date: "Mar 2025", historical: 215000, forecast: null, upper: null, lower: null },
      { date: "Apr 2025", historical: 190000, forecast: null, upper: null, lower: null },
      { date: "May 2025", historical: 130000, forecast: null, upper: null, lower: null },
      { date: "Jun 2025", historical: 175000, forecast: null, upper: null, lower: null },
      { date: "Jul 2025", historical: 365000, forecast: null, upper: null, lower: null },
      { date: "Aug 2025", historical: 300000, forecast: null, upper: null, lower: null },
      { date: "Sep 2025", historical: null, forecast: 140000, upper: 170000, lower: 110000 },
      { date: "Oct 2025", historical: null, forecast: 135000, upper: 165000, lower: 105000 },
      { date: "Nov 2025", historical: null, forecast: 135000, upper: 165000, lower: 105000 },
      { date: "Dec 2025", historical: null, forecast: 130000, upper: 160000, lower: 100000 },
      { date: "Jan 2026", historical: null, forecast: 125000, upper: 155000, lower: 95000 },
      { date: "Feb 2026", historical: null, forecast: 125000, upper: 155000, lower: 95000 },
    ],
    confidence: "6-month forecast with confidence range",
  }

  // Meta3.png: Existing Posts Reach Forecast (Next 6 Months)
  const existingPostsReachForecast = {
    title: "Existing Posts Reach Forecast (Next 6 Months)",
    subtitle: "Historical reach (6 mo) + Forecasted reach with confidence interval",
    data: [
      { date: "2025-03", historical: 165000, forecast: null, upper: null, lower: null },
      { date: "2025-04", historical: 110000, forecast: null, upper: null, lower: null },
      { date: "2025-05", historical: 120000, forecast: null, upper: null, lower: null },
      { date: "2025-06", historical: 40000, forecast: null, upper: null, lower: null },
      { date: "2025-07", historical: 175000, forecast: null, upper: null, lower: null },
      { date: "2025-08", historical: 30000, forecast: null, upper: null, lower: null },
      { date: "2025-09", historical: null, forecast: 40000, upper: 65000, lower: 20000 },
      { date: "2025-10", historical: null, forecast: 40000, upper: 65000, lower: 20000 },
      { date: "2025-11", historical: null, forecast: 40000, upper: 65000, lower: 20000 },
      { date: "2025-12", historical: null, forecast: 40000, upper: 65000, lower: 20000 },
      { date: "2026-01", historical: null, forecast: 40000, upper: 65000, lower: 20000 },
    ],
  }

  return NextResponse.json({
    backtestForecast,
    facebookReachForecast,
    existingPostsReachForecast,
  })
}
