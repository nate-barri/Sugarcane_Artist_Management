import { NextResponse } from "next/server"

export async function GET() {
  // Youtube.png: Historical Cumulative Views - Actual vs Model + 6-Month Forecast
  const cumulativeViewsChart = {
    title: "Historical Cumulative Views: Actual vs Model Estimate",
    subtitle: "Full historical timeline with model validation",
    historicalData: [
      { date: "2020-01", cumulative_actual: 100000, cumulative_model: 50000 },
      { date: "2020-06", cumulative_actual: 500000, cumulative_model: 300000 },
      { date: "2020-12", cumulative_actual: 2500000, cumulative_model: 1800000 },
      { date: "2021-06", cumulative_actual: 5800000, cumulative_model: 4200000 },
      { date: "2021-12", cumulative_actual: 10200000, cumulative_model: 8000000 },
      { date: "2022-06", cumulative_actual: 15000000, cumulative_model: 12000000 },
      { date: "2022-12", cumulative_actual: 25000000, cumulative_model: 20000000 },
      { date: "2023-06", cumulative_actual: 50000000, cumulative_model: 45000000 },
      { date: "2023-12", cumulative_actual: 80000000, cumulative_model: 48000000 },
      { date: "2024-06", cumulative_actual: 85000000, cumulative_model: 50000000 },
      { date: "2024-12", cumulative_actual: 87500000, cumulative_model: 51000000 },
      { date: "2025-07", cumulative_actual: 87500000, cumulative_model: 51000000 },
    ],
    sixMonthForecast: [
      { date: "2025-07", historical_last_6m: 87500000, forecast: 90000000, upper: 98000000, lower: 82000000 },
      { date: "2025-08", historical_last_6m: 87500000, forecast: 91000000, upper: 99500000, lower: 82500000 },
      { date: "2025-09", historical_last_6m: 87500000, forecast: 92000000, upper: 101000000, lower: 83000000 },
      { date: "2025-10", historical_last_6m: 87500000, forecast: 92500000, upper: 101500000, lower: 83500000 },
      { date: "2025-11", historical_last_6m: 87500000, forecast: 93500000, upper: 102500000, lower: 84500000 },
      { date: "2025-12", historical_last_6m: 87500000, forecast: 94500000, upper: 103500000, lower: 85500000 },
    ],
  }

  // Youtube1.png: Total Catalog Views - Backcast + Forecast with confidence
  const catalogViewsForecast = {
    title: "Total Catalog Views: Historical (Backcast) + Forecast (Baseline 6mo)",
    subtitle: "Confidence range (70%-130%)",
    data: [
      { month: "-6m", historical: 81000000, forecast: null, upper: null, lower: null },
      { month: "-5m", historical: 82500000, forecast: null, upper: null, lower: null },
      { month: "-4m", historical: 84000000, forecast: null, upper: null, lower: null },
      { month: "-3m", historical: 85000000, forecast: null, upper: null, lower: null },
      { month: "-2m", historical: 86500000, forecast: null, upper: null, lower: null },
      { month: "-1m", historical: 87394937, forecast: null, upper: null, lower: null },
      { month: "Now", historical: 87394937, forecast: 87394937, upper: 113812818, lower: 61176456 },
      { month: "+1m", historical: null, forecast: 89000000, upper: 115700000, lower: 62300000 },
      { month: "+2m", historical: null, forecast: 90500000, upper: 117650000, lower: 63350000 },
      { month: "+3m", historical: null, forecast: 92000000, upper: 119600000, lower: 64400000 },
      { month: "+4m", historical: null, forecast: 93071402, upper: 121092822, lower: 65150000 },
      { month: "+5m", historical: null, forecast: 93071402, upper: 121092822, lower: 65150000 },
      { month: "+6m", historical: null, forecast: 93071402, upper: 121092822, lower: 65150000 },
    ],
  }

  return NextResponse.json({
    cumulativeViewsChart,
    catalogViewsForecast,
  })
}
