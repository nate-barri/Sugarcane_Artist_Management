import { NextResponse } from "next/server"

export async function GET() {
  // Tiktok1.png: Total Channel Views - Last 6 Months + 6-Month Forecast
  const totalChannelViewsForecast = {
    title: "Total Channel Views: Last 6 Months + 6-Month Forecast",
    subtitle: "MAPE (15.8%) confidence range",
    data: [
      { date: "2025-05", historical: 50000, forecast: null, upper: null, lower: null },
      { date: "2025-06", historical: 100000, forecast: null, upper: null, lower: null },
      { date: "2025-07", historical: 150000, forecast: null, upper: null, lower: null },
      { date: "2025-08", historical: 400000, forecast: null, upper: null, lower: null },
      { date: "2025-09", historical: 650000, forecast: null, upper: null, lower: null },
      { date: "2025-10", historical: 1500000, forecast: null, upper: null, lower: null },
      { date: "2025-11", historical: null, forecast: 2000000, upper: 2500000, lower: 1500000 },
      { date: "2025-12", historical: null, forecast: 2800000, upper: 3500000, lower: 2100000 },
      { date: "2026-01", historical: null, forecast: 3800000, upper: 4750000, lower: 2850000 },
      { date: "2026-02", historical: null, forecast: 5000000, upper: 6250000, lower: 3750000 },
      { date: "2026-03", historical: null, forecast: 6500000, upper: 8125000, lower: 4875000 },
      { date: "2026-04", historical: null, forecast: 7800000, upper: 9750000, lower: 5850000 },
    ],
  }

  // Tiktok2.png: Predicted vs Actual Engagement Rate scatter plot
  const engagementScatterPlot = {
    title: "Predicted vs Actual (R²=0.303)",
    subtitle: "±3% Zone | MAE: 3.13% | RMSE: 4.15%",
    metrics: {
      r2: 0.303,
      mae: 3.13,
      rmse: 4.15,
      n: 77,
    },
    data: [
      { actual: 0.5, predicted: 1, color: "#16a34a" }, // green for within zone
      { actual: 1, predicted: 2, color: "#16a34a" },
      { actual: 2, predicted: 3, color: "#16a34a" },
      { actual: 3, predicted: 3.5, color: "#16a34a" },
      { actual: 4, predicted: 4.5, color: "#eab308" }, // yellow for warning zone
      { actual: 5, predicted: 5, color: "#16a34a" },
      { actual: 6, predicted: 6.5, color: "#16a34a" },
      { actual: 7, predicted: 7, color: "#16a34a" },
      { actual: 8, predicted: 8.5, color: "#eab308" },
      { actual: 9, predicted: 8, color: "#eab308" },
      { actual: 10, predicted: 10.5, color: "#16a34a" },
      { actual: 11, predicted: 11, color: "#16a34a" },
      { actual: 12, predicted: 12.5, color: "#16a34a" },
      { actual: 13, predicted: 13, color: "#eab308" },
      { actual: 14, predicted: 12, color: "#ea580c" }, // orange for outside zone
      { actual: 15, predicted: 15.5, color: "#16a34a" },
      { actual: 16, predicted: 14, color: "#ea580c" },
      { actual: 18, predicted: 20, color: "#dc2626" }, // red for large errors
      { actual: 19, predicted: 17, color: "#ea580c" },
      { actual: 20, predicted: 22, color: "#ea580c" },
      { actual: 21, predicted: 19, color: "#dc2626" },
      { actual: 22, predicted: 24, color: "#dc2626" },
      { actual: 23, predicted: 21, color: "#dc2626" },
    ],
  }

  // Tiktok3.png: Cumulative Views with 6-Month Forecast
  const cumulativeChannelViewsForecast = {
    title: "Total Channel Views: Last 6 Months + 6-Month Forecast",
    subtitle: "±MAPE (15.0%) confidence range",
    data: [
      { date: "2025-05", cumulative: 50000, forecast: null, upper: null, lower: null },
      { date: "2025-06", cumulative: 150000, forecast: null, upper: null, lower: null },
      { date: "2025-07", cumulative: 300000, forecast: null, upper: null, lower: null },
      { date: "2025-08", cumulative: 700000, forecast: null, upper: null, lower: null },
      { date: "2025-09", cumulative: 1350000, forecast: null, upper: null, lower: null },
      { date: "2025-10", cumulative: 2850000, forecast: null, upper: null, lower: null },
      { date: "2025-11", forecast: 4850000, upper: 5577500, lower: 4122500 },
      { date: "2025-12", forecast: 7650000, upper: 8797500, lower: 6502500 },
      { date: "2026-01", forecast: 11450000, upper: 13167500, lower: 9732500 },
      { date: "2026-02", forecast: 16450000, upper: 18917500, lower: 13982500 },
      { date: "2026-03", forecast: 22950000, upper: 26392500, lower: 19507500 },
      { date: "2026-04", forecast: 30750000, upper: 35362500, lower: 26137500 },
    ],
  }

  return NextResponse.json({
    totalChannelViewsForecast,
    engagementScatterPlot,
    cumulativeChannelViewsForecast,
  })
}
