import { NextResponse } from "next/server"

export async function GET() {
  try {
    console.log("[v0] Generating predictive analytics data...")

    function generateMetaBacktest() {
      const dates = []
      for (let i = 0; i < 5; i++) {
        const date = new Date(2023, 0, 1 + i * 90)
        dates.push(date.toISOString().slice(0, 7))
      }
      const actual = [0.3, 0.6, 0.9, 1.2, 0.8]
      const predicted = [0.4, 1.0, 1.5, 1.0, 0.2]

      return {
        title: "Backtest Forecast: Actual vs Predicted Reach (3-Month Rolling)",
        description: "validation test",
        label: "Backtesting",
        data: dates.map((d, i) => ({
          date: d,
          actual: actual[i] * 1e6,
          predicted: predicted[i] * 1e6,
        })),
        metrics: { mape: 20.74, r2: 0.943, mase: 0.146, mae: 15499 },
      }
    }

    function generateMetaReach6m() {
      const dates = []
      const baseDate = new Date(2025, 2, 1)
      for (let i = 0; i < 12; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates.push(date.toLocaleDateString("en-US", { month: "short", year: "numeric" }))
      }

      const historical = [210000, 190000, 130000, 180000, 370000, 300000]
      const forecast = [130000, 130000, 130000, 130000, 130000, 130000]
      const forecastUpper = [160000, 160000, 160000, 160000, 160000, 160000]
      const forecastLower = [100000, 100000, 100000, 100000, 100000, 100000]

      const dataPoints: any[] = []
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[i],
          historical: historical[i],
          forecast: null,
          upper: null,
          lower: null,
        })
      }
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[6 + i],
          historical: null,
          forecast: forecast[i],
          upper: forecastUpper[i],
          lower: forecastLower[i],
        })
      }

      return {
        title: "Facebook Reach: Last 6 Months (Actual) + Next 6 Months (Forecast)",
        description: "with confidence bands",
        videoType: "New Videos",
        data: dataPoints,
        metrics: { mape: 20.74, r2: 0.943, mase: 0.146, mae: 145000 },
      }
    }

    function generateMetaExistingPosts() {
      const dates = []
      const baseDate = new Date(2025, 2, 1)
      for (let i = 0; i < 12; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates.push(date.toLocaleDateString("en-US", { month: "short", year: "numeric" }))
      }

      const historical = [160000, 110000, 120000, 175000, 40000]
      const forecast = [40000, 40000, 40000, 40000, 40000, 40000]
      const forecastUpper = [70000, 70000, 70000, 70000, 70000, 70000]
      const forecastLower = [20000, 20000, 20000, 20000, 20000, 20000]

      const dataPoints: any[] = []
      for (let i = 0; i < 5; i++) {
        dataPoints.push({
          date: dates[i],
          historical: historical[i],
          forecast: null,
          upper: null,
          lower: null,
        })
      }
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[5 + i],
          historical: null,
          forecast: forecast[i],
          upper: forecastUpper[i],
          lower: forecastLower[i],
        })
      }

      return {
        title: "Existing Posts Reach Forecast (Next 6 Months)",
        description: "with historical 6mo + forecast",
        videoType: "Existing Videos",
        data: dataPoints,
        metrics: { mape: 20.74, r2: 0.943, mase: 0.146, mae: 15499 },
      }
    }

    function generateYoutubeCumulative() {
      const dates1 = []
      for (let i = 0; i < 8; i++) {
        const date = new Date(2020, 0, 1)
        date.setMonth(date.getMonth() + i * 2)
        dates1.push(date.toISOString().slice(0, 7))
      }

      const actualCumulative = [0.5, 1, 3, 8, 15, 30, 50, 82]
      const modelEstimate = [0.5, 1.5, 4, 10, 20, 40, 50, 82]

      const dates2 = []
      const baseDate = new Date(2025, 6, 1)
      for (let i = 0; i < 7; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates2.push(date.toISOString().slice(0, 10))
      }

      const forecastValues = [87.4, 89.5, 91.2, 92.8, 94.2, 94.9, 95.2]
      const forecastUpper = [98, 97.5, 97, 96.5, 96, 95.5, 95]

      return {
        title: "Historical Cumulative Views: Actual vs Model Estimate",
        description: "Full historical timeline with model validation",
        data: dates1.map((d, i) => ({
          date: d,
          actual: actualCumulative[i] * 1e7,
          model: modelEstimate[i] * 1e7,
        })),
        metrics: { r2: 0.8317, mase: 0.2788, mape: 35.72, mae: 148825 },
      }
    }

    function generateYoutubeCumulativeForecast() {
      const baseDate = new Date(2025, 8, 1) // September 2025
      const dates = []
      for (let i = -6; i < 7; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates.push(date.toISOString().slice(0, 10))
      }

      const historical = [85.2e6, 85.8e6, 86.2e6, 86.5e6, 87.0e6, 87.4e6]
      const forecast = [87.9e6, 88.5e6, 89.1e6, 89.8e6, 90.4e6, 91.1e6, 91.7e6]
      const forecastUpper = [92e6, 93.1e6, 94.2e6, 95.4e6, 96.5e6, 97.8e6, 98.9e6]
      const forecastLower = [83.8e6, 83.9e6, 84e6, 84.2e6, 84.3e6, 84.4e6, 84.5e6]

      const dataPoints: any[] = []
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[i],
          historical: historical[i],
          forecast: null,
          upper: null,
          lower: null,
        })
      }
      for (let i = 0; i < 7; i++) {
        dataPoints.push({
          date: dates[6 + i],
          historical: null,
          forecast: forecast[i],
          upper: forecastUpper[i],
          lower: forecastLower[i],
        })
      }

      return {
        title: "6-Month Cumulative View Forecast (Last 6M History + Next 6M)",
        description: "with confidence range projection",
        data: dataPoints,
        metrics: { mape: 0, r2: 0.596, mase: 0.311, mae: 136798 },
      }
    }

    function generateYoutubeCatalog() {
      const dates = []
      const baseDate = new Date(2025, 4, 1)
      for (let i = -6; i < 7; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates.push(date.toISOString().slice(0, 7))
      }

      const historical = [100000, 120000, 150000, 200000, 250000, 300000]
      const forecast = [350000, 380000, 420000, 450000, 480000, 510000, 550000]
      const forecastUpper = [380000, 420000, 460000, 500000, 530000, 570000, 600000]
      const forecastLower = [320000, 350000, 390000, 420000, 450000, 480000, 520000]

      const dataPoints: any[] = []
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[i],
          historical: historical[i] * 10,
          forecast: null,
          upper: null,
          lower: null,
        })
      }
      for (let i = 0; i < 7; i++) {
        dataPoints.push({
          date: dates[6 + i],
          historical: null,
          forecast: forecast[i] * 10,
          upper: forecastUpper[i] * 10,
          lower: forecastLower[i] * 10,
        })
      }

      return {
        title: "Total Catalog Views: Historical Backcast + Forecast",
        description: "(baseline 6mo) with 70-130% confidence",
        data: dataPoints,
        currentValue: 3.5e7,
        projectedValue: 5.5e7,
        metrics: { mape: 36.6, r2: 0.8268, mase: 0.12, mae: 145113, confidenceRange: "70-130%" },
      }
    }

    function generateTiktokChannelViews() {
      const dates = []
      const baseDate = new Date(2025, 4, 1)
      for (let i = -6; i < 7; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates.push(date.toISOString().slice(0, 7))
      }

      const historical = [1e6, 1.2e6, 1.5e6, 1.3e6, 1.8e6, 1.5e6]
      const forecast = [1.6e6, 1.9e6, 2.1e6, 2.3e6, 2.5e6, 2.7e6, 2.8e6]
      const forecastUpper = [1.86e6, 2.2e6, 2.45e6, 2.68e6, 2.9e6, 3.13e6, 3.2e6]
      const forecastLower = [1.34e6, 1.58e6, 1.75e6, 1.92e6, 2.1e6, 2.27e6, 2.4e6]

      const dataPoints: any[] = []
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[i],
          historical: historical[i],
          forecast: null,
          upper: null,
          lower: null,
        })
      }
      for (let i = 0; i < 7; i++) {
        dataPoints.push({
          date: dates[6 + i],
          historical: null,
          forecast: forecast[i],
          upper: forecastUpper[i],
          lower: forecastLower[i],
        })
      }

      return {
        title: "Total Channel Views: Last 6 Months + 6-Month Forecast",
        description: "with MAPE confidence",
        data: dataPoints,
        metrics: { mape: 16.7, r2: 0.907, mase: 0.647, mae: 117725 },
      }
    }

    function generateTiktokPredictionAccuracy() {
      const actualRates = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
      const dataPoints: any[] = []

      for (const actual of actualRates) {
        const noise = (Math.random() - 0.5) * 6
        const predicted = Math.max(0, actual + noise)
        const distanceFromLine = Math.abs(predicted - actual)

        let color = "green"
        if (distanceFromLine > 6) color = "red"
        else if (distanceFromLine > 3) color = "orange"
        else if (distanceFromLine > 1.5) color = "yellow"

        dataPoints.push({ actual, predicted, color })
      }

      return {
        title: "Predicted vs Actual",
        description: "scatter plot (R²=0.303) with ±3% zone",
        data: dataPoints,
        metrics: { r2: 0.303, mae: 3.13, mase: 4.15, mape: 11.2, n: 77 },
      }
    }

    function generateTiktokCumulativeForecast() {
      const dates = []
      const baseDate = new Date(2025, 4, 1)
      for (let i = -6; i < 7; i++) {
        const date = new Date(baseDate)
        date.setMonth(date.getMonth() + i)
        dates.push(date.toISOString().slice(0, 7))
      }

      const cumulative = [0.8e6, 1.2e6, 2e6, 3e6, 4e6, 5e6]
      const forecast = [5.5e6, 6.2e6, 7e6, 7.8e6, 8.5e6, 9.2e6, 10e6]
      const forecastUpper = [6.3e6, 7.1e6, 8e6, 8.9e6, 9.7e6, 10.6e6, 11.5e6]
      const forecastLower = [4.7e6, 5.3e6, 6e6, 6.7e6, 7.3e6, 7.8e6, 8.5e6]

      const dataPoints: any[] = []
      for (let i = 0; i < 6; i++) {
        dataPoints.push({
          date: dates[i],
          cumulative: cumulative[i],
          forecast: null,
          upper: null,
          lower: null,
        })
      }
      for (let i = 0; i < 7; i++) {
        dataPoints.push({
          date: dates[6 + i],
          cumulative: null,
          forecast: forecast[i],
          upper: forecastUpper[i],
          lower: forecastLower[i],
        })
      }

      return {
        title: "Total Channel Views: Historical + 6-Month Forecast",
        description: "with MAPE 15.0% confidence",
        data: dataPoints,
        metrics: { mape: 15.0, r2: 0.84, mase: 4.9, mae: 120894 },
      }
    }

    const modelsData = {
      meta: {
        backtest: generateMetaBacktest(),
        reach6m: generateMetaReach6m(),
        existingPostsForecast: generateMetaExistingPosts(),
      },
      youtube: {
        cumulativeModel: generateYoutubeCumulative(),
        cumulativeForecast: generateYoutubeCumulativeForecast(),
        catalogViews: generateYoutubeCatalog(),
      },
      tiktok: {
        channelViews: generateTiktokChannelViews(),
        predictionAccuracy: generateTiktokPredictionAccuracy(),
        cumulativeForecast: generateTiktokCumulativeForecast(),
      },
    }

    console.log("[v0] Data generation complete")

    return NextResponse.json({
      status: "success",
      timestamp: new Date().toISOString(),
      data: modelsData,
    })
  } catch (error) {
    console.error("[v0] Error generating data:", error)
    const errorMessage = error instanceof Error ? error.message : String(error)

    return NextResponse.json(
      {
        status: "error",
        message: "Failed to generate predictive data",
        error: errorMessage,
        timestamp: new Date().toISOString(),
      },
      { status: 500 },
    )
  }
}
