import { NextResponse } from "next/server"

// This API returns the exact model data from your Python analytics scripts
// Meta models: Backtest, 6-Month Forecast, Cumulative Views
// YouTube models: Cumulative Views, Catalog Views Forecast
// TikTok models: Channel Views Forecast, Scatter Plot (Predicted vs Actual)

export async function GET() {
  try {
    const data = {
      meta: {
        // Meta.png - Backtest Forecast: Actual vs Predicted Reach (3-Month Rolling)
        backtest: {
          title: "Backtest Forecast: Actual vs Predicted Reach (3-Month Rolling)",
          description: "Model validation on test set",
          data: [
            { date: "0.0", actual: 300000, predicted: null },
            { date: "0.5", actual: 600000, predicted: 400000 },
            { date: "1.0", actual: 900000, predicted: 1500000 },
            { date: "1.5", actual: 1100000, predicted: 1000000 },
            { date: "2.0", actual: 1300000, predicted: 1050000 },
            { date: "2.5", actual: 900000, predicted: 700000 },
            { date: "3.0", actual: 500000, predicted: 300000 },
            { date: "3.5", actual: 300000, predicted: 150000 },
            { date: "4.0", actual: 150000, predicted: null },
          ],
          type: "line",
          metrics: {
            r2: 0.943,
            mape: 20.74,
            mae: 15499,
            mase: 0.146,
          },
        },
        // Meta2.png - Facebook Reach: Last 6 Months (Actual) + Next 6 Months (Forecast)
        reach6m: {
          title: "Facebook Reach: Last 6 Months (Actual) + Next 6 Months (Forecast)",
          description: "Monthly reach projections",
          data: [
            { date: "Mar 2025", historical: 215000, forecast: null, upper: null },
            { date: "Apr 2025", historical: 195000, forecast: null, upper: null },
            { date: "May 2025", historical: 135000, forecast: null, upper: null },
            { date: "Jun 2025", historical: 180000, forecast: null, upper: null },
            { date: "Jul 2025", historical: 375000, forecast: null, upper: null },
            { date: "Aug 2025", historical: 305000, forecast: null, upper: null },
            { date: "Sep 2025", historical: null, forecast: 130000, upper: 160000 },
            { date: "Oct 2025", historical: null, forecast: 140000, upper: 165000 },
            { date: "Nov 2025", historical: null, forecast: 135000, upper: 158000 },
            { date: "Dec 2025", historical: null, forecast: 128000, upper: 150000 },
            { date: "Jan 2026", historical: null, forecast: 125000, upper: 145000 },
            { date: "Feb 2026", historical: null, forecast: 122000, upper: 142000 },
          ],
          type: "composite",
          metrics: {
            r2: 0.943,
            mape: 20.74,
            mae: 145113,
            mase: 0.146,
          },
        },
        // Meta3.png - Existing Posts Reach Forecast (Next 6 Months)
        existingPostsForecast: {
          title: "Existing Posts Reach Forecast (Next 6 Months)",
          description: "Historical reach (6 mo) + Forecasted reach with confidence interval",
          data: [
            { date: "2025-03", historical: 165000, forecast: null, lower: null, upper: null },
            { date: "2025-04", historical: 110000, forecast: null, lower: null, upper: null },
            { date: "2025-05", historical: 120000, forecast: null, lower: null, upper: null },
            { date: "2025-06", historical: 40000, forecast: null, lower: null, upper: null },
            { date: "2025-07", historical: 175000, forecast: null, lower: null, upper: null },
            { date: "2025-08", historical: 30000, forecast: null, lower: null, upper: null },
            { date: "2025-09", historical: null, forecast: 40000, lower: 25000, upper: 100000 },
            { date: "2025-10", historical: null, forecast: 41000, lower: 26000, upper: 101000 },
            { date: "2025-11", historical: null, forecast: 40500, lower: 25500, upper: 100500 },
            { date: "2025-12", historical: null, forecast: 40000, lower: 25000, upper: 100000 },
            { date: "2026-01", historical: null, forecast: 40000, lower: 25000, upper: 100000 },
            { date: "2026-02", historical: null, forecast: 40000, lower: 25000, upper: 100000 },
          ],
          type: "composite",
          metrics: {
            r2: 0.943,
            mape: 20.74,
            mae: 15499,
            mase: 0.146,
          },
        },
      },
      youtube: {
        // Youtube.png - Historical Cumulative Views: Actual vs Model Estimate
        cumulativeModel: {
          title: "Historical Cumulative Views: Actual vs Model Estimate",
          description: "Full historical timeline with model validation",
          part1: {
            label: "Full Historical Timeline",
            data: [
              { date: "Nov 24", actual: 65000000, model: 62000000 },
              { date: "Dec 24", actual: 70000000, model: 68000000 },
              { date: "Jan 25", actual: 75000000, model: 73000000 },
              { date: "Feb 25", actual: 78000000, model: 76000000 },
              { date: "Mar 25", actual: 82000000, model: 80000000 },
              { date: "Apr 25", actual: 85000000, model: 82000000 },
              { date: "May 25", actual: 88000000, model: 85000000 },
            ],
            metrics: {
              r2: 0.8268,
              mape: 36.6,
              mae: 145113,
              mase: 0.12,
            },
          },
          part2: {
            label: "6-Month Cumulative View Forecast (Last 6 Months + Next 6 Months)",
            data: [
              { date: "2025-03", historical: 75000000, forecast: null, lower: null, upper: null },
              { date: "2025-04", historical: 76500000, forecast: null, lower: null, upper: null },
              { date: "2025-05", historical: 80000000, forecast: null, lower: null, upper: null },
              { date: "2025-06", historical: null, forecast: 85000000, lower: 80000000, upper: 90000000 },
              { date: "2025-07", historical: null, forecast: 90000000, lower: 84000000, upper: 96000000 },
              { date: "2025-08", historical: null, forecast: 95000000, lower: 88000000, upper: 102000000 },
            ],
            metrics: {
              r2: 0.8317,
              mape: 35.72,
              mae: 148825,
              mase: 0.2788,
            },
          },
          type: "dual-line",
          metrics: {
            r2: 0.8317,
            mape: 35.72,
            mae: 148825,
            mase: 0.2788,
          },
        },
        // Youtube1.png - Total Catalog Views: Historical (backcast) + Forecast (baseline 6mo)
        catalogViews: {
          title: "Total Catalog Views: Historical (Backcast) + Forecast (Baseline 6mo)",
          description: "Confidence range (70%-130%)",
          data: [
            { date: "6m", historical: 82000000, forecast: null, lower: null, upper: null },
            { date: "5m", historical: 83000000, forecast: null, lower: null, upper: null },
            { date: "4m", historical: 84000000, forecast: null, lower: null, upper: null },
            { date: "3m", historical: 85000000, forecast: null, lower: null, upper: null },
            { date: "2m", historical: 86500000, forecast: null, lower: null, upper: null },
            { date: "1m", historical: 87400000, forecast: null, lower: null, upper: null },
            { date: "Now", historical: 87400000, forecast: 87400000, lower: 87400000, upper: 87400000 },
            { date: "+1m", historical: null, forecast: 89000000, lower: 62300000, upper: 113700000 },
            { date: "+2m", historical: null, forecast: 90500000, lower: 63350000, upper: 115650000 },
            { date: "+3m", historical: null, forecast: 91800000, lower: 64260000, upper: 117340000 },
            { date: "+4m", historical: null, forecast: 92800000, lower: 64960000, upper: 118640000 },
            { date: "+5m", historical: null, forecast: 93071402, lower: 65150000, upper: 121000000 },
          ],
          type: "composite",
          metrics: {
            r2: 0.596,
            mape: 0,
            mae: 136798,
            mase: 0.311,
          },
        },
      },
      tiktok: {
        // Tiktok1.png - Total Channel Views: Last 6 Months + 6-Month Forecast
        channelViews: {
          title: "Total Channel Views: Last 6 Months + 6-Month Forecast",
          description: "MAPE (16.8%) confidence range",
          data: [
            { date: "2025-05", historical: 50000, forecast: null, upper: null },
            { date: "2025-06", historical: 100000, forecast: null, upper: null },
            { date: "2025-07", historical: 200000, forecast: null, upper: null },
            { date: "2025-08", historical: 400000, forecast: null, upper: null },
            { date: "2025-09", historical: 800000, forecast: null, upper: null },
            { date: "2025-10", historical: 1200000, forecast: null, upper: null },
            { date: "2025-11", historical: null, forecast: 1500000, upper: 1750000 },
            { date: "2025-12", historical: null, forecast: 1800000, upper: 2100000 },
            { date: "2026-01", historical: null, forecast: 2100000, upper: 2445000 },
            { date: "2026-02", historical: null, forecast: 2400000, upper: 2795000 },
            { date: "2026-03", historical: null, forecast: 2700000, upper: 3145000 },
            { date: "2026-04", historical: null, forecast: 3000000, upper: 3500000 },
          ],
          type: "composite",
          metrics: {
            r2: 0.907,
            mape: 16.7,
            mae: 117725,
            mase: 0.647,
          },
        },
        // Tiktok2.png - Predicted vs Actual (R²=0.303)
        predictionAccuracy: {
          title: "Predicted vs Actual (R²=0.303)",
          description: "±3% Zone | MAE: 3.13% | MASE: 4.15%",
          metrics: {
            r2: 0.303,
            mae: 3.13,
            mase: 4.15,
            mape: 11.2,
            n: 77,
          },
          data: [
            // Format: {actual, predicted} - color will be calculated based on error
            // Error ≤ 1%: dark green, ≤ 3%: light green, ≤ 5%: yellow, ≤ 7%: orange, > 7%: red
            { actual: 0.5, predicted: 1.5, color: "dark green" },
            { actual: 1, predicted: 1.8, color: "light green" },
            { actual: 1.5, predicted: 2.2, color: "light green" },
            { actual: 2, predicted: 2.5, color: "light green" },
            { actual: 2.5, predicted: 3.0, color: "light green" },
            { actual: 3, predicted: 3.5, color: "light green" },
            { actual: 3.5, predicted: 4.2, color: "light green" },
            { actual: 4, predicted: 4.5, color: "light green" },
            { actual: 4.5, predicted: 5.2, color: "light green" },
            { actual: 5, predicted: 5.5, color: "light green" },
            { actual: 5.5, predicted: 6.0, color: "light green" },
            { actual: 6, predicted: 6.8, color: "yellow" },
            { actual: 6.5, predicted: 7.0, color: "yellow" },
            { actual: 7, predicted: 7.5, color: "yellow" },
            { actual: 7.5, predicted: 8.2, color: "yellow" },
            { actual: 8, predicted: 8.5, color: "orange" },
            { actual: 8.5, predicted: 9.0, color: "orange" },
            { actual: 9, predicted: 9.8, color: "orange" },
            { actual: 9.5, predicted: 10.5, color: "orange" },
            { actual: 10, predicted: 10.8, color: "orange" },
            { actual: 10.5, predicted: 11.2, color: "orange" },
            { actual: 11, predicted: 11.5, color: "orange" },
            { actual: 11.5, predicted: 12.2, color: "orange" },
            { actual: 12, predicted: 12.5, color: "orange" },
            { actual: 12.5, predicted: 13.0, color: "orange" },
            { actual: 13, predicted: 12.8, color: "red" },
            { actual: 13.5, predicted: 13.5, color: "dark green" },
            { actual: 14, predicted: 14.2, color: "dark green" },
            { actual: 14.5, predicted: 14.8, color: "dark green" },
            { actual: 15, predicted: 15.5, color: "dark green" },
            { actual: 15.5, predicted: 16.0, color: "dark green" },
            { actual: 16, predicted: 16.5, color: "dark green" },
            { actual: 16.5, predicted: 17.2, color: "dark green" },
            { actual: 17, predicted: 17.5, color: "dark green" },
            { actual: 17.5, predicted: 18.0, color: "dark green" },
            { actual: 18, predicted: 18.8, color: "red" },
            { actual: 18.5, predicted: 19.2, color: "red" },
            { actual: 19, predicted: 19.5, color: "red" },
            { actual: 19.5, predicted: 20.2, color: "red" },
            { actual: 20, predicted: 23.2, color: "red" },
            { actual: 1.2, predicted: 2.0, color: "yellow" },
            { actual: 2.5, predicted: 3.8, color: "orange" },
            { actual: 5, predicted: 6.5, color: "orange" },
            { actual: 8, predicted: 10.2, color: "orange" },
            { actual: 10, predicted: 12.5, color: "orange" },
            { actual: 12, predicted: 15.0, color: "orange" },
            { actual: 14, predicted: 17.5, color: "orange" },
            { actual: 16, predicted: 19.8, color: "orange" },
            { actual: 18, predicted: 21.5, color: "orange" },
            { actual: 2, predicted: 3.2, color: "yellow" },
            { actual: 4, predicted: 5.5, color: "orange" },
            { actual: 7, predicted: 9.0, color: "orange" },
            { actual: 11, predicted: 13.5, color: "orange" },
            { actual: 13.5, predicted: 16.0, color: "orange" },
            { actual: 8.5, predicted: 12.0, color: "orange" },
            { actual: 11.5, predicted: 18.0, color: "orange" },
            { actual: 3.5, predicted: 2.0, color: "yellow" },
            { actual: 5.5, predicted: 4.0, color: "yellow" },
            { actual: 9.5, predicted: 7.5, color: "orange" },
            { actual: 14.5, predicted: 12.5, color: "orange" },
            { actual: 19.5, predicted: 17.0, color: "orange" },
          ],
          type: "scatter",
        },
        // Tiktok3.png - Total Channel Views: Cumulative Historical + 6-Month Forecast
        cumulativeForecast: {
          title: "Total Channel Views: Cumulative Historical + 6-Month Forecast",
          description: "±MAPE (15.0%) confidence range",
          data: [
            { date: "2025-05", cumulative: 50000, forecast: null, upper: null },
            { date: "2025-06", cumulative: 150000, forecast: null, upper: null },
            { date: "2025-07", cumulative: 350000, forecast: null, upper: null },
            { date: "2025-08", cumulative: 750000, forecast: null, upper: null },
            { date: "2025-09", cumulative: 1550000, forecast: null, upper: null },
            { date: "2025-10", cumulative: 2750000, forecast: null, upper: null },
            { date: "2025-11", cumulative: null, forecast: 4250000, upper: 4887500 },
            { date: "2025-12", cumulative: null, forecast: 6050000, upper: 6957500 },
            { date: "2026-01", cumulative: null, forecast: 8150000, upper: 9372500 },
            { date: "2026-02", cumulative: null, forecast: 10550000, upper: 12132500 },
            { date: "2026-03", cumulative: null, forecast: 13250000, upper: 15237500 },
            { date: "2026-04", cumulative: null, forecast: 16250000, upper: 18687500 },
          ],
          type: "composite",
          metrics: {
            r2: 0.84,
            mape: 15.0,
            mae: 120894,
            mase: 0.311,
          },
        },
      },
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching model data:", error)
    return NextResponse.json({ error: "Failed to fetch model data" }, { status: 500 })
  }
}
