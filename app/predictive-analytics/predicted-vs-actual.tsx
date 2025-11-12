"use client"

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Line,
  LineChart,
} from "recharts"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { useEffect, useState } from "react"

export function PredictedVsActual() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch("/api/analytics/predictive/engagement-scatter")
      .then((res) => res.json())
      .then((json) => {
        setData(json)
        setLoading(false)
      })
  }, [])

  if (loading) return <div>Loading chart...</div>

  if (!data) return null

  const { scatter, metrics } = data

  // Separate scatter points by color
  const withinZone = scatter.filter((p: any) => p.zone === "within")
  const outsideZone = scatter.filter((p: any) => p.zone === "outside")

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle>Predicted vs Actual (R²={metrics.r2.toFixed(3)})</CardTitle>
        <CardDescription>
          ±3% Zone | MAE: {metrics.mae.toFixed(2)}% | MASE: {metrics.mase.toFixed(2)}% | n={metrics.n}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={400}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" name="Actual Engagement Rate (%)" domain={[0, 25]} />
            <YAxis dataKey="y" name="Predicted Engagement Rate (%)" domain={[0, 25]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} />

            {/* Perfect prediction line (red dashed) */}
            <LineChart data={metrics.perfectLine}>
              <Line
                type="monotone"
                dataKey="y"
                stroke="#e74c3c"
                strokeDasharray="5 5"
                name="Perfect Prediction"
                dot={false}
              />
            </LineChart>

            <Scatter name="Within ±3% (Green)" data={withinZone} fill="#2ecc71" />
            <Scatter name="Outside Zone (Red/Orange)" data={outsideZone} fill="#e74c3c" />

            <Legend />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}
