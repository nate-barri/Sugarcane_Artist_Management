export function MetricsCard({
  title,
  metrics,
  variant = "blue",
}: {
  title: string
  metrics: any
  variant?: string
}) {
  if (!metrics) return null

  const bgColor = {
    blue: "bg-blue-50 border-blue-200",
    green: "bg-green-50 border-green-200",
    red: "bg-red-50 border-red-200",
    orange: "bg-orange-50 border-orange-200",
    yellow: "bg-yellow-50 border-yellow-200",
  }[variant]

  const textColor = {
    blue: "text-blue-700",
    green: "text-green-700",
    red: "text-red-700",
    orange: "text-orange-700",
    yellow: "text-yellow-700",
  }[variant]

  return (
    <div className={`${bgColor} border rounded-lg p-4 flex-1 min-w-[180px]`}>
      <h3 className={`font-bold text-sm mb-3 ${textColor}`}>{title}</h3>
      <div className="space-y-2">
        {metrics.mape !== undefined && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">MAPE:</span>
            <span className="font-semibold">{metrics.mape.toFixed(1)}%</span>
          </div>
        )}
        {metrics.r2 !== undefined && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">R²:</span>
            <span className="font-semibold">{metrics.r2.toFixed(3)}</span>
          </div>
        )}
        {metrics.mase !== undefined && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">MASE:</span>
            <span className="font-semibold">{metrics.mase.toFixed(3)}</span>
          </div>
        )}
        {metrics.mae !== undefined && (
          <div className="flex justify-between text-sm">
            <span className="text-gray-600">MAE:</span>
            <span className="font-semibold">
              {typeof metrics.mae === "number" && metrics.mae > 1000
                ? (metrics.mae / 1000).toFixed(0) + "K"
                : metrics.mae.toFixed(0)}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}
