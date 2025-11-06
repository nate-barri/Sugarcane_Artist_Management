"""
================================================================================
EXISTING POSTS REACH FORECASTER
================================================================================
Forecasts reach for existing Facebook posts based on engagement history.
Now includes:
✅ Confidence interval (±1.96 × std of residuals)
✅ Only last 6 months of historical data
✅ 6-month future forecast visualization
✅ Printed forecasted values + total
================================================================================
"""

import psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# DATABASE CONNECTION
# ==============================================================================
print("\n" + "="*80)
print("EXISTING POSTS REACH FORECASTER")
print("="*80)
print("Connecting to database...")

db_params = {
    'dbname': 'neondb',
    'user': 'neondb_owner',
    'password': 'npg_dGzvq4CJPRx7',
    'host': 'ep-lingering-dawn-a410n0b8-pooler.us-east-1.aws.neon.tech',
    'port': '5432',
    'sslmode': 'require'
}

conn = psycopg2.connect(**db_params)

query = """
SELECT
    publish_time,
    reach,
    reactions AS likes,
    comments,
    shares,
    impressions,
    post_type
FROM facebook_data_set
WHERE publish_time IS NOT NULL AND reach IS NOT NULL
ORDER BY publish_time;
"""

df = pd.read_sql_query(query, conn)
conn.close()
print(f"✓ Loaded {len(df):,} existing posts\n")

# ==============================================================================
# DATA PREPROCESSING
# ==============================================================================
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['year'] = df['publish_time'].dt.year
df['month'] = df['publish_time'].dt.month

def remove_outliers(df):
    df['post_type'] = df['post_type'].fillna('Unknown')
    clean = []
    for t, g in df.groupby('post_type'):
        if len(g) > 0:
            threshold = g['reach'].quantile(0.95)
            clean.append(g[g['reach'] <= threshold])
    if len(clean) == 0:
        print("⚠ No valid post_type groups found, skipping outlier removal.")
        return df
    return pd.concat(clean, ignore_index=True)

df = remove_outliers(df)
print(f"✓ Cleaned dataset: {len(df):,} posts after outlier removal\n")

# Aggregate to monthly
monthly = (
    df.groupby([df['year'], df['month']])
      .agg({
          'reach': 'sum',
          'likes': 'sum',
          'comments': 'sum',
          'shares': 'sum',
          'impressions': 'sum'
      })
      .reset_index()
)
monthly['date'] = pd.to_datetime(monthly[['year', 'month']].assign(day=1))
monthly = monthly.sort_values('date')

# Keep only last 6 historical months for visualization
monthly = monthly.iloc[-6:]
print(f"✓ Showing last 6 months of historical data: {monthly['date'].min().strftime('%Y-%m')} → {monthly['date'].max().strftime('%Y-%m')}")

# ==============================================================================
# FEATURE ENGINEERING
# ==============================================================================
X = monthly[['likes', 'comments', 'shares', 'impressions']]
y = monthly['reach']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
split_idx = int(len(monthly) * 0.8)
X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# ==============================================================================
# MODEL TRAINING
# ==============================================================================
model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# Residuals for confidence intervals
residuals = y_test - preds
residual_std = np.std(residuals)

# ==============================================================================
# METRICS
# ==============================================================================
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
naive_pred = y_test.shift(1).fillna(y_test.mean())
mase = mae / np.mean(np.abs(y_test - naive_pred))

# --- Added MAPE ---
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

print("\n" + "="*80)
print("MODEL PERFORMANCE (Existing Posts)")
print("="*80)
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")
print(f"R²:   {r2:.3f}")
print(f"MASE: {mase:.3f}")
print(f"MAPE: {mape:.2f}%\n")  # <-- Added line

# ==============================================================================
# FORECASTING FUTURE REACH (6 MONTHS)
# ==============================================================================
future_months = 6
last_row = monthly.iloc[-1]
future_dates = pd.date_range(last_row['date'] + pd.offsets.MonthBegin(), periods=future_months, freq='MS')

future_data = pd.DataFrame({
    'likes': [last_row['likes']] * future_months,
    'comments': [last_row['comments']] * future_months,
    'shares': [last_row['shares']] * future_months,
    'impressions': [last_row['impressions']] * future_months
})

future_scaled = scaler.transform(future_data)
future_preds = model.predict(future_scaled)

# Confidence interval (±1.96 * residual_std)
ci_upper = future_preds + 1.96 * residual_std
ci_lower = future_preds - 1.96 * residual_std

future_df = pd.DataFrame({
    'date': future_dates,
    'forecasted_reach': future_preds,
    'upper_ci': ci_upper,
    'lower_ci': ci_lower
})

# ==============================================================================
# FORECAST OUTPUT TABLE
# ==============================================================================
print("="*80)
print("FORECASTED MONTHLY REACH (Next 6 Months)")
print("="*80)
for i, row in future_df.iterrows():
    print(f"{row['date'].strftime('%b %Y')}: {row['forecasted_reach']:,.0f} (CI: {row['lower_ci']:,.0f} – {row['upper_ci']:,.0f})")

total_forecast = future_df['forecasted_reach'].sum()
total_lower = future_df['lower_ci'].sum()
total_upper = future_df['upper_ci'].sum()

print("\n➡️ Total projected reach (6 months): {:,.0f}".format(total_forecast))
print("   95% CI range: {:,.0f} – {:,.0f}\n".format(total_lower, total_upper))

# ==============================================================================
# VISUALIZATION (Last 6 months + Forecast + Confidence Interval)
# ==============================================================================
plt.figure(figsize=(12,6))

# Combine historical and forecast for smooth transition
combined_dates = pd.concat([monthly['date'], future_df['date']])
combined_values = pd.concat([monthly['reach'], future_df['forecasted_reach']])

# Plot historical reach
plt.plot(monthly['date'], monthly['reach'], 'o-', label='Historical Reach (6 mo)', color='dodgerblue')

# Plot connecting dashed line from last historical to first forecast point
plt.plot(
    [monthly['date'].iloc[-1], future_df['date'].iloc[0]],
    [monthly['reach'].iloc[-1], future_df['forecasted_reach'].iloc[0]],
    linestyle='--', color='gray', alpha=0.8
)

# Plot forecasted reach
plt.plot(future_df['date'], future_df['forecasted_reach'], 'o--', label='Forecasted Reach', color='limegreen')

# Add confidence interval shading
plt.fill_between(future_df['date'], future_df['lower_ci'], future_df['upper_ci'],
                 color='limegreen', alpha=0.2, label='Confidence Interval')

# Title, labels, and legend
plt.title("Existing Posts Reach Forecast (Next 6 Months)")
plt.xlabel("Date")
plt.ylabel("Total Monthly Reach")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# ==============================================================================
# INTERPRETATION
# ==============================================================================
print("="*80)
print("INSIGHTS SUMMARY")
print("="*80)
if r2 < 0.5:
    print("❌ The model shows weak predictive power — reach trends vary heavily month-to-month.")
else:
    print("✅ The model demonstrates strong alignment with historical trends.")
print("Forecast includes a 95% confidence interval based on recent residual variation.")
print("Projected reach reflects engagement stability and post-type consistency.")
