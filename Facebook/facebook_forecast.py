"""
Facebook Reach Forecast (CSV)
- SARIMA and improved Prophet
- Handles NaNs and outliers
- Separate plots for SARIMA and Prophet
- Computes in-sample metrics
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Optional Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# -------------------------
# Settings
# -------------------------
CSV_PATH = "Facebook/facebook_data_set.csv"
AGG_FREQ = 'M'
FORECAST_MONTHS = 6
USE_LOG = True
SARIMA_GRID = {'p':[0,1],'d':[1],'q':[0,1],'P':[0,1],'D':[1],'Q':[0,1]}

# -------------------------
# Load CSV
# -------------------------
print("Loading data from CSV...")
df = pd.read_csv(CSV_PATH, parse_dates=['publish_time'])
print(f"Loaded {len(df)} rows")

# -------------------------
# Preprocess data
# -------------------------
df['duration_sec'] = df.apply(lambda x: x['duration_sec'] if x['post_type']=='video' else 0, axis=1)
df['reach'] = df['reach'].interpolate(method='linear')
df.set_index('publish_time', inplace=True)

# Monthly aggregation
monthly = df['reach'].resample(AGG_FREQ).mean()

# Smooth and clip outliers
monthly_smooth = monthly.rolling(6, center=True, min_periods=1).median()
upper, lower = monthly_smooth.quantile(0.95), monthly_smooth.quantile(0.05)
monthly_clean = monthly_smooth.clip(lower, upper)
print(f"Monthly series length: {len(monthly_clean)} from {monthly_clean.index.min().date()} to {monthly_clean.index.max().date()}")

# -------------------------
# SARIMA Forecast
# -------------------------
print("\nFitting SARIMA...")
train_series = np.log1p(monthly_clean) if USE_LOG else monthly_clean
s = 12  # monthly seasonality
best_aic, best_model = np.inf, None

for p,d,q in itertools.product(SARIMA_GRID['p'],SARIMA_GRID['d'],SARIMA_GRID['q']):
    for P,D,Q in itertools.product(SARIMA_GRID['P'],SARIMA_GRID['D'],SARIMA_GRID['Q']):
        try:
            model = SARIMAX(train_series, order=(p,d,q), seasonal_order=(P,D,Q,s),
                            enforce_stationarity=False, enforce_invertibility=False)
            res = model.fit(disp=False)
            if res.aic < best_aic:
                best_aic, best_model = res.aic, res
        except: continue

if best_model is not None:
    sarima_forecast = best_model.get_forecast(steps=FORECAST_MONTHS)
    sarima_mean = np.expm1(sarima_forecast.predicted_mean) if USE_LOG else sarima_forecast.predicted_mean
    sarima_ci = np.expm1(sarima_forecast.conf_int()) if USE_LOG else sarima_forecast.conf_int()
    forecast_index = pd.date_range(start=monthly.index[-1]+pd.offsets.MonthBegin(), periods=FORECAST_MONTHS, freq=AGG_FREQ)
    df_sarima = pd.DataFrame({'forecast': sarima_mean.values,
                              'lower_ci': sarima_ci.iloc[:,0].values,
                              'upper_ci': sarima_ci.iloc[:,1].values}, index=forecast_index)
    print("\nSARIMA Forecast:")
    print(df_sarima)

    # SARIMA in-sample metrics
    sarima_pred = best_model.fittedvalues
    if USE_LOG: sarima_pred = np.expm1(sarima_pred)
    actual = monthly_clean[sarima_pred.index]
    mae_pct = mean_absolute_error(actual, sarima_pred)/actual.mean()*100
    rmse_pct = np.sqrt(mean_squared_error(actual, sarima_pred))/actual.mean()*100
    smape = 100/len(actual)*np.sum(2*np.abs(sarima_pred-actual)/(np.abs(actual)+np.abs(sarima_pred)))
    r2 = r2_score(actual, sarima_pred)
    print("\nSARIMA In-Sample Accuracy Metrics:")
    print(f"MAE%: {mae_pct:.2f}% | RMSE%: {rmse_pct:.2f}% | SMAPE%: {smape:.2f}% | R²: {r2:.3f}")

# -------------------------
# Improved Prophet
# -------------------------
if PROPHET_AVAILABLE:
    print("\nRunning Improved Prophet forecast...")
    prophet_df = monthly_clean.reset_index().rename(columns={'publish_time':'ds','reach':'y'})
    
    # Log transform + smoothing
    if USE_LOG: prophet_df['y'] = np.log1p(prophet_df['y'])
    
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, seasonality_mode='multiplicative')
    m.add_seasonality(name='monthly', period=12, fourier_order=5)
    m.fit(prophet_df)

    future = m.make_future_dataframe(periods=FORECAST_MONTHS, freq='M')
    fcst = m.predict(future)
    if USE_LOG:
        fcst[['yhat','yhat_lower','yhat_upper']] = np.expm1(fcst[['yhat','yhat_lower','yhat_upper']])

    print("\nProphet Forecast:")
    print(fcst[['ds','yhat','yhat_lower','yhat_upper']].tail(FORECAST_MONTHS))

    # Prophet in-sample metrics
    prophet_pred = fcst.set_index('ds').loc[monthly_clean.index]['yhat']
    mae_pct = mean_absolute_error(monthly_clean, prophet_pred)/monthly_clean.mean()*100
    rmse_pct = np.sqrt(mean_squared_error(monthly_clean, prophet_pred))/monthly_clean.mean()*100
    smape = 100/len(monthly_clean)*np.sum(2*np.abs(prophet_pred-monthly_clean)/(np.abs(monthly_clean)+np.abs(prophet_pred)))
    r2 = r2_score(monthly_clean, prophet_pred)
    print("\nProphet In-Sample Accuracy Metrics:")
    print(f"MAE%: {mae_pct:.2f}% | RMSE%: {rmse_pct:.2f}% | SMAPE%: {smape:.2f}% | R²: {r2:.3f}")

# -------------------------
# Plotting
# -------------------------
plt.figure(figsize=(12,5))
plt.plot(monthly.index, monthly.values, label='Historical', marker='o')
if best_model is not None:
    plt.plot(df_sarima.index, df_sarima['forecast'], label='SARIMA Forecast', marker='s')
    plt.fill_between(df_sarima.index, df_sarima['lower_ci'], df_sarima['upper_ci'], color='gray', alpha=0.3)
plt.title("Facebook Reach - SARIMA Forecast")
plt.xlabel("Date"); plt.ylabel("Reach"); plt.legend(); plt.show()

if PROPHET_AVAILABLE:
    plt.figure(figsize=(12,5))
    plt.plot(monthly.index, monthly.values, label='Historical', marker='o')
    plt.plot(fcst['ds'], fcst['yhat'], label='Prophet Forecast', marker='^')
    plt.fill_between(fcst['ds'], fcst['yhat_lower'], fcst['yhat_upper'], color='orange', alpha=0.2)
    plt.title("Facebook Reach - Improved Prophet Forecast")
    plt.xlabel("Date"); plt.ylabel("Reach"); plt.legend(); plt.show()
